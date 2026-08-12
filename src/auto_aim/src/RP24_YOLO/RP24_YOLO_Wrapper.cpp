#include "RP24_YOLO/RP24_YOLO_Wrapper.h"
#include "pipeline/AutoAimPipeline.h"
#include "utils/ThreadPool.h"

std::pair<string, string> convertOnnxToIR(const string& onnx_path) {
    ov::Core core;
    cout << "[INFO] Loading ONNX model: " << onnx_path << endl;
    auto model = core.read_model(onnx_path);

    string base_path = onnx_path;
    // 去掉 .onnx 后缀
    size_t dot_pos = base_path.rfind(".onnx");
    if (dot_pos != string::npos) {
        base_path = base_path.substr(0, dot_pos);
    }

    string xml_path = base_path + ".xml";
    string bin_path = base_path + ".bin";

    cout << "[INFO] Serializing model to IR format..." << endl;
    ov::serialize(model, xml_path, bin_path);
    cout << "[INFO] IR files generated: " << xml_path << ", " << bin_path << endl;

    return {xml_path, bin_path};
}

RP24YOLOWrapper::RP24YOLOWrapper(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node, string model_path, string device) 
    : config_file_ptr(config_file_ptr), node(node) {
    fix_armor_class_ =
        (*config_file_ptr)["FIX_ARMOR_CLASS"] ? (*config_file_ptr)["FIX_ARMOR_CLASS"].as<int>() : -1;

    // -------------------- Step 1: 模型转换（ONNX -> IR） --------------------
    // 检查模型文件是否存在
    if (FILE* f = fopen(model_path.c_str(), "r")) {
        fclose(f);
    } else {
        cerr << "[ERROR] Model file not found: " << model_path << endl;
        throw runtime_error("Model file not found");
    }

    // 检查是否已经有同名的 .xml 文件（避免重复转换）
    string base_path = model_path;
    size_t dot_pos = base_path.rfind(".onnx");
    if (dot_pos != string::npos) base_path = base_path.substr(0, dot_pos);
    string xml_path_str = base_path + ".xml";
    string bin_path_str = base_path + ".bin";

    // 检查 .xml 和 .bin 是否都已存在
    bool need_convert = true;
    FILE* f_xml = fopen(xml_path_str.c_str(), "r");
    FILE* f_bin = fopen(bin_path_str.c_str(), "r");
    if (f_xml && f_bin) {
        need_convert = false;
        cout << "[INFO] IR files already exist, skipping conversion." << endl;
    }
    if (f_xml) fclose(f_xml);
    if (f_bin) fclose(f_bin);

    if (need_convert) {
        auto [xml_path, bin_path] = convertOnnxToIR(model_path);
        xml_path_str = xml_path;
        bin_path_str = bin_path;
        cout << "[INFO] Model converted to IR format successfully!" << endl;
    }

    // -------------------- Step 2: 初始化推理器 --------------------
    // 使用 OpenvinoInfer 的第一个构造函数
    // （参考 OpenvinoInfer.cpp 中的实现：BGR输入 -> RGB -> 归一化 -> NCHW）
    openvino_infer = std::make_shared<OpenvinoInfer>(xml_path_str, bin_path_str, device);
    cout << "[INFO] Inference model loaded successfully!" << endl;

    armor_tracker = std::make_shared<ArmorTracker>(config_file_ptr, node);

    // 检测流水线由线程池驱动：每帧一个任务，池内多帧并行；
    // 推理段（OpenvinoInfer 非线程安全）由 infer_mutex_ 串行保护
    cout << "[INFO] YOLO async pipeline started (thread pool)" << endl;
}

RP24YOLOWrapper::~RP24YOLOWrapper()
{
    stop();
}

void RP24YOLOWrapper::stop()
{
    {
        std::lock_guard<std::mutex> lock(results_mtx_);
        if (stopping_.exchange(true)) {
            return;  // 幂等
        }
    }
    // 等待已在飞的任务完成（preprocess/infer/postprocess 结束后 pending 归零）
    std::unique_lock<std::mutex> lock(results_mtx_);
    results_cv_.wait(lock, [this]() { return pending_tasks_.load() == 0; });
    results_.clear();
    results_cv_.notify_all();
}

uint64_t RP24YOLOWrapper::submitFrame(cv::Mat frame, int detect_color, void* user_data)
{
    uint64_t frame_id = next_frame_id_.fetch_add(1);
    if (stopping_.load()) {
        return frame_id;  // 停止中，丢弃新帧
    }

    auto work = std::make_shared<YoloWork>();
    work->frame_id = frame_id;
    work->user_data = user_data;
    work->frame = std::move(frame);  // cv::Mat 移动：只搬引用计数头，代价极小
    work->detect_color = detect_color;

    pending_tasks_.fetch_add(1);
    try {
        utils::threadPool().submit([this, work]() { processOneFrame(work); });
    } catch (const std::exception&) {
        // 池已停止等极端情况：不处理该帧
        pending_tasks_.fetch_sub(1);
    }
    return frame_id;
}

bool RP24YOLOWrapper::tryTakeResult(YoloResult* out)
{
    std::lock_guard<std::mutex> lock(results_mtx_);
    if (results_.empty()) {
        return false;
    }
    *out = std::move(results_.front());
    results_.pop_front();
    return true;
}

RP24YOLOWrapper::YoloResult RP24YOLOWrapper::takeResult(uint64_t frame_id)
{
    YoloResult result;
    std::unique_lock<std::mutex> lock(results_mtx_);
    results_cv_.wait(lock, [this]() { return stopping_.load() || !results_.empty(); });
    if (results_.empty()) {
        return result;
    }
    result = std::move(results_.front());
    results_.pop_front();
    if (result.frame_id != frame_id) {
        // 最新数据优先语义下，请求的帧可能已被更新的帧覆盖，属正常情况
        RCLCPP_DEBUG(node->get_logger(),
            "YOLO result superseded: expect %llu, got %llu",
            (unsigned long long)frame_id, (unsigned long long)result.frame_id);
    }
    return result;
}

void RP24YOLOWrapper::processOneFrame(std::shared_ptr<YoloWork> work)
{
    try {
        // 1. preprocess（池内多帧可并行）
        auto stage_start = std::chrono::steady_clock::now();
        work->infer_input = preprocess(work->frame);
        double stage_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - stage_start).count();
        if (work->user_data != nullptr) {
            auto* pd = static_cast<AutoAimPipelineData*>(work->user_data);
            if (pd->initial.performance_profile) {
                pd->initial.performance_profile->stages["yolo_preprocess"] += stage_ms;
            }
        }

        // 2. infer（OpenvinoInfer 非线程安全，串行保护）
        stage_start = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(infer_mutex_);
            work->objects = infer(work->infer_input, work->detect_color);
        }
        stage_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - stage_start).count();
        if (work->user_data != nullptr) {
            auto* pd = static_cast<AutoAimPipelineData*>(work->user_data);
            if (pd->initial.performance_profile) {
                pd->initial.performance_profile->stages["yolo_infer"] += stage_ms;
            }
        }

        // 3. postprocess（池内多帧可并行）
        stage_start = std::chrono::steady_clock::now();
        work->armors = postprocess(work->frame, work->objects, &work->rp24_classes);
        stage_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - stage_start).count();
        if (work->user_data != nullptr) {
            auto* pd = static_cast<AutoAimPipelineData*>(work->user_data);
            if (pd->initial.performance_profile) {
                pd->initial.performance_profile->stages["yolo_postprocess"] += stage_ms;
            }
        }

        // 4. 结果入队（最新优先，有界，满则丢最旧）
        YoloResult result;
        result.frame_id = work->frame_id;
        result.user_data = work->user_data;
        result.armors = std::move(work->armors);
        result.rp24_classes = std::move(work->rp24_classes);
        {
            std::lock_guard<std::mutex> lock(results_mtx_);
            if (!stopping_.load()) {
                if (results_.size() >= kMaxResults) {
                    results_.pop_front();
                }
                results_.push_back(std::move(result));
            }
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "YOLO processOneFrame exception: %s", e.what());
    }
    // 无论成功/异常都要归还任务计数并通知（stop 依赖 pending 归零）
    {
        std::lock_guard<std::mutex> lock(results_mtx_);
        pending_tasks_.fetch_sub(1);
    }
    results_cv_.notify_all();
}

cv::Mat RP24YOLOWrapper::preprocess(const cv::Mat& frame) {
    // 1. 缩放到 640x640 进行推理（模型输入要求 640x640）
    cv::Mat infer_frame;
    cv::resize(frame, infer_frame, cv::Size(640, 640));
    return infer_frame;
}

vector<Object> RP24YOLOWrapper::infer(const cv::Mat& input, int detect_color) {
    openvino_infer -> infer(input, detect_color);
    return openvino_infer -> tmp_objects;
}

vector<Armor> RP24YOLOWrapper::postprocess(const cv::Mat& frame, const vector<Object>& objects, vector<int>* rp24_classes) {
    // 2. 将检测结果的坐标从 640x640 映射回原图尺寸
    float scale_x = (float)frame.cols / 640.0f;
    float scale_y = (float)frame.rows / 640.0f;
    int img_w = frame.cols, img_h = frame.rows;
    vector<Object> objects_scaled = objects;
    for (auto& obj : objects_scaled) {
        obj.rect.x      = (int)(obj.rect.x * scale_x);
        obj.rect.y      = (int)(obj.rect.y * scale_y);
        obj.rect.width  = (int)(obj.rect.width * scale_x);
        obj.rect.height = (int)(obj.rect.height * scale_y);
        for (int i = 0; i < 8; i += 2) {
            obj.landmarks[i]   *= scale_x;
            obj.landmarks[i+1] *= scale_y;
        }
        obj.length *= scale_x;
        obj.width  *= scale_y;

        // // 钳位 rect 到图像范围内，防止绘制时越界
        // obj.rect.x = std::max(0.0f, obj.rect.x);
        // obj.rect.y = std::max(0.0f, obj.rect.y);
        // obj.rect.width  = std::min(obj.rect.width,  img_w - obj.rect.x);
        // obj.rect.height = std::min(obj.rect.height, img_h - obj.rect.y);
    }

    vector<Armor> armors;

    for (Object& object : objects_scaled) {
        std::vector<float> frame_keypoints(object.landmarks, object.landmarks + 8);

        RCLCPP_DEBUG(node->get_logger(), "scaled_yolo_data: %f, %f, %f, %f, %f, %f, %f, %f",
            frame_keypoints[0], frame_keypoints[1], frame_keypoints[2], frame_keypoints[3], 
            frame_keypoints[4], frame_keypoints[5], frame_keypoints[6], frame_keypoints[7]);
        cv::Vec2f leftLightBar_lengthVec((frame_keypoints[0] - frame_keypoints[2]), (frame_keypoints[1] - frame_keypoints[3]));
        cv::Vec2f rightLightBar_lengthVec((frame_keypoints[6] - frame_keypoints[4]), (frame_keypoints[7] - frame_keypoints[5]));
        cv::Point2f leftLightBar_center((frame_keypoints[0] + frame_keypoints[2]) / 2.0, (frame_keypoints[1] + frame_keypoints[3]) / 2.0);
        cv::Point2f rightLightBar_center((frame_keypoints[4] + frame_keypoints[6]) / 2.0, (frame_keypoints[5] + frame_keypoints[7]) / 2.0);
        float leftLightBar_length = cv::norm(leftLightBar_lengthVec);
        float rightLightBar_length = cv::norm(rightLightBar_lengthVec);
        cv::Size2f leftLightBar_size(leftLightBar_length * lightBarLengthScale / 8.0, leftLightBar_length * lightBarLengthScale);
        cv::Size2f rightLightBar_size(rightLightBar_length * lightBarLengthScale / 8.0, rightLightBar_length * lightBarLengthScale);
        float leftLightBar_angle = std::atan2(leftLightBar_lengthVec[1], leftLightBar_lengthVec[0]) * 180.0 / M_PI + 90.0;
        float rightLightBar_angle = std::atan2(rightLightBar_lengthVec[1], rightLightBar_lengthVec[0]) * 180.0 / M_PI + 90.0;

        cv::RotatedRect leftLightBar(leftLightBar_center, leftLightBar_size, leftLightBar_angle);
        cv::RotatedRect rightLightBar(rightLightBar_center, rightLightBar_size, rightLightBar_angle);
        armors.emplace_back(leftLightBar, rightLightBar, config_file_ptr, node);

        if (rp24_classes != nullptr) {
            rp24_classes->push_back(object.label);
        }
    }

    return armors;
}

vector<Armor> RP24YOLOWrapper::detectArmors(cv::Mat& frame, string detect_color, vector<int>* rp24_classes) {
    static bool first_call = true;
    if (first_call) {
        first_call = false;
        std::cerr << "[diag] detectArmors called (async path active)" << std::endl;
    }
    int detect_color_int = (detect_color == "BLUE") ? 0 : ((detect_color == "RED") ? 1 : -1);

    // 提交到三阶段流水线，阻塞等待该帧结果
    uint64_t frame_id = submitFrame(frame, detect_color_int);
    YoloResult result = takeResult(frame_id);

    if (rp24_classes != nullptr) {
        *rp24_classes = result.rp24_classes;
    }
    return result.armors;
}

vector<ArmorResult> RP24YOLOWrapper::detectArmorsWithClassifyAndTrack(cv::Mat& frame, string detect_color, 
        const cv::Point2f& ground_stable_point, vector<Armor>* armors_out) {

    vector<int> rp24_classes;
    vector<Armor> armors = detectArmors(frame, detect_color, &rp24_classes);

    vector<ArmorResult> results = classifyAndTrack(armors, rp24_classes, ground_stable_point);
    if (armors_out != nullptr) {
        *armors_out = std::move(armors);
    }
    return results;
}

vector<ArmorResult> RP24YOLOWrapper::classifyAndTrack(
    vector<Armor> armors, const vector<int>& rp24_classes,
    const cv::Point2f& ground_stable_point)
{
    armor_tracker -> preProcess(ground_stable_point);
    for (size_t i = 0; i < armors.size(); i++) {
        Armor& armor = armors[i];
        int number = class_map[rp24_classes[i]];
        bool is_large = big_map[rp24_classes[i]];
        if (fix_armor_class_ >= 0) {
            number = fix_armor_class_;
        }
        bool not_slant = true;
        float confidence = armor.confidence;

        armor_tracker -> addArmor(armor, number, is_large, not_slant, confidence);
    }

    return armor_tracker -> afterProcess();
}

