#include "RP24_YOLO/OpenvinoInfer.h"

#include <algorithm>
#include <chrono>

OpenvinoInfer::OpenvinoInfer(string model_path_xml, string model_path_bin, string device,
                             int infer_threads, int num_streams){
    input_shape = {1, static_cast<unsigned long>(IMAGE_HEIGHT), static_cast<unsigned long>(IMAGE_WIDTH), 3};
    // 限制 CPU 推理并行度：默认会占满所有逻辑核（16）且 TBB arena 大量自旋空转。
    // 单流 + 少量线程，避免 TBB 空转成为最大 CPU 热点。
    // [实验结论] 8 线程单流实测反而更慢（~17ms vs 4 线程 ~10ms），
    // 小模型同步开销主导，单流加线程是死路。
    // 提速方向：多流(num_streams>1) + 多 InferRequest 并发推理（请求池）。
    // infer_threads 为总线程数 = 流数 × 每流线程数（如 8 = 2流 × 4线程）。
    const int n_requests = std::max(1, num_streams);
    try {
        core.set_property("CPU", ov::inference_num_threads(infer_threads));
        core.set_property("CPU", ov::num_streams(n_requests));
    } catch (const std::exception& e) {
        std::cerr << "[OpenvinoInfer] set_property warning: " << e.what() << std::endl;
    }
    model = core.read_model(model_path_xml, model_path_bin);
    // Step . Inizialize Preprocessing for the model
    ppp = new ov::preprocess::PrePostProcessor(model);
    // Specify input image format
    ppp->input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR); 
    //NHWC:batchsize,height,width,channels
    // Specify preprocess pipeline to input image without resizing
    ppp->input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});
    //  Specify model's input layout
    ppp->input().model().set_layout("NCHW");
    // Specify output results format
    ppp->output().tensor().set_element_type(ov::element::f32);
    // Embed above steps in the graph
    model = ppp->build();

    compiled_model = core.compile_model(model, device);

    // 请求池：每请求独立输出缓冲，可并发推理（多帧并行）。流数<=1 时退化为单请求串行。
    num_requests_ = n_requests;
    infer_requests_.reserve(num_requests_);
    for (int i = 0; i < num_requests_; ++i) {
        infer_requests_.push_back(compiled_model.create_infer_request());
    }
    request_busy_.assign(num_requests_, 0);
}

std::vector<Object> OpenvinoInfer::infer(const cv::Mat& img, int detect_color,
                                         double* wait_ms, double* infer_ms)
{
    // 防御：经 header 内联构造函数创建的对象也要保证有请求可用
    if (infer_requests_.empty()) {
        std::lock_guard<std::mutex> lk(request_mtx_);
        if (infer_requests_.empty()) {
            num_requests_ = 1;
            infer_requests_.push_back(compiled_model.create_infer_request());
            request_busy_.assign(1, 0);
        }
    }

    // 1. 领一个空闲 InferRequest（都忙则等待；多请求 = 多帧并行推理）
    const auto wait_t0 = std::chrono::steady_clock::now();
    ov::InferRequest req;
    int slot = -1;
    {
        std::unique_lock<std::mutex> lk(request_mtx_);
        request_cv_.wait(lk, [this, &slot]() {
            for (int i = 0; i < num_requests_; ++i) {
                if (!request_busy_[i]) {
                    slot = i;
                    request_busy_[i] = 1;
                    return true;
                }
            }
            return false;
        });
        req = infer_requests_[slot];
    }
    if (wait_ms) {
        *wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wait_t0).count();
    }

    const auto infer_t0 = std::chrono::steady_clock::now();

    // 2. 推理 + 解码（全部局部变量，不写共享成员；输出缓冲属于 req，互不干扰）
    // 新模型格式（Ultralytics FasterNet-P345_pose，onnxruntime QDQ INT8）：
    //   输出 [1, 29, N]（N=8400@640 / 5376@512），channels-first：
    //   0-3            box cx,cy,w,h（输入像素空间，已 DFL 解码）
    //   4..4+nc-1      17 类得分（logits，需 sigmoid；无 objectness，取最大类得分做置信度）
    //   4+nc..4+nc+7   4 关键点 x,y（输入像素空间，已解码）
    //   类别 0-8 = 蓝色(B)，9-16 = 红色(R)，颜色内嵌于类别，无独立颜色通道
    std::vector<Object> objects;
    std::vector<Object> tmp_objects;

    uchar* input_data = (uchar*)img.data;
    ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
    req.set_input_tensor(input_tensor);
    req.infer();

    auto output = req.get_output_tensor(0);
    ov::Shape output_shape = output.get_shape();
    // output_buffer: rows=通道数(29)，cols=锚点数(8400) —— 注意与旧模型(锚点优先)相反
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output.data());
    const int n_anchors  = output_buffer.cols;
    const int n_channels = output_buffer.rows;
    const int nc = n_channels - 4 - 8;   // 类别数 = 29 - 4 - 8 = 17
    const int kClsStart   = 4;
    const int kKptStart   = 4 + nc;
    constexpr int kBlueClasses = 9;      // 类别 0-8 蓝色，9-16 红色
    float conf_threshold = 0.65;
    float nms_threshold  = 0.45;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    for (int j = 0; j < n_anchors; ++j) {
        // 无 objectness 分支：取最大类得分（sigmoid 后）作为置信度
        float max_score = -1e9f;
        int best_cls = -1;
        for (int c = 0; c < nc; ++c) {
            float s = sigmoid(output_buffer.at<float>(kClsStart + c, j));
            if (s > max_score) { max_score = s; best_cls = c; }
        }
        if (best_cls < 0 || max_score < conf_threshold) continue;

        // 敌方颜色过滤：类别前缀 B(0-8)/R(9-16)；detect_color: 0=敌方蓝 1=敌方红
        const bool is_blue = (best_cls < kBlueClasses);
        if (detect_color == 0 && !is_blue) continue;
        if (detect_color == 1 && is_blue)  continue;

        const float cx = output_buffer.at<float>(0, j);
        const float cy = output_buffer.at<float>(1, j);
        const float w  = output_buffer.at<float>(2, j);
        const float h  = output_buffer.at<float>(3, j);

        Object obj;
        obj.prob  = max_score;
        obj.color = is_blue ? 1 : 0;   // 沿用旧约定：blue:1, red:0
        obj.label = best_cls;
        for (int k = 0; k < 8; ++k) {
            obj.landmarks[k] = output_buffer.at<float>(kKptStart + k, j);
        }
        // 关键点顺序假设与旧模型一致：[左灯条上, 左灯条下, 右灯条上, 右灯条下]
        // （若新数据集标注顺序不同，只需调整这里的取点下标）
        obj.length = cv::norm(cv::Point2f(obj.landmarks[0] - obj.landmarks[2],
                                          obj.landmarks[1] - obj.landmarks[3]));
        obj.width  = cv::norm(cv::Point2f(obj.landmarks[0] - obj.landmarks[4],
                                          obj.landmarks[1] - obj.landmarks[5]));
        obj.ratio  = obj.length / obj.width;

        // box 输出 cxcywh（输入空间）→ 左上角 + 宽高
        cv::Rect rect((int)(cx - w * 0.5f), (int)(cy - h * 0.5f), (int)w, (int)h);
        obj.rect = rect;
        objects.push_back(obj);
        boxes.push_back(rect);
        confidences.push_back(max_score);
    }

    // NMS（输入空间坐标）
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    for (int valid_index : indices) {
        if (valid_index < (int)objects.size()) {
            tmp_objects.push_back(objects[valid_index]);
        }
    }

    // 3. 归还请求
    {
        std::lock_guard<std::mutex> lk(request_mtx_);
        request_busy_[slot] = 0;
    }
    request_cv_.notify_one();

    if (infer_ms) {
        *infer_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - infer_t0).count();
    }

    return tmp_objects;
}
