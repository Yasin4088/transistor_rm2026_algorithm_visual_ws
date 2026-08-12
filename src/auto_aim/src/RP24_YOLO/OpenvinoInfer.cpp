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
    std::vector<Object> objects;
    std::vector<Object> tmp_objects;

    uchar* input_data = (uchar*)img.data;
    ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
    req.set_input_tensor(input_tensor);
    req.infer();

    auto output = req.get_output_tensor(0);
    ov::Shape output_shape = output.get_shape();
    // 25200 x 85 Matrix
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output.data());
    float conf_threshold = 0.65 ;
    float nms_threshold = 0.45;
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<float> confidences;
    // cx,cy,w,h,confidence,c1,c2,...c80
    for (int i = 0; i < output_buffer.rows; i++) {
        //通过置信度阈值筛选
        float confidence = output_buffer.at<float>(i, 8);
        confidence = sigmoid(confidence);
        if (confidence < conf_threshold)
        {
            continue;
        }
        //颜色和类别独热向量
        cv::Mat color_scores = output_buffer.row(i).colRange(9, 13);  //color
        cv::Mat classes_scores = output_buffer.row(i).colRange(13, 22); //num
        cv::Point class_id,color_id;
        int _class_id,_color_id;
        double score_color, score_num;
        cv::minMaxLoc(classes_scores, NULL, &score_num, NULL, &class_id);
        cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);
        // cout<<score_color<<" "<<color_id.x<<endl;
        // cout<<score_num<<" "<<class_id.x<<endl;
        // class score: 0~3
    //    cout<<"class_id.x:"<<class_id.x<<endl;
    //    cout<<"detect_color:"<<detect_color<<endl;
        // None 或者Purple 丢掉
        if(color_id.x == 2 || color_id.x == 3)
        {
            continue;
        }
        else if(detect_color == 0 && color_id.x == 1)   // detect blue
        {
            continue;
        }
        else if(detect_color == 1 && color_id.x == 0)   // detect red
        {
            continue;
        }
        _class_id = class_id.x;
        _color_id = color_id.x;
        Object obj;
        obj.prob = confidence;
        obj.color = _color_id;
        obj.label = _class_id;
        obj.landmarks[0]=output_buffer.at<float>(i, 0);
        obj.landmarks[1]=output_buffer.at<float>(i, 1);
        obj.landmarks[2]=output_buffer.at<float>(i, 2);
        obj.landmarks[3]=output_buffer.at<float>(i, 3);
        obj.landmarks[4]=output_buffer.at<float>(i, 4);
        obj.landmarks[5]=output_buffer.at<float>(i, 5);
        obj.landmarks[6]=output_buffer.at<float>(i, 6);
        obj.landmarks[7]=output_buffer.at<float>(i, 7);
        obj.length = cv::norm(cv::Point2f(obj.landmarks[0] - obj.landmarks[6])-cv::Point2f(obj.landmarks[1]-obj.landmarks[7]));
        obj.width = cv::norm(cv::Point2f(obj.landmarks[0] - obj.landmarks[2])-cv::Point2f(obj.landmarks[1]-obj.landmarks[3]));
        obj.ratio = obj.length / obj.width;

        std::vector<cv::Point2f> points;
        //landmarks为左上逆时针，points应为左上顺时针
        points.push_back(cv::Point2f(obj.landmarks[0], obj.landmarks[1]));
        points.push_back(cv::Point2f(obj.landmarks[6], obj.landmarks[7]));
        points.push_back(cv::Point2f( obj.landmarks[4], obj.landmarks[5]));
        points.push_back(cv::Point2f( obj.landmarks[2], obj.landmarks[3]));

        // Find the minimum and maximum x and y coordinates
        float min_x = points[0].x;
        float max_x = points[0].x;
        float min_y = points[0].y;
        float max_y = points[0].y;

        for (int i = 1; i < points.size(); i++)
        {
            if (points[i].x < min_x)
                min_x = points[i].x;
            if (points[i].x > max_x)
                max_x = points[i].x;
            if (points[i].y < min_y)
                min_y = points[i].y;
            if (points[i].y > max_y)
                max_y = points[i].y;
        }

        // Create the rectangle
        cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);
        obj.rect = rect;
        objects.push_back(obj);
        boxes.push_back(rect);
        confidences.push_back(score_num);
    }
    // NMS
//        std::cout<<"object_size: "<<objects.size()<<endl;
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    int index = 0, index_indices = 0;
    for(int valid_index:indices){
        if(valid_index <= objects.size()){
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
