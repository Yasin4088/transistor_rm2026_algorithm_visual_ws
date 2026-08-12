#ifndef RP24_YOLO_WRAPPER_H
#define RP24_YOLO_WRAPPER_H

#include "RP24_YOLO/OpenvinoInfer.h"
#include "2d_armor_detector/Armor.h"
#include "2d_armor_detector/ArmorTracker.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <utility>

class RP24YOLOWrapper {
public:
    // 异步流水线的一帧结果
    struct YoloResult {
        uint64_t frame_id = 0;
        void* user_data = nullptr;      // 调用方自定义数据（Stage1 用它带回数据指针）
        vector<Armor> armors;
        vector<int> rp24_classes;
    };

    RP24YOLOWrapper(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node, string model_path, string device);
    ~RP24YOLOWrapper();

    vector<Armor> detectArmors(cv::Mat& frame, string detect_color, vector<int>* rp24_classes = nullptr);
    vector<ArmorResult> detectArmorsWithClassifyAndTrack(cv::Mat& frame, string detect_color, 
        const cv::Point2f& ground_stable_point, vector<Armor>* armors_out = nullptr);

    // 异步接口：submit 立即返回，结果按提交顺序用 takeResult 取回
    uint64_t submitFrame(cv::Mat frame, int detect_color, void* user_data = nullptr);
    YoloResult takeResult(uint64_t frame_id);
    // 停止流水线（幂等）：关闭队列/寄存器并 join 三个线程
    void stop();
    bool tryTakeResult(YoloResult* out);   // 非阻塞取结果：有则 true，无则 false
    // 结果就绪回调：有结果推入 results_ 时触发（事件唤醒，替代调用方轮询）
    void setResultNotify(std::function<void()> notify);
    vector<ArmorResult> classifyAndTrack(vector<Armor> armors, const vector<int>& rp24_classes,
                                         const cv::Point2f& ground_stable_point);

private:
    // 一帧在流水线中流动的数据
    struct YoloWork {
        uint64_t frame_id = 0;
        void* user_data = nullptr;
        cv::Mat frame;              // 原图（postprocess 缩放用）
        int detect_color = -1;
        cv::Mat infer_input;        // preprocess 输出
        vector<Object> objects;     // infer 输出
        vector<int> rp24_classes;   // postprocess 输出
        vector<Armor> armors;       // postprocess 输出
    };

    // ---------- 阶段函数（原有逻辑，保持不变） ----------
    cv::Mat preprocess(const cv::Mat& frame);
    vector<Object> infer(const cv::Mat& input, int detect_color);
    vector<Armor> postprocess(const cv::Mat& frame, const vector<Object>& objects, vector<int>* rp24_classes);

    // 线程池任务：一帧完整处理（preprocess -> infer -> postprocess -> 结果入队）
    void processOneFrame(std::shared_ptr<YoloWork> work);
    // 结果就绪时触发 result_notify_（锁外调用回调）
    void notifyResultAvailable();

    // ---------- 线程池流水线成员 ----------
    static constexpr size_t kMaxResults = 8;   // 结果队列上限（最新优先，满则丢最旧）
    std::deque<YoloResult> results_;           // 已完成的帧结果
    std::mutex results_mtx_;
    std::condition_variable results_cv_;
    std::function<void()> result_notify_;      // 受 results_mtx_ 保护
    std::mutex infer_mutex_;                   // OpenvinoInfer 非线程安全，推理串行
    std::atomic<size_t> pending_tasks_{0};     // 在飞任务数（stop 时等待归零）
    std::atomic<bool> stopping_{false};
    std::atomic<uint64_t> next_frame_id_{0};

    std::shared_ptr<OpenvinoInfer> openvino_infer;
    std::shared_ptr<YAML::Node> config_file_ptr;
    rclcpp::Node* node;
    float lightBarLengthScale = 0.82;

    int class_map[9] = {5, 0, 1, 2, 3, 4, 6, 7, 7};
    bool big_map[9] = {false, true, false, false, false, false, false, false, true};
    std::shared_ptr<ArmorTracker> armor_tracker;
    int fix_armor_class_ = -1;
};

#endif  // RP24_YOLO_WRAPPER_H
