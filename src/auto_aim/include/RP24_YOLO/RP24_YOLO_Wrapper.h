#include "RP24_YOLO/OpenvinoInfer.h"
#include "memory"
#include "2d_armor_detector/Armor.h"
#include "2d_armor_detector/ArmorTracker.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <utility>
#include "macro/AutoAimMacro.h"
#include "utils/PerformanceMonitor.h"

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

    void setProfiler(std::shared_ptr<PerformanceMonitor> profiler);
    vector<Armor> detectArmors(cv::Mat& frame, string detect_color, vector<int>* rp24_classes = nullptr);
    vector<ArmorResult> detectArmorsWithClassifyAndTrack(cv::Mat& frame, string detect_color, 
        const cv::Point2f& ground_stable_point, vector<Armor>* armors_out = nullptr);

    // 异步接口：submit 立即返回，结果按提交顺序用 takeResult 取回
    uint64_t submitFrame(cv::Mat frame, int detect_color, void* user_data = nullptr);
    YoloResult takeResult(uint64_t frame_id);
    // 停止流水线（幂等）：关闭队列/寄存器并 join 三个线程
    void stop();
    bool tryTakeResult(YoloResult* out);   // 非阻塞取结果：有则 true，无则 false
    vector<ArmorResult> classifyAndTrack(vector<Armor> armors, const vector<int>& rp24_classes,
                                         const cv::Point2f& ground_stable_point);
    void reportFrameLatency(double ms);    // 上报整帧延迟（提交→取回）

private:
    // ---------- 三阶段流水线数据结构 ----------
    // 主队列中的任务
    struct YoloTask {
        uint64_t frame_id = 0;
        void* user_data = nullptr;
        cv::Mat frame;
        int detect_color = -1;
    };

    // 寄存器中流动的中间/最终数据
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

    // 单槽寄存器（最新数据优先）：一个生产者与一个消费者之间的握手槽。
    // 槽满时 put 不阻塞，直接覆盖旧数据——流水线永远处理最新帧；
    // take 在槽空时阻塞等待。
    template <typename T>
    class Register {
    public:
        bool put(T item) {
            std::lock_guard<std::mutex> lock(mtx_);
            if (closed_) return false;
            item_ = std::move(item);
            has_ = true;
            cv_.notify_one();
            return true;
        }
        bool tryTake(T* out) {
            std::lock_guard<std::mutex> lock(mtx_);
            if (!has_) return false;
            *out = std::move(item_);
            has_ = false;
            cv_.notify_one();
            return true;
        }
        bool take(T* out) {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this]() { return closed_ || has_; });
            if (!has_) return false;
            *out = std::move(item_);
            has_ = false;
            lock.unlock();
            cv_.notify_one();
            return true;
        }
        void close() {
            {
                std::lock_guard<std::mutex> lock(mtx_);
                closed_ = true;
            }
            cv_.notify_all();
        }
    private:
        std::mutex mtx_;
        std::condition_variable cv_;
        bool closed_ = false;
        bool has_ = false;
        T item_;
    };

    // 主队列（有界多槽，最新数据优先）：帧从这里进入流水线。
    // 满时 push 不阻塞，丢弃最旧帧，保持最新的 max_size_ 帧。
    class InputQueue {
    public:
        explicit InputQueue(size_t max_size) : max_size_(max_size) {}
        bool push(YoloTask task) {
            std::lock_guard<std::mutex> lock(mtx_);
            if (closed_) return false;
            if (queue_.size() >= max_size_) {
                queue_.pop_front();  // 丢弃最旧的一帧
            }
            queue_.push_back(std::move(task));
            cv_.notify_one();
            return true;
        }
        bool pop(YoloTask* out) {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this]() { return closed_ || !queue_.empty(); });
            if (queue_.empty()) return false;
            *out = std::move(queue_.front());
            queue_.pop_front();
            lock.unlock();
            cv_.notify_one();
            return true;
        }
        void close() {
            {
                std::lock_guard<std::mutex> lock(mtx_);
                closed_ = true;
            }
            cv_.notify_all();
        }
    private:
        std::mutex mtx_;
        std::condition_variable cv_;
        std::deque<YoloTask> queue_;
        size_t max_size_ = 4;
        bool closed_ = false;
    };

    // ---------- 三阶段线程 ----------
    void preprocessLoop();
    void inferLoop();
    void postprocessLoop();

    // ---------- 阶段函数（原有逻辑，保持不变） ----------
    cv::Mat preprocess(const cv::Mat& frame);
    vector<Object> infer(const cv::Mat& input, int detect_color);
    vector<Armor> postprocess(const cv::Mat& frame, const vector<Object>& objects, vector<int>* rp24_classes);

    // ---------- 流水线与线程成员 ----------
    InputQueue input_queue_{4};             // 主队列：帧入口
    Register<YoloWork> pre_register_;       // preprocess -> infer 的寄存器
    Register<YoloWork> infer_register_;     // infer -> postprocess 的寄存器
    Register<YoloWork> result_register_;    // postprocess -> 调用方 的结果寄存器
    std::thread preprocess_thread_;
    std::thread infer_thread_;
    std::thread postprocess_thread_;
    std::atomic<uint64_t> next_frame_id_{0};

    std::shared_ptr<PerformanceMonitor> profiler_;
    std::shared_ptr<OpenvinoInfer> openvino_infer;
    std::shared_ptr<YAML::Node> config_file_ptr;
    rclcpp::Node* node;
    float lightBarLengthScale = 0.82;

    int class_map[9] = {5, 0, 1, 2, 3, 4, 6, 7, 7};
    bool big_map[9] = {false, true, false, false, false, false, false, false, true};
    std::shared_ptr<ArmorTracker> armor_tracker;
};
