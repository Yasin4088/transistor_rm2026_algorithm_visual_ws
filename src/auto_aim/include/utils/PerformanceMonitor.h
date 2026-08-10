#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// 轻量性能统计，不依赖 ROS / OpenCV，仅使用标准库。
//
// 两种计时方式，可混用：
//
// 1) 单线程顺序计时（同步流水线）：
//    pm.beginFrame(id);
//    pm.mark("yolo_preprocess");  // 阶段开始前调用
//    ... preprocess ...
//    pm.mark("yolo_infer");       // 上一个阶段在此结束
//    ... infer ...
//    pm.mark("yolo_postprocess");
//    ... postprocess ...
//    pm.endFrame();
//    某阶段耗时 = 下一次 mark()（或 endFrame()）与本次 mark() 的时间差。
//
// 2) 多线程阶段计时（异步流水线）：
//    各阶段线程自己测量耗时，完成后上报，与帧边界解耦：
//    pm.beginFrame(id);                       // 提交线程
//    ... 阶段线程 ...
//    pm.addStageTime("yolo_infer", ms);       // 各阶段线程
//    pm.endFrame();                           // 结果消费线程
//
// 所有方法线程安全；报告按统计窗口（N 帧）输出平均值。
class PerformanceMonitor {
public:
    explicit PerformanceMonitor(size_t report_interval = 90);

    void beginFrame(uint64_t frame_id);
    void mark(const std::string& stage);
    void addStageTime(const std::string& stage, double ms);
    void addTotalTime(double ms);   // 异步路径：提交→取回的整帧延迟（与阶段解耦）
    void endFrame();

private:
    struct FrameRecord {
        uint64_t frame_id = 0;
        double total_ms = 0.0;
        std::chrono::steady_clock::time_point completed_at;  // 帧完成时刻（算真实吞吐）
    };

    void report();

    size_t report_interval_;
    std::vector<FrameRecord> frames_;

    std::mutex mtx_;
    bool frame_active_ = false;
    bool has_marks_ = false;
    uint64_t frame_id_ = 0;
    std::string pending_stage_;
    std::chrono::steady_clock::time_point frame_start_;
    std::chrono::steady_clock::time_point last_mark_;
    std::map<std::string, double> stage_window_ms_;  // 当前统计窗口内各阶段累计耗时
    std::vector<std::string> stage_order_;           // 阶段首次出现顺序，输出更稳定
};
