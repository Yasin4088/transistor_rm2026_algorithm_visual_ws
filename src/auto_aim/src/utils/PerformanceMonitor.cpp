#include "utils/PerformanceMonitor.h"

#include <iomanip>
#include <iostream>
#include <utility>

PerformanceMonitor::PerformanceMonitor(size_t report_interval)
    : report_interval_(report_interval)
{
    // 启动标记：用于确认二进制是否为最新（stderr 无缓冲，任何环境下都能看到）
    std::cerr << "[PerformanceMonitor] initialized, report interval = "
              << report_interval_ << " frames\n";
}

void PerformanceMonitor::beginFrame(uint64_t frame_id)
{
    std::lock_guard<std::mutex> lock(mtx_);
    frame_id_ = frame_id;
    frame_start_ = std::chrono::steady_clock::now();
    frame_active_ = true;
    has_marks_ = false;
    pending_stage_.clear();
}

void PerformanceMonitor::mark(const std::string& stage)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!frame_active_) {
        return;
    }

    auto now = std::chrono::steady_clock::now();

    // 关闭上一个阶段：其耗时 = 本次 mark 与上次 mark 的时间差
    if (has_marks_) {
        double ms =
            std::chrono::duration<double, std::milli>(now - last_mark_).count();
        stage_window_ms_[pending_stage_] += ms;
    }

    pending_stage_ = stage;
    bool first_seen = true;
    for (const auto& name : stage_order_) {
        if (name == stage) {
            first_seen = false;
            break;
        }
    }
    if (first_seen) {
        stage_order_.push_back(stage);
    }
    last_mark_ = now;
    has_marks_ = true;
}

void PerformanceMonitor::addStageTime(const std::string& stage, double ms)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!frame_active_) {
        return;
    }

    stage_window_ms_[stage] += ms;
    bool first_seen = true;
    for (const auto& name : stage_order_) {
        if (name == stage) {
            first_seen = false;
            break;
        }
    }
    if (first_seen) {
        stage_order_.push_back(stage);
    }
}

void PerformanceMonitor::addTotalTime(double ms)
{
    std::lock_guard<std::mutex> lock(mtx_);
    FrameRecord record;
    record.total_ms = ms;
    record.completed_at = std::chrono::steady_clock::now();
    frames_.push_back(std::move(record));

    if (frames_.size() >= report_interval_) {
        report();
        frames_.clear();
        stage_window_ms_.clear();
    }
}

void PerformanceMonitor::endFrame()
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!frame_active_) {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(now - frame_start_).count();

    // 关闭最后一个阶段
    if (has_marks_) {
        double ms =
            std::chrono::duration<double, std::milli>(now - last_mark_).count();
        stage_window_ms_[pending_stage_] += ms;
    }

    FrameRecord record;
    record.frame_id = frame_id_;
    record.total_ms = total_ms;
    record.completed_at = std::chrono::steady_clock::now();
    frames_.push_back(std::move(record));

    frame_active_ = false;
    has_marks_ = false;
    pending_stage_.clear();

    if (frames_.size() >= report_interval_) {
        report();
        frames_.clear();
        stage_window_ms_.clear();
    }
}

void PerformanceMonitor::report()
{
    size_t count = frames_.size();
    if (count == 0) {
        return;
    }

    double total_sum = 0.0;
    for (const auto& frame : frames_) {
        total_sum += frame.total_ms;
    }

    double avg_total_ms = total_sum / static_cast<double>(count);
    double fps = 0.0;
    if (count >= 2) {
        // 真实吞吐：窗口内完成帧数 / 首尾完成时间差（异步多帧在飞时 latency 倒数不再等于吞吐）
        double span_s = std::chrono::duration<double>(
            frames_.back().completed_at - frames_.front().completed_at).count();
        if (span_s > 0.0) {
            fps = static_cast<double>(count - 1) / span_s;
        }
    }
    if (fps <= 0.0 && avg_total_ms > 0.0) {
        fps = 1000.0 / avg_total_ms;  // 兜底（样本不足时）
    }

    std::cerr << "========== Performance ==========\n";
    std::cerr << "Frames: " << count << "\n\n";

    for (const auto& name : stage_order_) {
        double avg_ms = 0.0;
        auto it = stage_window_ms_.find(name);
        if (it != stage_window_ms_.end()) {
            avg_ms = it->second / static_cast<double>(count);
        }
        std::cerr << name << ":\n"
                  << std::fixed << std::setprecision(2) << avg_ms << " ms\n\n";
    }

    std::cerr << "Total:\n"
              << std::fixed << std::setprecision(2) << avg_total_ms << " ms\n\n";
    std::cerr << "FPS:\n"
              << std::fixed << std::setprecision(2) << fps << "\n\n";
    std::cerr << "================================\n";
    // ros2 launch / 重定向等管道环境下 stdout 是全缓冲，
    // 必须显式 flush，否则报告会一直憋在缓冲区里不显示
    std::cerr.flush();
}
