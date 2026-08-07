#include "utils/PerformanceMonitor.h"

#include <iomanip>
#include <iostream>

PerformanceMonitor::PerformanceMonitor(bool enabled, size_t report_interval)
    : enabled_(enabled)
    , report_interval_(report_interval == 0 ? 1 : report_interval)
{
}

bool PerformanceMonitor::enabled() const
{
    return enabled_.load();
}

void PerformanceMonitor::setEnabled(bool enabled)
{
    enabled_.store(enabled);
}

FrameProfile PerformanceMonitor::beginFrame(uint64_t frame_id,
                                            PerfTimePoint start_time) const
{
    FrameProfile profile;
    profile.frame_id = frame_id;
    profile.frame_start_time = start_time;
    return profile;
}

void PerformanceMonitor::recordStage(FrameProfile& profile,
                                     const std::string& stage,
                                     PerfTimePoint start_time,
                                     PerfTimePoint end_time) const
{
    if (!enabled()) return;
    profile.stages[stage] += durationMs(start_time, end_time);
}

void PerformanceMonitor::endFrame(FrameProfile& profile,
                                  PerfTimePoint end_time)
{
    if (!enabled()) return;

    profile.total_ms = durationMs(profile.frame_start_time, end_time);

    std::lock_guard<std::mutex> lock(history_mtx_);
    history_.push_back(profile);
    if (history_.size() >= report_interval_) {
        printStatisticsLocked();
        history_.clear();
    }
}

double PerformanceMonitor::durationMs(PerfTimePoint start_time,
                                      PerfTimePoint end_time)
{
    return std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

void PerformanceMonitor::printStatistics()
{
    std::lock_guard<std::mutex> lock(history_mtx_);
    printStatisticsLocked();
}

void PerformanceMonitor::reset()
{
    std::lock_guard<std::mutex> lock(history_mtx_);
    history_.clear();
}

void PerformanceMonitor::printStatisticsLocked() const
{
    if (history_.empty()) return;

    const std::vector<std::string> stage_order = {
        "stage1_2d_detect_classify",
        "stage2_3d_solve_transform",
        "stage3_predict_command",
        "stage4_visualize_log",
    };

    std::unordered_map<std::string, double> stage_total;
    double total_time = 0.0;

    for (const auto& frame : history_) {
        total_time += frame.total_ms;
        for (const auto& stage : frame.stages) {
            stage_total[stage.first] += stage.second;
        }
    }

    const double frame_count = static_cast<double>(history_.size());

    std::cout << "\n========== Auto Aim Performance ==========\n";
    std::cout << "Frames: " << history_.size() << "\n";
    std::cout << std::fixed << std::setprecision(3);

    for (const auto& stage : stage_order) {
        const auto it = stage_total.find(stage);
        const double avg_ms = (it == stage_total.end()) ? 0.0 : it->second / frame_count;
        std::cout << stage << ": " << avg_ms << " ms\n";
    }

    const double avg_total = total_time / frame_count;
    std::cout << "------------------------------------------\n";
    std::cout << "Total(from data init): " << avg_total << " ms\n";
    if (avg_total > 0.0) {
        std::cout << "FPS: " << 1000.0 / avg_total << "\n";
    }
    std::cout << "==========================================\n\n" << std::flush;
}
