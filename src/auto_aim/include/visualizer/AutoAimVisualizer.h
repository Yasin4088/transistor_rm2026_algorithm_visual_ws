#ifndef AUTO_AIM_VISUALIZER_H
#define AUTO_AIM_VISUALIZER_H

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include "2d_armor_detector/Armor.h"
#include "2d_armor_detector/LightBar.h"
#include "3d_processing/ArmorSolver.h"
#include "3d_processing/RestFrame.h"
#include "predictor/AllPredictor.h"
#include "utils/FrameRateCounter.h"
#include "visualizer/VisualizerConfig.h"
#include "visualizer/YawVisualizer.h"

struct AutoAimVisualizerInput {
    const cv::Mat* frame = nullptr;
    std::chrono::steady_clock::time_point node_start_time;

    float bullet_velocity = 0.0f;
    std::string enemy_color;
    float pitch = 0.0f;
    float yaw = 0.0f;
    float roll = 0.0f;
    cv::Point2f ground_stable_point;

    bool use_head_imu = false;
    float to_mcu_delta_yaw = 0.0f;
    float to_mcu_delta_pitch = 0.0f;

    const std::vector<Light>* lights = nullptr;
    const std::vector<Armor>* armors = nullptr;
    const std::vector<ArmorResult>* solved_results = nullptr;
    const PredictorResult* predictor_result = nullptr;
    float mcu_command_yaw = 0.0f;
};

struct AutoAimVisualizerOutput {
    cv::Mat display;
    cv::Mat yaw_visualizer_frame;
    cv::Mat rmm_visualize_frame;
    cv::Mat common_debug_oscilloscope_frame;
    size_t armor_count = 0;
};

class AutoAimVisualizer {
public:
    AutoAimVisualizer(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node);

    AutoAimVisualizerOutput render(const AutoAimVisualizerInput& input);
    const VisualizerConfig& config() const;

private:
    std::shared_ptr<ArmorSolver> armor_solver_;
    std::shared_ptr<RestFrame> rest_frame_;
    std::shared_ptr<FrameRateCounter> fps_counter_;
    std::shared_ptr<YawVisualizer> yaw_visualizer_;
    VisualizerConfig config_;

    bool shouldDrawMainDisplay() const;
    void drawStatusText(cv::Mat& image, const AutoAimVisualizerInput& input);
    void drawResults(cv::Mat& image, const AutoAimVisualizerInput& input);
    void drawGimbalCoordinate(cv::Mat& image, const AutoAimVisualizerInput& input);
};

#endif // AUTO_AIM_VISUALIZER_H
