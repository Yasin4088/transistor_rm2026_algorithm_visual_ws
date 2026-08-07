// AutoAimVisualizer_Node.cpp
#include <filesystem>
#include <iostream>
#include <limits.h>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <yaml-cpp/yaml.h>

#include "auto_aim/msg/debug_armor.hpp"
#include "auto_aim/msg/debug_armor_result.hpp"
#include "auto_aim/msg/debug_light.hpp"
#include "auto_aim/msg/point2f.hpp"
#include "auto_aim/msg/visualizer_debug_data.hpp"
#include "visualizer/AutoAimVisualizer.h"
#include "visualizer/VisualizerConfig.h"

namespace fs = std::filesystem;

class AutoAimVisualizerNode : public rclcpp::Node {
public:
    AutoAimVisualizerNode() : Node("auto_aim_visualizer_node")
    {
        node_start_time_ = std::chrono::steady_clock::now();
        config_file_ptr_ = loadConfig();
        if (!config_file_ptr_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load visualizer config");
            return;
        }

        visualizer_config_ = VisualizerConfig::fromYaml(*config_file_ptr_);
        visualizer_ = std::make_shared<AutoAimVisualizer>(config_file_ptr_, this);
        RCLCPP_INFO(this->get_logger(),
            "AutoAimVisualizerNode: visualizer=%s, show_windows=%s",
            visualizer_config_.enable ? "enabled" : "disabled",
            visualizer_config_.show_windows ? "enabled" : "disabled");

        if (!visualizer_config_.enable || !visualizer_config_.show_windows) {
            return;
        }

        const auto qos = rclcpp::SensorDataQoS();
        raw_frame_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/auto_aim/visualizer/raw_frame",
            qos,
            [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                handleRawFrame(msg);
            });
        debug_data_sub_ = create_subscription<auto_aim::msg::VisualizerDebugData>(
            "/auto_aim/visualizer/debug_data",
            qos,
            [this](auto_aim::msg::VisualizerDebugData::ConstSharedPtr msg) {
                handleDebugData(msg);
            });
        rmm_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/auto_aim/visualizer/rmm",
            qos,
            [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                showImage(msg, "RMM visualize");
            });
        cdo_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/auto_aim/visualizer/common_debug_oscilloscope",
            qos,
            [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                showImage(msg, "Common Debug Oscilloscope");
            });
    }

    ~AutoAimVisualizerNode()
    {
        cv::destroyAllWindows();
    }

private:
    std::shared_ptr<YAML::Node> config_file_ptr_;
    VisualizerConfig visualizer_config_;
    std::shared_ptr<AutoAimVisualizer> visualizer_;
    std::chrono::steady_clock::time_point node_start_time_;
    cv::Mat latest_raw_frame_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_frame_sub_;
    rclcpp::Subscription<auto_aim::msg::VisualizerDebugData>::SharedPtr debug_data_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rmm_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cdo_sub_;

    std::shared_ptr<YAML::Node> loadConfig()
    {
        char exec_path[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", exec_path, sizeof(exec_path) - 1);
        if (len == -1) {
            perror("readlink");
            return nullptr;
        }
        exec_path[len] = '\0';

        fs::path full_path(exec_path);
        std::string full_path_str = full_path.string();
        const std::string ws_dir_name = "transistor_rm2026_algorithm_visual_ws";
        size_t pos = full_path_str.find(ws_dir_name);
        if (pos == std::string::npos) {
            std::cerr << "Error: Workspace directory not found in path" << std::endl;
            return nullptr;
        }

        fs::path ws_dir_path = full_path_str.substr(0, pos + ws_dir_name.length());
        fs::path config_file_path = ws_dir_path / "src/shared_files/config.yaml";
        return std::make_shared<YAML::Node>(YAML::LoadFile(config_file_path));
    }

    cv::Point2f toCvPoint(const auto_aim::msg::Point2f& point) const
    {
        return {point.x, point.y};
    }

    template <typename PointArray>
    std::vector<cv::Point2f> toPointVector(const PointArray& points) const
    {
        std::vector<cv::Point2f> result;
        result.reserve(points.size());
        for (const auto& point : points) {
            result.emplace_back(toCvPoint(point));
        }
        return result;
    }

    Light toLight(const auto_aim::msg::DebugLight& msg) const
    {
        std::vector<cv::Point2f> vertices = toPointVector(msg.vertices);
        return Light(cv::minAreaRect(vertices));
    }

    Armor toArmor(const auto_aim::msg::DebugArmor& msg) const
    {
        Armor armor;
        armor.corners = toPointVector(msg.corners);
        armor.light_bar_corners = toPointVector(msg.light_bar_corners);
        armor.confidence = msg.confidence;
        return armor;
    }

    ArmorResult toArmorResult(const auto_aim::msg::DebugArmorResult& msg) const
    {
        Armor armor;
        armor.corners = toPointVector(msg.corners);
        armor.light_bar_corners = toPointVector(msg.light_bar_corners);
        armor.center = toCvPoint(msg.center);
        armor.confidence = msg.confidence;

        std::vector<cv::Point2f> predictions;
        predictions.reserve(msg.predictions.size());
        for (const auto& prediction : msg.predictions) {
            predictions.emplace_back(toCvPoint(prediction));
        }

        ArmorResult result(
            armor,
            msg.number,
            msg.confidence,
            msg.is_tracked_now,
            false,
            true,
            std::move(predictions),
            toCvPoint(msg.center_predicted),
            msg.is_tracked_now);
        return result;
    }

    void handleRawFrame(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
    {
        try {
            latest_raw_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_WARN(this->get_logger(), "Failed to convert raw frame: %s", e.what());
        }
    }

    void handleDebugData(const auto_aim::msg::VisualizerDebugData::ConstSharedPtr& msg)
    {
        if (latest_raw_frame_.empty()) {
            return;
        }

        std::vector<Light> lights;
        lights.reserve(msg->lights.size());
        for (const auto& light_msg : msg->lights) {
            lights.emplace_back(toLight(light_msg));
        }

        std::vector<Armor> armors;
        armors.reserve(msg->armors.size());
        for (const auto& armor_msg : msg->armors) {
            armors.emplace_back(toArmor(armor_msg));
        }

        std::vector<ArmorResult> solved_results;
        solved_results.reserve(msg->solved_results.size());
        for (const auto& result_msg : msg->solved_results) {
            solved_results.emplace_back(toArmorResult(result_msg));
        }

        AutoAimVisualizerInput input;
        input.frame = &latest_raw_frame_;
        input.node_start_time = node_start_time_;
        input.bullet_velocity = msg->bullet_velocity;
        input.enemy_color = msg->enemy_color;
        input.pitch = msg->pitch;
        input.yaw = msg->yaw;
        input.roll = msg->roll;
        input.ground_stable_point = toCvPoint(msg->ground_stable_point);
        input.lights = &lights;
        input.armors = &armors;
        input.solved_results = &solved_results;
        input.has_predictor_state = true;
        input.armor_type = static_cast<ArmorType::ArmorType>(msg->armor_type);
        input.predictor_type = static_cast<PredictorType::PredictorType>(msg->predictor_type);
        input.mcu_command_yaw = msg->mcu_command_yaw;

        AutoAimVisualizerOutput output = visualizer_->render(input);
        if (!output.display.empty()) {
            cv::imshow("Armor Detection", output.display);
        }
        if (!output.yaw_visualizer_frame.empty()) {
            cv::imshow("Yaw Visualizer", output.yaw_visualizer_frame);
        }
        cv::waitKey(1);
    }

    void showImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg, const std::string& window_name)
    {
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
            if (cv_ptr->image.empty()) {
                return;
            }
            cv::imshow(window_name, cv_ptr->image);
            cv::waitKey(1);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_WARN(this->get_logger(),
                "Failed to convert image for %s: %s",
                window_name.c_str(),
                e.what());
        }
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<AutoAimVisualizerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
