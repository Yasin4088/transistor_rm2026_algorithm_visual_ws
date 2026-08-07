// AutoAimVisualizer_Node.cpp
#include <filesystem>
#include <iostream>
#include <limits.h>
#include <memory>
#include <string>
#include <unistd.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <yaml-cpp/yaml.h>

#include "visualizer/VisualizerConfig.h"

namespace fs = std::filesystem;

class AutoAimVisualizerNode : public rclcpp::Node {
public:
    AutoAimVisualizerNode() : Node("auto_aim_visualizer_node")
    {
        config_file_ptr_ = loadConfig();
        if (!config_file_ptr_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load visualizer config");
            return;
        }

        visualizer_config_ = VisualizerConfig::fromYaml(*config_file_ptr_);
        RCLCPP_INFO(this->get_logger(),
            "AutoAimVisualizerNode: visualizer=%s, show_windows=%s",
            visualizer_config_.enable ? "enabled" : "disabled",
            visualizer_config_.show_windows ? "enabled" : "disabled");

        if (!visualizer_config_.enable || !visualizer_config_.show_windows) {
            return;
        }

        const auto qos = rclcpp::SensorDataQoS();
        result_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/auto_aim/visualizer/result",
            qos,
            [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                showImage(msg, "Armor Detection");
            });
        yaw_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/auto_aim/visualizer/yaw",
            qos,
            [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                showImage(msg, "Yaw Visualizer");
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
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr result_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr yaw_sub_;
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
