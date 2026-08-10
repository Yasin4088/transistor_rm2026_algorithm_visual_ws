#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
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

namespace fs = std::filesystem;

namespace {

struct VisualizerConfig {
    struct DrawConfig {
        bool main_result = true;
        bool status_text = true;
        bool ground_stable_point = true;
        bool lights = true;
        bool armors = true;
        bool solved_armors = true;
        bool predictions = true;
        bool yaw_curve = true;
        bool rmm = true;
        bool common_debug_oscilloscope = true;
        bool gimbal_coordinate = true;
    };

    bool enable = true;
    bool show_windows = true;
    DrawConfig draw;

    static VisualizerConfig fromYaml(const YAML::Node& root)
    {
        VisualizerConfig config;
        const YAML::Node visualizer = root["visualizer"];
        if (!visualizer) {
            return config;
        }

        config.enable = readBool(visualizer, "enable", config.enable);
        config.show_windows = readBool(visualizer, "show_windows", config.show_windows);

        const YAML::Node draw = visualizer["draw"];
        if (draw) {
            config.draw.main_result = readBool(draw, "main_result", config.draw.main_result);
            config.draw.status_text = readBool(draw, "status_text", config.draw.status_text);
            config.draw.ground_stable_point = readBool(draw, "ground_stable_point", config.draw.ground_stable_point);
            config.draw.lights = readBool(draw, "lights", config.draw.lights);
            config.draw.armors = readBool(draw, "armors", config.draw.armors);
            config.draw.solved_armors = readBool(draw, "solved_armors", config.draw.solved_armors);
            config.draw.predictions = readBool(draw, "predictions", config.draw.predictions);
            config.draw.yaw_curve = readBool(draw, "yaw_curve", config.draw.yaw_curve);
            config.draw.rmm = readBool(draw, "rmm", config.draw.rmm);
            config.draw.common_debug_oscilloscope =
                readBool(draw, "common_debug_oscilloscope", config.draw.common_debug_oscilloscope);
            config.draw.gimbal_coordinate = readBool(draw, "gimbal_coordinate", config.draw.gimbal_coordinate);
        }

        return config;
    }

private:
    static bool readBool(const YAML::Node& node, const char* key, bool fallback)
    {
        const YAML::Node value = node[key];
        return value ? value.as<bool>() : fallback;
    }
};

const std::array<std::string, 10> kArmorTypeStrings = {
    "Hero",
    "Engineer",
    "Infantry1",
    "Infantry2",
    "Infantry3",
    "Sentry",
    "Outpost",
    "Base",
    "Middle",
    "Nearest",
};

const std::array<std::string, 3> kPredictorTypeStrings = {
    "None",
    "RMM",
    "AutoSwitch",
};

cv::Point2f toCvPoint(const auto_aim::msg::Point2f& point)
{
    return {point.x, point.y};
}

template <typename PointArray>
std::array<cv::Point2f, 4> toQuad(const PointArray& points)
{
    std::array<cv::Point2f, 4> quad{};
    for (size_t i = 0; i < quad.size() && i < points.size(); ++i) {
        quad[i] = toCvPoint(points[i]);
    }
    return quad;
}

void drawQuad(cv::Mat& image, const std::array<cv::Point2f, 4>& quad, const cv::Scalar& color, int thickness)
{
    for (size_t i = 0; i < quad.size(); ++i) {
        cv::line(image, quad[i], quad[(i + 1) % quad.size()], color, thickness);
    }
}

} // namespace

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
                if (visualizer_config_.draw.rmm) showImage(msg, "RMM visualize");
            });
        cdo_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/auto_aim/visualizer/common_debug_oscilloscope",
            qos,
            [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                if (visualizer_config_.draw.common_debug_oscilloscope) {
                    showImage(msg, "Common Debug Oscilloscope");
                }
            });
    }

    ~AutoAimVisualizerNode()
    {
        cv::destroyAllWindows();
    }

private:
    std::shared_ptr<YAML::Node> config_file_ptr_;
    VisualizerConfig visualizer_config_;
    std::chrono::steady_clock::time_point node_start_time_;
    cv::Mat latest_raw_frame_;

    float last_current_yaw_ = 0.0f;
    float last_target_yaw_ = 0.0f;
    int current_yaw_circle_ = 0;
    int target_yaw_circle_ = 0;
    std::chrono::steady_clock::time_point last_frame_time_;
    std::deque<double> frame_time_history_;
    double frame_time_sum_ = 0.0;
    std::deque<float> current_yaw_history_;
    std::deque<float> target_yaw_history_;

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
        // 从可执行文件路径向上查找工作空间根目录（含 src/shared_files/config.yaml 的目录）
        fs::path ws_dir_path = full_path.parent_path();
        while (!ws_dir_path.empty() &&
               !fs::exists(ws_dir_path / "src" / "shared_files" / "config.yaml")) {
            ws_dir_path = ws_dir_path.parent_path();
        }
        if (ws_dir_path.empty() ||
            !fs::exists(ws_dir_path / "src" / "shared_files" / "config.yaml")) {
            std::cerr << "Error: Workspace directory not found in path" << std::endl;
            return nullptr;
        }

        fs::path config_file_path = ws_dir_path / "src/shared_files/config.yaml";
        return std::make_shared<YAML::Node>(YAML::LoadFile(config_file_path));
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

        if (visualizer_config_.draw.main_result) {
            cv::Mat display = latest_raw_frame_.clone();
            drawMainResult(display, *msg);
            cv::imshow("Armor Detection", display);
        }

        if (visualizer_config_.draw.yaw_curve) {
            cv::imshow("Yaw Visualizer", renderYawCurve(msg->yaw, msg->mcu_command_yaw));
        }

        cv::waitKey(1);
    }

    void drawMainResult(cv::Mat& image, const auto_aim::msg::VisualizerDebugData& msg)
    {
        if (visualizer_config_.draw.status_text) {
            drawStatusText(image, msg);
        }
        if (visualizer_config_.draw.ground_stable_point) {
            cv::circle(image, toCvPoint(msg.ground_stable_point), 10, cv::Scalar(0, 255, 0), 2);
        }
        if (visualizer_config_.draw.lights) {
            for (const auto& light : msg.lights) {
                drawQuad(image, toQuad(light.vertices), cv::Scalar(0, 255, 0), 2);
            }
        }
        if (visualizer_config_.draw.armors) {
            for (const auto& armor : msg.armors) {
                const auto corners = toQuad(armor.corners);
                drawQuad(image, corners, cv::Scalar(0, 255, 255), 2);
                drawQuad(image, toQuad(armor.light_bar_corners), cv::Scalar(255, 0, 0), 2);
                cv::putText(image,
                    cv::format("conf: %.2f", armor.confidence),
                    corners[0] + cv::Point2f(0, -10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 255, 255),
                    1);
            }
        }
        if (visualizer_config_.draw.solved_armors) {
            drawSolvedArmors(image, msg);
        }
        if (visualizer_config_.draw.gimbal_coordinate) {
            drawGimbalCoordinate(image, msg);
        }
    }

    void drawStatusText(cv::Mat& image, const auto_aim::msg::VisualizerDebugData& msg)
    {
        cv::putText(image,
            cv::format("V: %.1f m/s, P: %.1f, Y: %.1f",
                msg.bullet_velocity, msg.pitch, msg.yaw),
            cv::Point(20, 50),
            cv::FONT_HERSHEY_COMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            1);
        cv::putText(image,
            "enemy_color: " + msg.enemy_color,
            cv::Point(20, 80),
            cv::FONT_HERSHEY_COMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            1);

        const std::string armor_type = msg.armor_type < kArmorTypeStrings.size()
            ? kArmorTypeStrings[msg.armor_type]
            : "Unknown";
        const std::string predictor_type = msg.predictor_type < kPredictorTypeStrings.size()
            ? kPredictorTypeStrings[msg.predictor_type]
            : "Unknown";
        cv::putText(image,
            "aiming " + armor_type + ": " + predictor_type,
            cv::Point(20, 110),
            cv::FONT_HERSHEY_COMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            1);

        const auto since_start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - node_start_time_).count();
        cv::putText(image,
            cv::format("frame rate: %.1f fps", updateFrameRate()),
            cv::Point(20, 140),
            cv::FONT_HERSHEY_COMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            1);
        cv::putText(image,
            cv::format("since visualizer start: %.4f s", static_cast<float>(since_start_ms) / 1000.0f),
            cv::Point(20, 170),
            cv::FONT_HERSHEY_COMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            1);
    }

    float updateFrameRate()
    {
        const auto now = std::chrono::steady_clock::now();
        if (last_frame_time_.time_since_epoch().count() == 0) {
            last_frame_time_ = now;
            return 0.0f;
        }

        const double frame_time = std::chrono::duration<double>(now - last_frame_time_).count();
        last_frame_time_ = now;

        frame_time_history_.push_back(frame_time);
        frame_time_sum_ += frame_time;
        while (frame_time_history_.size() > 30) {
            frame_time_sum_ -= frame_time_history_.front();
            frame_time_history_.pop_front();
        }

        if (frame_time_sum_ <= 0.0) {
            return 0.0f;
        }
        return static_cast<float>(static_cast<double>(frame_time_history_.size()) / frame_time_sum_);
    }

    void drawSolvedArmors(cv::Mat& image, const auto_aim::msg::VisualizerDebugData& msg)
    {
        for (const auto& result : msg.solved_results) {
            const auto corners = toQuad(result.corners);
            const cv::Scalar contour_color = result.is_tracked_now ? cv::Scalar(0, 0, 255)
                                                                   : cv::Scalar(255, 0, 255);
            drawQuad(image, corners, contour_color, 2);
            drawQuad(image, toQuad(result.light_bar_corners), cv::Scalar(0, 255, 255), 2);

            if (visualizer_config_.draw.predictions) {
                for (const auto& prediction : result.predictions) {
                    cv::circle(image, toCvPoint(prediction), 3, cv::Scalar(255, 0, 255), -1);
                }
                cv::circle(image, toCvPoint(result.center_predicted), 3, cv::Scalar(0, 255, 255), -1);
            }
            cv::circle(image, toCvPoint(result.center), 3, cv::Scalar(0, 0, 255), -1);

            const std::string text = cv::format("N%d (%.2f)", result.number, result.confidence);
            cv::Point2f text_pos = corners[1] + cv::Point2f(0, -10);
            cv::putText(image, text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 0), 3);
            cv::putText(image, text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 255), 1);

            cv::Point2f track_pos = toCvPoint(result.center) + cv::Point2f(-30, 30);
            cv::putText(image, "TRACKING", track_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1);
        }
    }

    void drawGimbalCoordinate(cv::Mat& image, const auto_aim::msg::VisualizerDebugData& msg)
    {
        const cv::Point origin(image.cols - 180, image.rows - 120);
        const int axis_length = 70;
        cv::arrowedLine(image, origin, origin + cv::Point(axis_length, 0),
            cv::Scalar(0, 0, 255), 2);
        cv::arrowedLine(image, origin, origin + cv::Point(0, -axis_length),
            cv::Scalar(0, 255, 0), 2);
        cv::putText(image, "yaw +", origin + cv::Point(axis_length + 5, 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        cv::putText(image, "pitch +", origin + cv::Point(-15, -axis_length - 8),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        cv::putText(image, cv::format("roll: %.3f", msg.roll), origin + cv::Point(-20, 35),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        cv::putText(image, cv::format("cmd_yaw: %.3f", msg.mcu_command_yaw), origin + cv::Point(-20, 58),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    cv::Mat renderYawCurve(float current_yaw, float target_yaw)
    {
        constexpr int width = 800;
        constexpr int height = 800;
        cv::Mat display = cv::Mat::zeros(height, width, CV_8UC3);

        current_yaw = std::atan2(std::sin(current_yaw), std::cos(current_yaw));
        target_yaw = std::atan2(std::sin(target_yaw), std::cos(target_yaw));
        if (current_yaw < -M_PI / 2 && last_current_yaw_ > M_PI / 2) current_yaw_circle_ += 1;
        if (current_yaw > M_PI / 2 && last_current_yaw_ < -M_PI / 2) current_yaw_circle_ -= 1;
        if (target_yaw < -M_PI / 2 && last_target_yaw_ > M_PI / 2) target_yaw_circle_ += 1;
        if (target_yaw > M_PI / 2 && last_target_yaw_ < -M_PI / 2) target_yaw_circle_ -= 1;

        const float total_current_yaw = current_yaw + 2 * M_PI * current_yaw_circle_;
        const float total_target_yaw = target_yaw + 2 * M_PI * target_yaw_circle_;
        current_yaw_history_.push_back(total_current_yaw);
        target_yaw_history_.push_back(total_target_yaw);
        while (current_yaw_history_.size() > width) current_yaw_history_.pop_front();
        while (target_yaw_history_.size() > width) target_yaw_history_.pop_front();

        drawYawHistory(display, target_yaw_history_, cv::Scalar(0, 255, 0));
        drawYawHistory(display, current_yaw_history_, cv::Scalar(0, 0, 255));
        cv::putText(display, cv::format("total_target_yaw: %.3f", total_target_yaw),
            cv::Point(20, 50), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 255, 0), 1);
        cv::putText(display, cv::format("total_current_yaw: %.3f", total_current_yaw),
            cv::Point(20, 100), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 0, 255), 1);

        cv::line(display, cv::Point(400, 400),
            cv::Point(400 - std::sin(total_target_yaw) * 100, 400 - std::cos(total_target_yaw) * 100),
            cv::Scalar(0, 255, 0), 2);
        cv::line(display, cv::Point(400, 400),
            cv::Point(400 - std::sin(total_current_yaw) * 100, 400 - std::cos(total_current_yaw) * 100),
            cv::Scalar(0, 0, 255), 2);

        last_current_yaw_ = current_yaw;
        last_target_yaw_ = target_yaw;
        return display;
    }

    void drawYawHistory(cv::Mat& display, const std::deque<float>& history, const cv::Scalar& color)
    {
        if (history.empty()) return;

        const auto [min_it, max_it] = std::minmax_element(history.begin(), history.end());
        const float min_value = *min_it;
        const float max_value = *max_it;
        const float range = std::max(max_value - min_value, 0.1f);

        for (size_t i = 1; i < history.size(); ++i) {
            const int x0 = static_cast<int>(i - 1);
            const int x1 = static_cast<int>(i);
            const int y0 = 760 - static_cast<int>((history[i - 1] - min_value) / range * 300.0f);
            const int y1 = 760 - static_cast<int>((history[i] - min_value) / range * 300.0f);
            cv::line(display, cv::Point(x0, y0), cv::Point(x1, y1), color, 1);
        }
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
