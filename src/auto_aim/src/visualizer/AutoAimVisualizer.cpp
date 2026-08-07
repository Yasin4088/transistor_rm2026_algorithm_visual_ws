#include "visualizer/AutoAimVisualizer.h"

#include <cmath>
#include <ctime>

#include "visualizer/RestFrameDraw.h"

AutoAimVisualizer::AutoAimVisualizer(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node)
    : armor_solver_(std::make_shared<ArmorSolver>(config_file_ptr, node))
    , rest_frame_(std::make_shared<RestFrame>())
    , fps_counter_(std::make_shared<FrameRateCounter>(30))
    , yaw_visualizer_(std::make_shared<YawVisualizer>())
    , config_(VisualizerConfig::fromYaml(*config_file_ptr))
{
    rest_frame_->updateCamOrientation(0, 0, 0);
    rest_frame_->updateCamPosition(0, 0, 0);
}

AutoAimVisualizerOutput AutoAimVisualizer::render(const AutoAimVisualizerInput& input)
{
    AutoAimVisualizerOutput output;
    if (input.solved_results) {
        output.armor_count = input.solved_results->size();
    }

    if (!config_.enable) {
        return output;
    }

    rest_frame_->updateCamOrientation(input.yaw, input.pitch, input.roll);
    rest_frame_->updateCamPosition(0, 0, 0);

    if (shouldDrawMainDisplay() && input.frame && !input.frame->empty()) {
        output.display = input.frame->clone();

        if (config_.draw.status_text) {
            drawStatusText(output.display, input);
        }
        if (config_.draw.rest_frame) {
            drawRestFrame(output.display, rest_frame_, armor_solver_);
        }
        drawResults(output.display, input);
        if (config_.draw.gimbal_coordinate) {
            drawGimbalCoordinate(output.display, input);
        }
    }

    if (config_.draw.yaw_curve) {
        yaw_visualizer_->update(
            input.yaw + (input.use_head_imu ? input.to_mcu_delta_yaw : 0.0f),
            input.mcu_command_yaw);
        output.yaw_visualizer_frame = yaw_visualizer_->getDisplay();
    }

    if (input.predictor_result) {
        if (config_.draw.rmm) {
            output.rmm_visualize_frame = input.predictor_result->info_images.RMM_visualize_frame;
        }
        if (config_.draw.common_debug_oscilloscope) {
            output.common_debug_oscilloscope_frame =
                input.predictor_result->info_images.common_debug_oscilloscope_frame;
        }
    }

    return output;
}

const VisualizerConfig& AutoAimVisualizer::config() const
{
    return config_;
}

bool AutoAimVisualizer::shouldDrawMainDisplay() const
{
    return config_.draw.main_result;
}

void AutoAimVisualizer::drawStatusText(cv::Mat& image, const AutoAimVisualizerInput& input)
{
    cv::putText(image,
        cv::format("V: %.1f m/s, P: %.1f, Y: %.1f",
            input.bullet_velocity, input.pitch, input.yaw),
        cv::Point(20, 50),
        cv::FONT_HERSHEY_COMPLEX,
        0.7,
        cv::Scalar(0, 255, 0),
        1,
        8,
        false);
    cv::putText(image,
        "enemy_color: " + input.enemy_color,
        cv::Point2f(20, 80),
        cv::FONT_HERSHEY_COMPLEX,
        0.7,
        cv::Scalar(0, 255, 0),
        1,
        8,
        false);

    if (input.predictor_result) {
        cv::putText(image,
            "aiming " +
                ArmorType::ArmorTypeStrings[input.predictor_result->armor_type] +
                ": " +
                PredictorType::PredictorTypeStrings[input.predictor_result->predictor_type],
            cv::Point2f(20, 110),
            cv::FONT_HERSHEY_COMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            1,
            8,
            false);
    }

    fps_counter_->tick();
    cv::putText(image,
        cv::format("frame rate: %.1f fps", fps_counter_->fps()),
        cv::Point(20, 140),
        cv::FONT_HERSHEY_COMPLEX,
        0.7,
        cv::Scalar(0, 255, 0),
        1,
        8,
        false);
    cv::putText(image,
        cv::format("since start: %.4f s",
            static_cast<float>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - input.node_start_time).count()) / 1000.0f),
        cv::Point(20, 170),
        cv::FONT_HERSHEY_COMPLEX,
        0.7,
        cv::Scalar(0, 255, 0),
        1,
        8,
        false);

    auto system_clock_now = std::chrono::system_clock::now();
    std::time_t system_clock_now_t = std::chrono::system_clock::to_time_t(system_clock_now);
    std::tm* system_clock_now_tm = std::localtime(&system_clock_now_t);
    char system_clock_now_str_buffer[80];
    std::strftime(system_clock_now_str_buffer, sizeof(system_clock_now_str_buffer),
        "%Y-%m-%d %H:%M:%S", system_clock_now_tm);
    cv::putText(image,
        cv::format("system_clock: %s", system_clock_now_str_buffer),
        cv::Point(20, 200),
        cv::FONT_HERSHEY_COMPLEX,
        0.7,
        cv::Scalar(0, 255, 0),
        1,
        8,
        false);
}

void AutoAimVisualizer::drawResults(cv::Mat& image, const AutoAimVisualizerInput& input)
{
    if (config_.draw.ground_stable_point) {
        cv::circle(image, input.ground_stable_point, 10, cv::Scalar(0, 255, 0), 2);
    }

    if (config_.draw.aim_reference_point) {
        cv::Point3f test_point_pos = rest_frame_->worldToPnpP3f({0, 1000, 0});
        cv::Point2f test_point_pos_pixel = armor_solver_->project3DToPixel(test_point_pos);
        cv::circle(image, test_point_pos_pixel, 8, cv::Scalar(255, 0, 255), 2);
    }

    if (config_.draw.lights && input.lights) {
        for (const auto& light : *input.lights) {
            cv::Point2f vertices[4];
            light.el.points(vertices);
            for (int i = 0; i < 4; i++) {
                cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    if (config_.draw.armors && input.armors) {
        for (const auto& armor : *input.armors) {
            for (size_t i = 0; i < armor.corners.size() && i < 4; i++) {
                cv::line(image, armor.corners[i], armor.corners[(i + 1) % 4],
                    cv::Scalar(0, 255, 255), 2);
            }
            if (!armor.corners.empty()) {
                std::string conf_str = cv::format("conf: %.2f", armor.confidence);
                cv::Point text_pos(armor.corners[0].x, armor.corners[0].y - 10);
                cv::putText(image, conf_str, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 255), 1);
            }
            for (size_t i = 0; i < armor.light_bar_corners.size() && i < 4; i++) {
                cv::line(image, armor.light_bar_corners[i], armor.light_bar_corners[(i + 1) % 4],
                    cv::Scalar(255, 0, 0), 2);
            }
        }
    }

    if (!config_.draw.solved_armors || !input.solved_results) {
        return;
    }

    for (const auto& res : *input.solved_results) {
        cv::Scalar contour_color = res.is_tracked_now ? cv::Scalar(0, 0, 255)
                                                      : cv::Scalar(255, 0, 255);
        for (size_t i = 0; i < res.corners.size() && i < 4; i++) {
            cv::line(image, res.corners[i], res.corners[(i + 1) % 4], contour_color, 2);
        }
        for (size_t i = 0; i < res.armor.light_bar_corners.size() && i < 4; i++) {
            cv::line(image, res.armor.light_bar_corners[i],
                res.armor.light_bar_corners[(i + 1) % 4],
                cv::Scalar(0, 255, 255),
                2);
        }
        if (config_.draw.predictions) {
            for (auto& prediction : res.predictions) {
                cv::circle(image, prediction, 3, cv::Scalar(255, 0, 255), -1);
            }
            cv::circle(image, res.center_predicted, 3, cv::Scalar(0, 255, 255), -1);
        }
        cv::circle(image, res.center, 3, cv::Scalar(0, 0, 255), -1);

        if (res.corners.size() > 1) {
            std::string text = cv::format("N%d (%.2f)", res.number, res.confidence);
            cv::Point text_pos(res.corners[1].x, res.corners[1].y - 10);
            cv::putText(image, text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 0), 3);
            cv::putText(image, text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 255), 1);
        }

        cv::Point track_pos(res.center.x - 30, res.center.y + 30);
        cv::putText(image, "TRACKING", track_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(0, 255, 0), 1);
    }
}

void AutoAimVisualizer::drawGimbalCoordinate(cv::Mat& image, const AutoAimVisualizerInput& input)
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

    cv::putText(image,
        cv::format("roll: %.3f", input.roll),
        origin + cv::Point(-20, 35),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(255, 255, 255),
        1);
    cv::putText(image,
        cv::format("cmd_yaw: %.3f", input.mcu_command_yaw),
        origin + cv::Point(-20, 58),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(255, 255, 255),
        1);
}
