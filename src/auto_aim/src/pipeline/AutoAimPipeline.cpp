#include "pipeline/AutoAimPipeline.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <stdexcept>

#include "macro/AutoAimMacro.h"
#include "visualizer/RestFrameDraw.h"

namespace {

Params loadDetectorParams(const std::shared_ptr<YAML::Node>& config_file_ptr, std::string* enemy_color_out)
{
    Params params;

#ifdef FIX_ENEMY_COLOR
    std::string enemy_color = (FIX_ENEMY_COLOR == 0) ? "RED" : "BLUE";
#else
    std::string enemy_color = (*config_file_ptr)["init_enemy_color"].as<std::string>();
#endif

    if (enemy_color == "RED") {
        params.enemy_color = Params::RED;
    } else if (enemy_color == "BLUE") {
        params.enemy_color = Params::BLUE;
    } else if (enemy_color == "GREEN") {
        params.enemy_color = Params::GREEN;
    } else if (enemy_color == "BOTH") {
        params.enemy_color = Params::BOTH;
    } else {
        enemy_color = "GREEN";
        params.enemy_color = Params::GREEN;
    }

    params.min_light_height = (*config_file_ptr)["min_light_height"].as<int>();
    params.light_min_area = (*config_file_ptr)["light_min_area"].as<int>();
    params.light_max_area = (*config_file_ptr)["light_max_area"].as<int>();
    params.max_light_wh_ratio = (*config_file_ptr)["max_light_wh_ratio"].as<float>();
    params.min_light_wh_ratio = (*config_file_ptr)["min_light_wh_ratio"].as<float>();
    params.light_max_tilt_angle = (*config_file_ptr)["light_max_tilt_angle"].as<float>();

    if (enemy_color_out) {
        *enemy_color_out = enemy_color;
    }
    return params;
}

Params::EnemyColor toEnemyColor(const std::string& enemy_color)
{
    if (enemy_color == "RED") return Params::RED;
    if (enemy_color == "BLUE") return Params::BLUE;
    if (enemy_color == "GREEN") return Params::GREEN;
    if (enemy_color == "BOTH") return Params::BOTH;
    return Params::GREEN;
}

}  // namespace

// ==================== Stage1: 2D检测与分类 ====================

AutoAimPipeline::Stage1::Stage1(std::shared_ptr<YAML::Node> config_file_ptr,
                                rclcpp::Node* node,
                                const std::filesystem::path& workspace_path)
{
    std::string init_enemy_color;
    Params params = loadDetectorParams(config_file_ptr, &init_enemy_color);

    use_rp24_yolo = (*config_file_ptr)["use_RP24_YOLO"].as<bool>();
    light_detector = std::make_shared<LightBarDetector>(params, config_file_ptr, node);
    armor_detector = std::make_shared<ArmorDetector>(config_file_ptr, node);
    classifier = std::make_shared<ArmorClassifier>(config_file_ptr, node, workspace_path);
    rp24_yolo_wrapper = std::make_shared<RP24YOLOWrapper>(
        config_file_ptr,
        node,
        workspace_path / (*config_file_ptr)["RP24_YOLO_model_relative_path"].as<std::string>(),
        (*config_file_ptr)["RP24_YOLO_device"].as<std::string>());
}

void AutoAimPipeline::Stage1::start(AutoAimPipelineData& d)
{
    if (!idle.load()) {
        throw std::runtime_error("AutoAimPipeline::Stage1::start: stage is not idle");
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        data = &d;
        idle.store(false);
    }
    cv.notify_one();
}

bool AutoAimPipeline::Stage1::isIdle() const
{
    return idle.load();
}

void AutoAimPipeline::Stage1::run()
{
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]() { return !idle.load() || exit_flag; });
        if (exit_flag) return;

        AutoAimPipelineData* d = data;
        lock.unlock();

        d->stage1 = AutoAimPipelineData::Stage1Data{};
        d->stage1.used_yolo = use_rp24_yolo;

        if (use_rp24_yolo) {
            d->stage1.classify_results =
                rp24_yolo_wrapper->detectArmorsWithClassifyAndTrack(
                    d->initial.frame,
                    d->initial.enemy_color,
                    d->initial.ground_stable_point,
                    &d->stage1.armors);
            for (Armor& armor : d->stage1.armors) {
                d->stage1.lights.emplace_back(armor.leftLight);
                d->stage1.lights.emplace_back(armor.rightLight);
            }
        } else {
            light_detector->setEnemyColor(toEnemyColor(d->initial.enemy_color));
            light_detector->detectLights(d->initial.frame);
            light_detector->processLights();
            d->stage1.lights = light_detector->getLights();
            d->stage1.armors = armor_detector->detectArmors(d->stage1.lights);
            d->stage1.classify_results =
                classifier->classify(
                    d->initial.frame,
                    d->stage1.armors,
                    d->initial.ground_stable_point);
        }

        idle.store(true);
    }
}

// ==================== Stage2: 3D解算与坐标转换 ====================

AutoAimPipeline::Stage2::Stage2(std::shared_ptr<YAML::Node> config_file_ptr,
                                rclcpp::Node* node)
{
    armor_solver = std::make_shared<ArmorSolver>(config_file_ptr, node);
    rest_frame = std::make_shared<RestFrame>();
    rest_frame->updateCamOrientation(0, 0, 0);
    rest_frame->updateCamPosition(0, 0, 0);
    max_armor_position_height = (*config_file_ptr)["max_armor_position_height"].as<float>();
}

void AutoAimPipeline::Stage2::start(AutoAimPipelineData& d)
{
    if (!idle.load()) {
        throw std::runtime_error("AutoAimPipeline::Stage2::start: stage is not idle");
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        data = &d;
        idle.store(false);
    }
    cv.notify_one();
}

bool AutoAimPipeline::Stage2::isIdle() const
{
    return idle.load();
}

void AutoAimPipeline::Stage2::run()
{
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]() { return !idle.load() || exit_flag; });
        if (exit_flag) return;

        AutoAimPipelineData* d = data;
        lock.unlock();

        d->stage2 = AutoAimPipelineData::Stage2Data{};
        rest_frame->updateCamOrientation(
            d->initial.yaw,
            d->initial.pitch,
            d->initial.roll);
        rest_frame->updateCamPosition(0, 0, 0);

        for (ArmorResult classify_result : d->stage1.classify_results) {
            AimResult solve_armor_result =
                armor_solver->solveArmor(classify_result, d->initial.pitch, d->initial.yaw);
            cv::Point3f rest_frame_pos = rest_frame->pnpToWorldP3f(solve_armor_result.position);
            if (rest_frame_pos.z < max_armor_position_height && solve_armor_result.valid) {
                classify_result.solve_armor_result = solve_armor_result;
                d->stage2.solved_results.emplace_back(std::move(classify_result));
                d->stage2.rest_frame_positions.emplace_back(rest_frame_pos);
            }
        }
        d->stage2.valid_count = d->stage2.solved_results.size();

        idle.store(true);
    }
}

// ==================== Stage3: 预测与命令 ====================

AutoAimPipeline::Stage3::Stage3(std::shared_ptr<YAML::Node> config_file_ptr,
                                rclcpp::Node* node,
                                std::chrono::steady_clock::time_point node_start_time)
{
    armor_solver = std::make_shared<ArmorSolver>(config_file_ptr, node);
    ballistic_solver = std::make_shared<BallisticSolver>(config_file_ptr, node);
    rest_frame = std::make_shared<RestFrame>();
    rest_frame->updateCamOrientation(0, 0, 0);
    rest_frame->updateCamPosition(0, 0, 0);
    fps_counter = std::make_shared<FrameRateCounter>(30);
    predictor_main = std::make_shared<PredictorMain>(
        config_file_ptr,
        node,
        node_start_time,
        armor_solver,
        ballistic_solver,
        rest_frame,
        fps_counter);
}

void AutoAimPipeline::Stage3::start(AutoAimPipelineData& d)
{
    if (!idle.load()) {
        throw std::runtime_error("AutoAimPipeline::Stage3::start: stage is not idle");
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        data = &d;
        idle.store(false);
    }
    cv.notify_one();
}

bool AutoAimPipeline::Stage3::isIdle() const
{
    return idle.load();
}

void AutoAimPipeline::Stage3::run()
{
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]() { return !idle.load() || exit_flag; });
        if (exit_flag) return;

        AutoAimPipelineData* d = data;
        lock.unlock();

        d->stage3 = AutoAimPipelineData::Stage3Data{};
        rest_frame->updateCamOrientation(
            d->initial.yaw,
            d->initial.pitch,
            d->initial.roll);
        rest_frame->updateCamPosition(0, 0, 0);
        predictor_main->update_serial_info(
            d->initial.bullet_velocity,
            d->initial.pitch,
            d->initial.yaw,
            d->initial.total_yaw);

        d->stage3.predictor_result =
            predictor_main->step(
                d->stage2.solved_results,
                d->initial.frame,
                PredictorType::AutoSwitch,
                ArmorType::Nearest,
                d->initial.auto_aim_switch,
                d->initial.mcu_yaw_online);

        d->stage3.mcu_command_pitch = d->stage3.predictor_result.command_pitch;
        d->stage3.mcu_command_yaw = d->stage3.predictor_result.command_yaw;
        if (d->initial.use_head_imu) {
            d->stage3.mcu_command_pitch = d->stage3.predictor_result.command_pitch;
            d->stage3.mcu_command_yaw =
                d->stage3.predictor_result.command_yaw + d->initial.to_mcu_delta_yaw;
        }
        d->stage3.should_send_reset = d->stage3.predictor_result.reset;

        idle.store(true);
    }
}

void AutoAimPipeline::Stage3::resetYawIntegration()
{
    predictor_main->reset_yaw_integration();
}

// ==================== Stage4: 可视化输出与日志记录 ====================

AutoAimPipeline::Stage4::Stage4(std::shared_ptr<YAML::Node> config_file_ptr,
                                rclcpp::Node* node,
                                const std::filesystem::path& workspace_path)
{
    armor_solver = std::make_shared<ArmorSolver>(config_file_ptr, node);
    rest_frame = std::make_shared<RestFrame>();
    rest_frame->updateCamOrientation(0, 0, 0);
    rest_frame->updateCamPosition(0, 0, 0);
    fps_counter = std::make_shared<FrameRateCounter>(30);
    yaw_visualizer = std::make_shared<YawVisualizer>();
#if (defined LOG_RESULT_VIDEO) || (defined LOG_ORIGIN_VIDEO)
    two_video_logger = std::make_shared<TwoVideoLogger>(workspace_path / "VideoLog");
#endif
}

void AutoAimPipeline::Stage4::start(AutoAimPipelineData& d)
{
    if (!idle.load()) {
        throw std::runtime_error("AutoAimPipeline::Stage4::start: stage is not idle");
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        data = &d;
        idle.store(false);
    }
    cv.notify_one();
}

bool AutoAimPipeline::Stage4::isIdle() const
{
    return idle.load();
}

void AutoAimPipeline::Stage4::run()
{
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]() { return !idle.load() || exit_flag; });
        if (exit_flag) return;

        AutoAimPipelineData* d = data;
        lock.unlock();

        d->stage4 = AutoAimPipelineData::Stage4Data{};
        rest_frame->updateCamOrientation(
            d->initial.yaw,
            d->initial.pitch,
            d->initial.roll);
        rest_frame->updateCamPosition(0, 0, 0);

        cv::Mat display = d->initial.frame.clone();

        cv::putText(display,
            cv::format("V: %.1f m/s, P: %.1f, Y: %.1f",
                d->initial.bullet_velocity, d->initial.pitch, d->initial.yaw),
            cv::Point(20, 50),
            cv::FONT_HERSHEY_COMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            1,
            8,
            false);
        cv::putText(display,
            "enemy_color: " + d->initial.enemy_color,
            cv::Point2f(20, 80),
            cv::FONT_HERSHEY_COMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            1,
            8,
            false);
        cv::putText(display,
            "aiming " +
                ArmorType::ArmorTypeStrings[d->stage3.predictor_result.armor_type] +
                ": " +
                PredictorType::PredictorTypeStrings[d->stage3.predictor_result.predictor_type],
            cv::Point2f(20, 110),
            cv::FONT_HERSHEY_COMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            1,
            8,
            false);

        drawRestFrame(display, rest_frame, armor_solver);
        drawResults(display, *d);

        yaw_visualizer->update(
            d->initial.yaw + (d->initial.use_head_imu ? d->initial.to_mcu_delta_yaw : 0.0f),
            d->stage3.mcu_command_yaw);
        d->stage4.yaw_visualizer_frame = yaw_visualizer->getDisplay();

        fps_counter->tick();
        cv::putText(display,
            cv::format("frame rate: %.1f fps", fps_counter->fps()),
            cv::Point(20, 140),
            cv::FONT_HERSHEY_COMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            1,
            8,
            false);
        cv::putText(display,
            cv::format("since start: %.4f s",
                static_cast<float>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - d->initial.node_start_time).count()) / 1000.0f),
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
        cv::putText(display,
            cv::format("system_clock: %s", system_clock_now_str_buffer),
            cv::Point(20, 200),
            cv::FONT_HERSHEY_COMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            1,
            8,
            false);

        d->stage4.rmm_visualize_frame =
            d->stage3.predictor_result.info_images.RMM_visualize_frame;
        d->stage4.common_debug_oscilloscope_frame =
            d->stage3.predictor_result.info_images.common_debug_oscilloscope_frame;
        d->stage4.armor_count = d->stage2.solved_results.size();

#if (defined LOG_RESULT_VIDEO) || (defined LOG_ORIGIN_VIDEO)
        if (two_video_logger) {
            two_video_logger->updateOriginFrame(d->initial.frame);
            two_video_logger->updateDrewFrame(display);
            two_video_logger->updateRMMFrame(d->stage4.rmm_visualize_frame);
            two_video_logger->updateCDOFrame(d->stage4.common_debug_oscilloscope_frame);
            two_video_logger->updateYawFrame(d->stage4.yaw_visualizer_frame);
            two_video_logger->updateComFrame(d->initial.com_data_visualize_frame);
            two_video_logger->writeTwoFrame();
            d->stage4.request_com_frame_refresh = true;
        }
#endif

        d->stage4.display = std::move(display);
        idle.store(true);
    }
}

void AutoAimPipeline::Stage4::drawResults(cv::Mat& image, const AutoAimPipelineData& d)
{
    cv::circle(image, d.initial.ground_stable_point, 10, cv::Scalar(0, 255, 0), 2);

    cv::Point3f test_point_pos = rest_frame->worldToPnpP3f({0, 1000, 0});
    cv::Point2f test_point_pos_pixel = armor_solver->project3DToPixel(test_point_pos);
    cv::circle(image, test_point_pos_pixel, 8, cv::Scalar(255, 0, 255), 2);

    for (const auto& light : d.stage1.lights) {
        cv::Point2f vertices[4];
        light.el.points(vertices);
        for (int i = 0; i < 4; i++) {
            cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }

    for (const auto& armor : d.stage1.armors) {
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

    for (const auto& res : d.stage2.solved_results) {
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
        for (auto& prediction : res.predictions) {
            cv::circle(image, prediction, 3, cv::Scalar(255, 0, 255), -1);
        }
        cv::circle(image, res.center_predicted, 3, cv::Scalar(0, 255, 255), -1);
        cv::circle(image, res.center, 3, cv::Scalar(0, 0, 255), -1);

        std::string text = cv::format("N%d (%.2f)", res.number, res.confidence);
        cv::Point text_pos(res.corners[1].x, res.corners[1].y - 10);
        cv::putText(image, text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(0, 0, 0), 3);
        cv::putText(image, text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(0, 0, 255), 1);

        cv::Point track_pos(res.center.x - 30, res.center.y + 30);
        cv::putText(image, "TRACKING", track_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(0, 255, 0), 1);
    }
}

// ==================== Pipeline scheduler and public API ====================

AutoAimPipeline::AutoAimPipeline(std::shared_ptr<YAML::Node> config_file_ptr,
                                 rclcpp::Node* node,
                                 const std::filesystem::path& workspace_path,
                                 std::chrono::steady_clock::time_point node_start_time,
                                 int max_queue_size,
                                 float max_delay_seconds)
    : max_queue_size_(max_queue_size)
    , max_delay_seconds_(max_delay_seconds)
    , stage1_(config_file_ptr, node, workspace_path)
    , stage2_(config_file_ptr, node)
    , stage3_(config_file_ptr, node, node_start_time)
    , stage4_(config_file_ptr, node, workspace_path)
{
    for (auto& queue_size : queue_sizes_) {
        queue_size.store(0);
    }

    stage1_.worker = std::thread(&Stage1::run, &stage1_);
    stage2_.worker = std::thread(&Stage2::run, &stage2_);
    stage3_.worker = std::thread(&Stage3::run, &stage3_);
    stage4_.worker = std::thread(&Stage4::run, &stage4_);
    scheduler_thread_ = std::thread(&AutoAimPipeline::schedulerLoop, this);
}

AutoAimPipeline::~AutoAimPipeline()
{
    scheduler_exit_.store(true);
    if (scheduler_thread_.joinable()) scheduler_thread_.join();

    {
        std::lock_guard<std::mutex> lock(stage1_.mtx);
        stage1_.exit_flag = true;
    }
    {
        std::lock_guard<std::mutex> lock(stage2_.mtx);
        stage2_.exit_flag = true;
    }
    {
        std::lock_guard<std::mutex> lock(stage3_.mtx);
        stage3_.exit_flag = true;
    }
    {
        std::lock_guard<std::mutex> lock(stage4_.mtx);
        stage4_.exit_flag = true;
    }
    stage1_.cv.notify_one();
    stage2_.cv.notify_one();
    stage3_.cv.notify_one();
    stage4_.cv.notify_one();

    if (stage1_.worker.joinable()) stage1_.worker.join();
    if (stage2_.worker.joinable()) stage2_.worker.join();
    if (stage3_.worker.joinable()) stage3_.worker.join();
    if (stage4_.worker.joinable()) stage4_.worker.join();
}

void AutoAimPipeline::addFrame(AutoAimPipelineData::InitialData initial)
{
    auto data = std::make_unique<AutoAimPipelineData>();
    data->initial = std::move(initial);

    std::lock_guard<std::mutex> lock(input_mtx_);
    if (scheduler_exit_.load()) return;
    if (input_queue_.size() >= static_cast<size_t>(max_queue_size_)) {
        input_queue_.pop_front();
    }
    input_queue_.push_back(std::move(data));
}

AutoAimPipeline::ProcessResult
AutoAimPipeline::tryPopResult(const std::chrono::steady_clock::time_point& timestamp)
{
    ProcessResult result;
    result.always_valid_data.queue_input = queue_sizes_[0].load();
    result.always_valid_data.queue_inter0 = queue_sizes_[1].load();
    result.always_valid_data.queue_inter1 = queue_sizes_[2].load();
    result.always_valid_data.queue_inter2 = queue_sizes_[3].load();
    result.always_valid_data.queue_output = queue_sizes_[4].load();

    std::lock_guard<std::mutex> lock(output_mtx_);
    if (output_queue_.empty()) return result;

    auto& front = output_queue_.front();
    float diff = std::chrono::duration<float>(
        timestamp - front->initial.frame_timestamp).count();
    if (diff >= max_delay_seconds_) {
        result.valid_data.predictor_result = front->stage3.predictor_result;
        result.valid_data.mcu_command_pitch = front->stage3.mcu_command_pitch;
        result.valid_data.mcu_command_yaw = front->stage3.mcu_command_yaw;
        result.valid_data.should_send_reset = front->stage3.should_send_reset;
        result.valid_data.display = std::move(front->stage4.display);
        result.valid_data.yaw_visualizer_frame = std::move(front->stage4.yaw_visualizer_frame);
        result.valid_data.rmm_visualize_frame = std::move(front->stage4.rmm_visualize_frame);
        result.valid_data.common_debug_oscilloscope_frame =
            std::move(front->stage4.common_debug_oscilloscope_frame);
        result.valid_data.armor_count = front->stage4.armor_count;
        result.valid_data.request_com_frame_refresh = front->stage4.request_com_frame_refresh;
        result.valid = true;
        output_queue_.pop_front();
    }

    return result;
}

void AutoAimPipeline::resetYawIntegration()
{
    stage3_.resetYawIntegration();
}

void AutoAimPipeline::schedulerLoop()
{
    while (!scheduler_exit_.load()) {
        bool any_work = false;

        if (stage1_.isIdle()) {
            if (in_flight_[0]) {
                inter_queues_[0].push_back(std::move(in_flight_[0]));
            }
            if (inter_queues_[0].size() < static_cast<size_t>(max_queue_size_)) {
                std::unique_lock<std::mutex> lk(input_mtx_);
                if (!input_queue_.empty()) {
                    in_flight_[0] = std::move(input_queue_.front());
                    input_queue_.pop_front();
                    lk.unlock();
                    input_cv_.notify_one();
                    stage1_.start(*in_flight_[0]);
                    any_work = true;
                }
            }
        }

        if (stage2_.isIdle()) {
            if (in_flight_[1]) {
                inter_queues_[1].push_back(std::move(in_flight_[1]));
            }
            if (inter_queues_[1].size() < static_cast<size_t>(max_queue_size_)
                && !inter_queues_[0].empty()) {
                in_flight_[1] = std::move(inter_queues_[0].front());
                inter_queues_[0].pop_front();
                stage2_.start(*in_flight_[1]);
                any_work = true;
            }
        }

        if (stage3_.isIdle()) {
            if (in_flight_[2]) {
                inter_queues_[2].push_back(std::move(in_flight_[2]));
            }
            if (inter_queues_[2].size() < static_cast<size_t>(max_queue_size_)
                && !inter_queues_[1].empty()) {
                in_flight_[2] = std::move(inter_queues_[1].front());
                inter_queues_[1].pop_front();
                stage3_.start(*in_flight_[2]);
                any_work = true;
            }
        }

        if (stage4_.isIdle()) {
            if (in_flight_[3]) {
                std::lock_guard<std::mutex> lk(output_mtx_);
                output_queue_.push_back(std::move(in_flight_[3]));
            }
            {
                std::lock_guard<std::mutex> lk(output_mtx_);
                if (output_queue_.size() < static_cast<size_t>(max_queue_size_)
                    && !inter_queues_[2].empty()) {
                    in_flight_[3] = std::move(inter_queues_[2].front());
                    inter_queues_[2].pop_front();
                    stage4_.start(*in_flight_[3]);
                    any_work = true;
                }
            }
        }

        updateQueueSizes();
        if (!any_work) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    input_cv_.notify_all();
}

void AutoAimPipeline::updateQueueSizes()
{
    {
        std::lock_guard<std::mutex> lk(input_mtx_);
        queue_sizes_[0].store(static_cast<int>(input_queue_.size()));
    }
    queue_sizes_[1].store(static_cast<int>(inter_queues_[0].size()));
    queue_sizes_[2].store(static_cast<int>(inter_queues_[1].size()));
    queue_sizes_[3].store(static_cast<int>(inter_queues_[2].size()));
    {
        std::lock_guard<std::mutex> lk(output_mtx_);
        queue_sizes_[4].store(static_cast<int>(output_queue_.size()));
    }
}
