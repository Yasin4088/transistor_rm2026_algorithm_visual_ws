// ArmorDetect_Node.cpp
#define _USE_MATH_DEFINES // 启用数学常量

#include <cmath>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include "2d_armor_detector/Params.h"
#include "camera/Camera.h"
#include "communication/Com.h"
#include "communication/HeadIMU.h"
#include "communication/WatchdogClient.h"
#include "macro/AutoAimMacro.h"
#include "other_input/ImagesInput.h"
#include "other_input/VideoInput.h"
#include "pipeline/AutoAimPipeline.h"
#include "utils/PerformanceMonitor.h"

namespace fs = std::filesystem;

// 全局变量定义
cv::Mat g_image;
pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
bool g_bExit = false;
bool image_used = true;

class ArmorDetectNode : public rclcpp::Node {
public:
    ArmorDetectNode() : Node("armor_detect_node") {
        node_start_time = std::chrono::steady_clock::now();

        // 路径与配置
        // 获取可执行文件路径
        char exec_path[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", exec_path, sizeof(exec_path) - 1);
        if (len == -1) {
            perror("readlink");
            return;
        }
        exec_path[len] = '\0';
        RCLCPP_INFO(this->get_logger(), "info from C++ | Path: %s\n", exec_path);

        fs::path full_path(exec_path);
        std::string full_path_str = full_path.string(); //转换为字符串，便于查找

        // 获取工作空间路径
        // 从可执行文件路径向上查找工作空间根目录（含 src/shared_files/config.yaml 的目录）
        fs::path ws_dir_path = full_path.parent_path();
        while (!ws_dir_path.empty() &&
               !fs::exists(ws_dir_path / "src" / "shared_files" / "config.yaml")) {
            ws_dir_path = ws_dir_path.parent_path();
        }
        if (ws_dir_path.empty() ||
            !fs::exists(ws_dir_path / "src" / "shared_files" / "config.yaml")) {
            std::cerr << "Error: Workspace directory not found in path" << std::endl;
            return;
        }

        // 获取配置文件路径并加载
        const std::string config_file_relative_path = "src/shared_files/config.yaml";
        fs::path config_file_path = ws_dir_path / config_file_relative_path;

        config_file_ptr = std::make_shared<YAML::Node>(YAML::LoadFile(config_file_path));

        // 参数初始化
        // 初始化敌方颜色
#ifdef FIX_ENEMY_COLOR
        enemy_color_ = (FIX_ENEMY_COLOR == 0) ? "RED" : "BLUE";
#else
        enemy_color_ = (*config_file_ptr)["init_enemy_color"].as<std::string>();
#endif

        // 初始化子弹速度
#ifdef FIX_BULLET_VELOCITY
        bullet_velocity_ = FIX_BULLET_VELOCITY;
#else
        bullet_velocity_ = (*config_file_ptr)["bullet_velocity_"].as<float>();
#endif

        // 输入源初始化
#ifdef USE_VIDEO
        video_input_ = std::make_shared<VideoInput>(ws_dir_path / (*config_file_ptr)["video_relative_path"].as<std::string>());
#else
#ifdef USE_IMAGES
        images_input_ = std::make_shared<ImagesInput>(ws_dir_path / (*config_file_ptr)["images_relative_path"].as<std::string>());
#else
        camera_ = std::make_shared<Camera>((*config_file_ptr)["cam_ip"].as<std::string>(), (*config_file_ptr)["pc_ip"].as<std::string>());
        camera_->setExposureTime((*config_file_ptr)["camera_ExposureTime"].as<float>());
        camera_->setGain((*config_file_ptr)["camera_Gain"].as<float>());
        camera_->start();
#endif
#endif

        // 自瞄参数初始化
        // 根据相机内参自动提取
        const YAML::Node& camera_matrix_Node = (*config_file_ptr)["camera_matrix"];
        yaw_rad_to_x_pixel_ratio = camera_matrix_Node[0][0].as<float>(); 
        pitch_rad_to_y_pixel_ratio = camera_matrix_Node[1][1].as<float>(); 

        params_.min_light_height = (*config_file_ptr)["min_light_height"].as<int>();
        params_.light_min_area = (*config_file_ptr)["light_min_area"].as<int>();
        params_.light_max_area = (*config_file_ptr)["light_max_area"].as<int>();
        params_.max_light_wh_ratio = (*config_file_ptr)["max_light_wh_ratio"].as<float>();
        params_.min_light_wh_ratio = (*config_file_ptr)["min_light_wh_ratio"].as<float>();
        params_.light_max_tilt_angle = (*config_file_ptr)["light_max_tilt_angle"].as<float>();
        
        frame_rate_ = (*config_file_ptr)["frame_rate"].as<float>();
        serial_delay_time = (*config_file_ptr)["serial_delay_time"].as<float>();

        if (enemy_color_ == "RED") {
            params_.enemy_color = Params::RED;
        } else if (enemy_color_ == "BLUE") {
            params_.enemy_color = Params::BLUE;
        } else if (enemy_color_ == "GREEN") {
            params_.enemy_color = Params::GREEN;
        } else if (enemy_color_ == "BOTH") {
            params_.enemy_color = Params::BOTH;
        } else {
            enemy_color_ = "GREEN";
            params_.enemy_color = Params::GREEN; // 处理错误情况，设置默认值
        }

        com_data_visualize_frame = cv::Mat::zeros(480, 640, CV_8UC3);

        // 算法流水线初始化
        // 性能统计：每 90 帧输出一次各阶段平均耗时
        profiler_ = std::make_shared<PerformanceMonitor>(90);
        auto_aim_pipeline_ = std::make_shared<AutoAimPipeline>(
            config_file_ptr,
            this,
            ws_dir_path,
            node_start_time);
        auto_aim_pipeline_->setProfiler(profiler_);

        // 串口与后台任务初始化
        DelayInfos init_serial_infos;
        init_serial_infos.last_pitch_rad_ = 0.0;
        init_serial_infos.last_yaw_rad_ = 0.0;
        init_serial_infos.total_yaw_rad_ = 0.0;
        init_serial_infos.last_roll_rad_ = 0.0;
        init_serial_infos.push_time = node_start_time;
        serial_infos_delay_.push(init_serial_infos);

        // 串口通信器初始化
        serial_communication_ = std::make_shared<SerialCommunicationClass>(this, std::bind(&ArmorDetectNode::serialDataCallback, this, std::placeholders::_1));

        com_timer_thread_ = std::thread(std::bind(&SerialCommunicationClass::timerThread, serial_communication_));

        headIMUInfos.headIMU_communication_ = std::make_shared<HeadIMUSerialCommunicationClass>(std::bind(&ArmorDetectNode::headIMUSerialDataCallback, this, std::placeholders::_1));
        headIMUInfos.headIMU_timer_thread_ = std::thread(std::bind(&HeadIMUSerialCommunicationClass::timerThread, headIMUInfos.headIMU_communication_));

        // 串口通信下位机初始化
        serial_communication_->sendData(0, 0, false);

        watchdog_client = std::make_shared<WatchdogClient>();
        watchdog_client->init();
        watchdog_client->feed();
        last_feed_dog_time = std::chrono::steady_clock::now();

#ifdef DEBUG_CODE
        debug_code();
#endif

        // 主线程创建
        main_loop_thread_ = std::thread(std::bind(&ArmorDetectNode::main_loop_func, this));

        RCLCPP_INFO(this->get_logger(), "ArmorDetectNode initialized");
    }

    ~ArmorDetectNode() {
        std::cerr << "[diag] node dtor: begin" << std::endl;
        // 退出主循环与视频取流线程
        g_bExit = true;
        // 停止两个串口通信线程（running = false）
        serial_communication_->~SerialCommunicationClass();
        std::cerr << "[diag] node dtor: serial stopped" << std::endl;
        headIMUInfos.headIMU_communication_->~HeadIMUSerialCommunicationClass();
        std::cerr << "[diag] node dtor: imu stopped" << std::endl;
        // join 所有 std::thread 成员，避免析构时 terminate
        if (com_timer_thread_.joinable()) com_timer_thread_.join();
        std::cerr << "[diag] node dtor: com timer joined" << std::endl;
        if (headIMUInfos.headIMU_timer_thread_.joinable()) headIMUInfos.headIMU_timer_thread_.join();
        std::cerr << "[diag] node dtor: imu timer joined" << std::endl;
        if (main_loop_thread_.joinable()) main_loop_thread_.join();
        std::cerr << "[diag] node dtor: main loop joined" << std::endl;
        // 先销毁流水线（join 所有 stage 线程），避免 cv::destroyAllWindows
        // 与 stage4 的 cv::imshow 并发访问 HighGUI 导致段错误
        auto_aim_pipeline_.reset();
        std::cerr << "[diag] node dtor: pipeline stopped" << std::endl;
        cv::destroyAllWindows();
        pthread_mutex_destroy(&g_mutex);
        RCLCPP_INFO(this->get_logger(), "ArmorDetectNode destroyed");
    }

private:

    // 成员变量定义
    // 配置与线程
    std::shared_ptr<YAML::Node> config_file_ptr; 
    std::thread com_timer_thread_;
    std::thread main_loop_thread_;

    // 输入源
    std::shared_ptr<Camera> camera_;
    std::shared_ptr<VideoInput> video_input_;
    std::shared_ptr<ImagesInput> images_input_;
    float frame_rate_;

    // 时间与基础参数
    std::chrono::time_point<std::chrono::steady_clock> node_start_time;
    float bullet_velocity_;

    // MCU 姿态状态
    float last_pitch_rad_mcu_;
    float last_yaw_rad_mcu_;
    float total_yaw_rad_mcu_;
    int current_yaw_circle_mcu_ = 0;

    // Head IMU 姿态状态
    float last_pitch_rad_imu_;
    float last_yaw_rad_imu_;
    float total_yaw_rad_imu_;
    float last_roll_rad_imu_;
    int current_yaw_circle_imu_ = 0;

    // 延迟对齐后的本帧输入状态
    float last_pitch_rad_delayed_ = 0;
    float last_yaw_rad_delayed_ = 0;
    float total_yaw_rad_delayed_ = 0;
    float last_roll_rad_delayed_ = 0;

    struct DelayInfos {
        float last_pitch_rad_;
        float last_yaw_rad_;
        float last_roll_rad_;
        float total_yaw_rad_;
        std::chrono::steady_clock::time_point push_time;
    };
    std::queue<DelayInfos> serial_infos_delay_;
    float serial_delay_time;
    std::string enemy_color_;
    Params params_;

#ifdef SAVE_IMG_FREQ
    long long frame_count_ = 0;
#endif

    cv::Point2f ground_stable_point;
    float yaw_rad_to_x_pixel_ratio;
    float pitch_rad_to_y_pixel_ratio;

    // 外设、可视化与流水线
    std::shared_ptr<SerialCommunicationClass> serial_communication_;
    std::shared_ptr<WatchdogClient> watchdog_client;
    std::chrono::steady_clock::time_point last_feed_dog_time;
    cv::Mat com_data_visualize_frame;
    bool com_data_visualize_frame_used = true;
    std::shared_ptr<AutoAimPipeline> auto_aim_pipeline_;
    std::shared_ptr<PerformanceMonitor> profiler_;

    // Head IMU 通信状态
    struct {
        std::shared_ptr<HeadIMUSerialCommunicationClass> headIMU_communication_;
        std::thread headIMU_timer_thread_;

        bool use_head_imu = true;

        float head_imu_yaw;
        float head_imu_pitch;
        float head_imu_roll;

        float mcu_yaw;
        float mcu_pitch;
        
        float last_mcu_yaw;
        float latest_head_imu_yaw_when_mcu_yaw_update;
        std::chrono::steady_clock::time_point last_mcu_yaw_update_time;
        bool mcu_yaw_online = true;
        float last_mcu_command_yaw;
        float latest_mcu_command_yaw_when_mcu_yaw_update;

        float to_mcu_delta_yaw;
        float to_mcu_delta_pitch;

        bool last_auto_aim_switch = true; // 用于在开启自瞄时进行校准
    } headIMUInfos;
    
    // 主循环
    void main_loop_func() {
        while (!g_bExit) {
            std::chrono::steady_clock::time_point loop_start_time = std::chrono::steady_clock::now();
            processImage();
            std::this_thread::sleep_until(loop_start_time + std::chrono::microseconds(static_cast<int>(1e6 / frame_rate_)));
        }
    }

    // 调试与校准
    void debug_code() {
        std::thread([&]() {
            double debug_time_count = 0.0;
            while (true) {
                auto start = std::chrono::steady_clock::now();

                SerialData fakeSerialData;
                fakeSerialData.bullet_velocity = 25.0;  // 子弹速度
                fakeSerialData.bullet_angle = std::sin(debug_time_count * 0.5 * (2*M_PI)) * 1.8 / 30 * 15;    // 子弹角度
                fakeSerialData.gimbal_yaw =  
                    static_cast<int16_t>(std::atan2(std::sin(debug_time_count * 0.3), std::cos(debug_time_count * 0.3)) * 4095.0 / M_PI);
                fakeSerialData.color = 1;            // 敌方颜色(0:红色, 1:蓝色)

                serialDataCallback(fakeSerialData);

                std::this_thread::sleep_until(start + std::chrono::microseconds(10000));  // 大约10ms周期
                debug_time_count += 0.01;
            }
        }).detach();
    }

    // Head IMU 校准
    void recalibrateHeadIMU() {
        float start_yaw = last_yaw_rad_imu_ + headIMUInfos.to_mcu_delta_yaw;
        float start_pitch = last_pitch_rad_delayed_;

        if (auto_aim_pipeline_) {
            auto_aim_pipeline_->resetYawIntegration();
        }

        for (int i = 0; i < 20; i++) {
            serial_communication_->sendData(0.0, start_yaw, false);
            usleep(30*1000);
        }

        float new_yaw = last_yaw_rad_imu_;

        float delta_yaw = new_yaw - start_yaw;

        headIMUInfos.to_mcu_delta_yaw = -delta_yaw;

        serial_communication_->sendData(start_pitch, start_yaw + headIMUInfos.to_mcu_delta_yaw, false);
    }

    // 串口与 IMU 回调
    void headIMUSerialDataCallback(const HeadIMUSerialData& msg) {

        float current_pitch_;
        float current_yaw_;
        float current_roll_;
        float last_pitch_rad_;
        float last_yaw_rad_;
        float total_yaw_rad_;

        current_pitch_ = msg.euler_pitch;
        current_yaw_ = msg.euler_yaw;
        current_roll_ = msg.euler_roll;

        headIMUInfos.head_imu_yaw = msg.euler_yaw;
        headIMUInfos.head_imu_pitch = msg.euler_pitch;
        headIMUInfos.head_imu_roll = msg.euler_roll;
        headIMUInfos.to_mcu_delta_pitch = headIMUInfos.mcu_pitch - headIMUInfos.head_imu_pitch;

        while (current_yaw_ < -M_PI) {
            current_yaw_ += 2 * M_PI;
        }
        while (current_yaw_ > M_PI) {
            current_yaw_ -= 2 * M_PI;
        }
        
        if (current_yaw_ < -M_PI/2 && last_yaw_rad_imu_ > M_PI/2) {
            current_yaw_circle_imu_ += 1;
        } else if (current_yaw_ > M_PI/2 && last_yaw_rad_imu_ < -M_PI/2) {
            current_yaw_circle_imu_ -= 1;
        }

        total_yaw_rad_imu_ = current_yaw_circle_imu_ * 2 * M_PI + current_yaw_;
        last_pitch_rad_imu_ = current_pitch_;
        last_yaw_rad_imu_ = current_yaw_;
        last_roll_rad_imu_ = current_roll_;

        if (headIMUInfos.use_head_imu) {
            std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
            DelayInfos now_serial_infos;
            now_serial_infos.last_pitch_rad_ = last_pitch_rad_imu_;
            now_serial_infos.last_pitch_rad_ = last_pitch_rad_imu_; // last_pitch_rad_mcu_
            now_serial_infos.last_roll_rad_ = last_roll_rad_imu_;
            now_serial_infos.last_yaw_rad_ = last_yaw_rad_imu_;
            now_serial_infos.total_yaw_rad_ = total_yaw_rad_imu_;
            now_serial_infos.push_time = current_time;
            serial_infos_delay_.push(now_serial_infos);
        }
    }

    // 串口数据回调
    void serialDataCallback(const SerialData& msg) {
        if (com_data_visualize_frame_used) {
            const MCUDataFrame& odf = msg.origin_data_frame;
            com_data_visualize_frame.setTo(cv::Scalar(0, 0, 0));
            cv::putText(com_data_visualize_frame, 
                cv::format("bullet_velocity: %.6f", odf.bullet_velocity), 
                cv::Point(20, 20),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            cv::putText(com_data_visualize_frame, 
                cv::format("bullet_angle: %.6f", odf.bullet_angle), 
                cv::Point(20, 50),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            cv::putText(com_data_visualize_frame, 
                cv::format("gimbal_yaw: %d", odf.gimbal_yaw), 
                cv::Point(20, 80),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            cv::putText(com_data_visualize_frame, 
                cv::format("mark: %u", odf.mark), 
                cv::Point(20, 110),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            cv::putText(com_data_visualize_frame, 
                cv::format("color: %u", odf.color), 
                cv::Point(20, 140),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            cv::putText(com_data_visualize_frame, 
                cv::format("z_rotation_velocity: %.6f", odf.z_rotation_velocity), 
                cv::Point(20, 170),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            com_data_visualize_frame_used = false;
        }

        SerialData processed_msg = msg;
#ifdef FIX_ENEMY_COLOR
        processed_msg.color = FIX_ENEMY_COLOR;
#endif
#ifdef FIX_BULLET_VELOCITY
        processed_msg.bullet_velocity = FIX_BULLET_VELOCITY;
#endif

        float current_pitch_;
        float current_yaw_;

        bullet_velocity_ = processed_msg.bullet_velocity;
        current_pitch_ = ((float)(processed_msg.bullet_angle)) * 30 / 1.8 * M_PI / 180; // 测定pitch轴传入数据1.8大约对应30°
        current_yaw_ = ((float)(processed_msg.gimbal_yaw)) * M_PI / 4096.0;  // 一圈对应[-4096, 4095]

        headIMUInfos.mcu_yaw = current_yaw_;
        headIMUInfos.mcu_pitch = current_pitch_;
        if (headIMUInfos.last_mcu_yaw != headIMUInfos.mcu_yaw) {
            headIMUInfos.latest_head_imu_yaw_when_mcu_yaw_update = headIMUInfos.head_imu_yaw;
            headIMUInfos.last_mcu_yaw = current_yaw_;
            headIMUInfos.last_mcu_yaw_update_time = std::chrono::steady_clock::now();
            headIMUInfos.mcu_yaw_online = true;
            headIMUInfos.latest_mcu_command_yaw_when_mcu_yaw_update = headIMUInfos.last_mcu_command_yaw;
            headIMUInfos.to_mcu_delta_yaw = headIMUInfos.mcu_yaw - headIMUInfos.latest_head_imu_yaw_when_mcu_yaw_update;
        }
        headIMUInfos.to_mcu_delta_pitch = headIMUInfos.mcu_pitch - headIMUInfos.head_imu_pitch;

        while (current_yaw_ < -M_PI) {
            current_yaw_ += 2 * M_PI;
        }
        while (current_yaw_ > M_PI) {
            current_yaw_ -= 2 * M_PI;
        }
        enemy_color_ = (processed_msg.color == 0) ? "RED" : "BLUE";
        if (enemy_color_ == "RED") {
            params_.enemy_color = Params::RED;
        } else if (enemy_color_ == "BLUE") {
            params_.enemy_color = Params::BLUE;
        }
        if (current_yaw_ < -M_PI/2 && last_yaw_rad_mcu_ > M_PI/2) {
            current_yaw_circle_mcu_ += 1;
        } else if (current_yaw_ > M_PI/2 && last_yaw_rad_mcu_ < -M_PI/2) {
            current_yaw_circle_mcu_ -= 1;
        }

        total_yaw_rad_mcu_ = current_yaw_circle_mcu_ * 2 * M_PI + current_yaw_;
        last_pitch_rad_mcu_ = current_pitch_;
        last_yaw_rad_mcu_ = current_yaw_;

        RCLCPP_DEBUG(this->get_logger(), 
            "Received serial data: v=%.2f, pitch=%.2f, yaw=%.2f, color=%s \nyaw_circle=%d, total_yaw_rad=%.2f",
            bullet_velocity_, current_pitch_, current_yaw_, enemy_color_.c_str(),
            current_yaw_circle_mcu_, total_yaw_rad_mcu_);

        if (!headIMUInfos.use_head_imu) {
            std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
            DelayInfos now_serial_infos;
            now_serial_infos.last_pitch_rad_ = last_pitch_rad_mcu_;
            now_serial_infos.last_yaw_rad_ = last_yaw_rad_mcu_;
            now_serial_infos.last_roll_rad_ = 0.0;
            now_serial_infos.total_yaw_rad_ = total_yaw_rad_mcu_;
            now_serial_infos.push_time = current_time;
            serial_infos_delay_.push(now_serial_infos);
        }
    }

    // 图像投喂与结果处理
    void processImage() {
        auto now = std::chrono::steady_clock::now();

        if (std::chrono::duration_cast<std::chrono::milliseconds>(
            now - headIMUInfos.last_mcu_yaw_update_time).count() > 3000
        ) {
            if (fabs(headIMUInfos.last_mcu_command_yaw - headIMUInfos.latest_mcu_command_yaw_when_mcu_yaw_update)
                > 5.0 * M_PI / 180.0
            ) {
                headIMUInfos.mcu_yaw_online = false;
            }
        }

        while (serial_infos_delay_.size() > 1 &&
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - serial_infos_delay_.front().push_time).count() > serial_delay_time) {
            serial_infos_delay_.pop();
        }

        DelayInfos delayed_serial_infos = serial_infos_delay_.front();
        last_pitch_rad_delayed_ = delayed_serial_infos.last_pitch_rad_;
        last_yaw_rad_delayed_ = delayed_serial_infos.last_yaw_rad_;
        total_yaw_rad_delayed_ = delayed_serial_infos.total_yaw_rad_;
        last_roll_rad_delayed_ = delayed_serial_infos.last_roll_rad_;
        ground_stable_point = cv::Point2f(
            500 + total_yaw_rad_delayed_ * yaw_rad_to_x_pixel_ratio,
            500 + last_pitch_rad_delayed_ * pitch_rad_to_y_pixel_ratio);
        RCLCPP_DEBUG(this->get_logger(), "ground_stable_point: %.2f %.2f",
            ground_stable_point.x, ground_stable_point.y);

        cv::Mat frame;
#if defined(USE_VIDEO) || defined(USE_IMAGES) || defined(SYNC_CAMERA_FPS)
        while (image_used && !g_bExit) {
            usleep(1000);
        }
#endif
        pthread_mutex_lock(&g_mutex);
        if (!g_image.empty()) {
            frame = g_image.clone();
            image_used = true;
        }
        pthread_mutex_unlock(&g_mutex);

        bool auto_aim_switch = true;
        if (((!headIMUInfos.last_auto_aim_switch) && auto_aim_switch) &&
            (headIMUInfos.use_head_imu && (!headIMUInfos.mcu_yaw_online))
        ) {
            recalibrateHeadIMU();
        }
        headIMUInfos.last_auto_aim_switch = auto_aim_switch;

        if (!frame.empty()) {
#ifdef SAVE_IMG_FREQ
            frame_count_ += 1;
            if (frame_count_ % SAVE_IMG_FREQ == 0 && frame_count_ / SAVE_IMG_FREQ < 2000) {
                fs::create_directories("camera_images");
                std::ostringstream filename;
                filename << "camera_images/"
                         << std::setw(5) << std::setfill('0') << (frame_count_ / SAVE_IMG_FREQ)
                         << ".jpg";
                cv::imwrite(filename.str(), frame);
            }
#endif

            AutoAimPipelineData::InitialData initial;
            initial.frame = std::move(frame);
            initial.com_data_visualize_frame = com_data_visualize_frame.clone();
            initial.frame_timestamp = now;
            initial.node_start_time = node_start_time;
            initial.bullet_velocity = bullet_velocity_;
            initial.enemy_color = enemy_color_;
            initial.pitch = last_pitch_rad_delayed_;
            initial.yaw = last_yaw_rad_delayed_;
            initial.total_yaw = total_yaw_rad_delayed_;
            initial.roll = last_roll_rad_delayed_;
            initial.ground_stable_point = ground_stable_point;
            initial.auto_aim_switch = auto_aim_switch;
            initial.use_head_imu = headIMUInfos.use_head_imu;
            initial.mcu_yaw_online = headIMUInfos.mcu_yaw_online;
            initial.to_mcu_delta_yaw = headIMUInfos.to_mcu_delta_yaw;
            initial.to_mcu_delta_pitch = headIMUInfos.to_mcu_delta_pitch;
            auto_aim_pipeline_->addFrame(std::move(initial));
        }

        AutoAimPipeline::ProcessResult result = auto_aim_pipeline_->tryPopResult(now);
        if (!result.valid) {
            return;
        }

        headIMUInfos.last_mcu_command_yaw = result.valid_data.mcu_command_yaw;
        if (result.valid_data.should_send_reset) {
            serial_communication_->sendData(0.0, 0.0, false);
        } else {
            serial_communication_->sendData(
                result.valid_data.mcu_command_pitch,
                result.valid_data.mcu_command_yaw,
                result.valid_data.predictor_result.fire_flag);
        }

#if (defined LOG_RESULT_VIDEO) || (defined LOG_ORIGIN_VIDEO)
        if (result.valid_data.request_com_frame_refresh) {
            com_data_visualize_frame_used = true;
        }
#endif

#ifdef SHOW_WINDOWS
        if (!result.valid_data.rmm_visualize_frame.empty()) {
            cv::imshow("RMM visualize", result.valid_data.rmm_visualize_frame);
        }
        if (!result.valid_data.common_debug_oscilloscope_frame.empty()) {
            cv::imshow("Common Debug Oscilloscope", result.valid_data.common_debug_oscilloscope_frame);
        }
        if (!result.valid_data.yaw_visualizer_frame.empty()) {
            cv::imshow("Yaw Visualizer", result.valid_data.yaw_visualizer_frame);
        }
        if (!result.valid_data.display.empty()) {
            cv::imshow("Armor Detection", result.valid_data.display);
        }
        cv::waitKey(1);
#endif

        if (std::chrono::steady_clock::now() - last_feed_dog_time >= std::chrono::seconds(3)) {
            watchdog_client->feed();
            last_feed_dog_time = std::chrono::steady_clock::now();
        }

        RCLCPP_INFO(this->get_logger(),
            "armor_count: %zu | Q[in:%d i0:%d i1:%d i2:%d out:%d]",
            result.valid_data.armor_count,
            result.always_valid_data.queue_input,
            result.always_valid_data.queue_inter0,
            result.always_valid_data.queue_inter1,
            result.always_valid_data.queue_inter2,
            result.always_valid_data.queue_output);
    }
};

std::shared_ptr<ArmorDetectNode> node;
void signalHandler(int signum) {
    if (node) {
        rclcpp::shutdown();
    }
}

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    node = std::make_shared<ArmorDetectNode>();
    signal(SIGINT, signalHandler);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
