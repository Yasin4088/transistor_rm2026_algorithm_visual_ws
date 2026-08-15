#pragma once

#include "torque_controller/RobotController.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>

// 与旧串口 SerialData 语义对齐的状态快照（yaw 已是多圈连续角）
struct TorqueStateSnapshot {
    bool   valid = false;
    float  bullet_velocity = 0.0f;
    float  pitch = 0.0f;        // rad
    double total_yaw = 0.0;     // rad，多圈连续
    uint8_t color = 0;          // 0=红 1=蓝
    uint8_t auto_aim_switch = 0;
    float  chassis_roll = 0.0f;
};

/**
 * @brief TorqueController 桥接层
 *
 * 封装 RobotController（MCU+IMU 串口、融合、MPC、100Hz 下行发送线程），
 * 对外提供：
 *   - 轮询线程（默认 500Hz）读取 getState()，经回调上抛 MCU 状态快照；
 *   - setCommand() 转发下行指令（auto_aim_enable / target_yaw / pitch / fire）。
 *
 * 注意：RobotController 会占用 MCU 与 IMU 两个串口设备
 * （IMU 产品名为 AutoAim_IMU_Com），启用后旧 SerialCommunicationClass
 * 与 HeadIMU 串口必须停用，否则端口冲突。
 */
class TorqueBridge {
public:
    using Callback = std::function<void(const TorqueStateSnapshot&)>;

    TorqueBridge(double dt_control, int N,
                 double J, double tau_c, double b, double tau_d,
                 double max_torque, double max_torque_rate,
                 double Q, double R, double Rd, int max_iter,
                 Callback callback,
                 int poll_hz = 500);
    ~TorqueBridge();

    TorqueBridge(const TorqueBridge&) = delete;
    TorqueBridge& operator=(const TorqueBridge&) = delete;

    // 下行：直通 RobotController::set（后台 100Hz 求解 + 发送）
    void setCommand(bool auto_aim_enable, bool yaw_torque_only_mode,
                    double target_yaw, double pitch_target_angle, bool fire);

    // 停止轮询线程（幂等）；RobotController 析构时自动停串口/MPC 线程
    void stop();

private:
    void pollLoop();

    std::unique_ptr<RobotController> controller_;
    Callback callback_;
    std::atomic<bool> running_{true};
    std::thread poll_thread_;
    int poll_hz_ = 500;
};
