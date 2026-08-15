#include "communication/TorqueBridge.h"

#include <chrono>

TorqueBridge::TorqueBridge(double dt_control, int N,
                           double J, double tau_c, double b, double tau_d,
                           double max_torque, double max_torque_rate,
                           double Q, double R, double Rd, int max_iter,
                           Callback callback, int poll_hz)
    : controller_(std::make_unique<RobotController>(
          dt_control, N, J, tau_c, b, tau_d,
          max_torque, max_torque_rate, Q, R, Rd, max_iter))
    , callback_(std::move(callback))
    , poll_hz_(poll_hz > 0 ? poll_hz : 500)
{
    poll_thread_ = std::thread(&TorqueBridge::pollLoop, this);
}

TorqueBridge::~TorqueBridge()
{
    stop();
    // controller_ 析构：mcu_mpc_ 停 100Hz 发送线程，comm_ 停串口线程
}

void TorqueBridge::stop()
{
    if (running_.exchange(false)) {
        if (poll_thread_.joinable()) {
            poll_thread_.join();
        }
    }
}

void TorqueBridge::setCommand(bool auto_aim_enable, bool yaw_torque_only_mode,
                              double target_yaw, double pitch_target_angle,
                              bool fire)
{
    controller_->set(auto_aim_enable, yaw_torque_only_mode,
                     target_yaw, pitch_target_angle, fire);
}

void TorqueBridge::pollLoop()
{
    const auto period = std::chrono::microseconds(1000000 / poll_hz_);
    while (running_.load()) {
        const auto t0 = std::chrono::steady_clock::now();

        const auto st = controller_->getState();
        if (st.mcu.valid && callback_) {
            TorqueStateSnapshot snap;
            snap.valid           = true;
            snap.bullet_velocity = st.mcu.bullet_velocity;
            snap.pitch           = st.mcu.pitch_angle;
            snap.total_yaw       = st.mcu.yaw_angle;   // 已多圈连续
            snap.color           = st.mcu.color;
            snap.auto_aim_switch = st.mcu.auto_aim_switch;
            snap.chassis_roll    = st.fused.valid ? (float)st.fused.chassis_roll : 0.0f;
            callback_(snap);
        }

        const auto elapsed = std::chrono::steady_clock::now() - t0;
        if (elapsed < period) {
            std::this_thread::sleep_for(period - elapsed);
        }
    }
}
