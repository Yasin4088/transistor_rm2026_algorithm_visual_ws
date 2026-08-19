// DirectMotionModel.h
// 直接运动模型：把装甲板当作一个自由 3D 点目标（匀速/匀加速 + 卡尔曼平滑）。
// 不假设圆周运动、不假设多块装甲板、不假设恒定角速度，适合手持单块板等场景。
#pragma once
#include <memory>
#include <cmath>
#include <Eigen/Dense>
#include "predictor/RotationMotionModel.h" // 复用 ObservedData / SimpleArmor / PredictResult

struct DirectMotionState {
    double x;
    double y;
    double z;
    double vx;
    double vy;
    double vz;
    double ax;
    double ay;
    double az;
    double yaw;
    unsigned long long update_frames;
};

class DirectMotionModel {
public:
    // 参数单位：位置 mm，速度 mm/s，加速度 mm/s^2
    DirectMotionModel(ObservedData& initObservedData,
                      double pos_noise_std = 10.0,   // PnP 位置噪声标准差（mm）
                      double jerk_std = 800.0,       // 过程噪声：加加速度标准差（mm/s^3）
                      double yaw_alpha = 0.3);       // yaw 低通滤波系数

    void update(ObservedData& observedData);
    void emptyUpdate(double update_time);
    PredictResult predict(double predictTime) const;
    DirectMotionState getState() const;

private:
    static constexpr int STATE_DIM = 9; // [px,py,pz,vx,vy,vz,ax,ay,az]
    static constexpr double MAX_DT = 0.5; // 单步最大时间间隔（秒），防止长时间丢帧后状态飞掉

    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
    Eigen::MatrixXd R_;

    double pos_noise_std_;
    double jerk_std_;
    double yaw_alpha_;
    double yaw_smoothed_;
    double last_update_time_;
    unsigned long long update_frames_ = 0;

    Eigen::MatrixXd buildF(double dt) const;
    Eigen::MatrixXd buildQ(double dt) const;
    void predictStep(double dt);
    void measurementUpdate(double px, double py, double pz);
    static double wrapAngle(double a);
};
