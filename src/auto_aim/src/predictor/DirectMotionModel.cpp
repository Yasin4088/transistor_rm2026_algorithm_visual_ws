// DirectMotionModel.cpp
#include "predictor/DirectMotionModel.h"

using namespace Eigen;

DirectMotionModel::DirectMotionModel(ObservedData& initObservedData,
                                     double pos_noise_std,
                                     double jerk_std,
                                     double yaw_alpha)
    : pos_noise_std_(pos_noise_std),
      jerk_std_(jerk_std),
      yaw_alpha_(yaw_alpha),
      yaw_smoothed_(initObservedData.yaw),
      last_update_time_(initObservedData.t) {
    x_ = VectorXd::Zero(STATE_DIM);
    x_(0) = initObservedData.x;
    x_(1) = initObservedData.y;
    x_(2) = initObservedData.z;

    P_ = MatrixXd::Identity(STATE_DIM, STATE_DIM);
    P_(0, 0) = 100.0 * 100.0;  // 位置初始不确定度 (mm)^2
    P_(1, 1) = P_(0, 0);
    P_(2, 2) = P_(0, 0);
    P_(3, 3) = 1000.0 * 1000.0; // 速度 (mm/s)^2
    P_(4, 4) = P_(3, 3);
    P_(5, 5) = P_(3, 3);
    P_(6, 6) = 5000.0 * 5000.0; // 加速度 (mm/s^2)^2
    P_(7, 7) = P_(6, 6);
    P_(8, 8) = P_(6, 6);

    R_ = MatrixXd::Identity(3, 3) * (pos_noise_std_ * pos_noise_std_);
    update_frames_ = 1;
}

MatrixXd DirectMotionModel::buildF(double dt) const {
    MatrixXd F = MatrixXd::Identity(STATE_DIM, STATE_DIM);
    double dt2 = 0.5 * dt * dt;
    for (int i = 0; i < 3; ++i) {
        int p = 3 * i;
        int v = 3 * i + 1;
        int a = 3 * i + 2;
        F(p, v) = dt;
        F(p, a) = dt2;
        F(v, a) = dt;
    }
    return F;
}

MatrixXd DirectMotionModel::buildQ(double dt) const {
    // 分段常值加加速度（jerk）白噪声模型
    MatrixXd Q = MatrixXd::Zero(STATE_DIM, STATE_DIM);
    double s2j = jerk_std_ * jerk_std_;
    double qpp = std::pow(dt, 5) / 20.0 * s2j;
    double qpv = std::pow(dt, 4) / 8.0 * s2j;
    double qpa = std::pow(dt, 3) / 6.0 * s2j;
    double qvv = std::pow(dt, 3) / 3.0 * s2j;
    double qva = std::pow(dt, 2) / 2.0 * s2j;
    double qaa = dt * s2j;
    for (int i = 0; i < 3; ++i) {
        int p = 3 * i;
        int v = 3 * i + 1;
        int a = 3 * i + 2;
        Q(p, p) = qpp; Q(p, v) = qpv; Q(p, a) = qpa;
        Q(v, p) = qpv; Q(v, v) = qvv; Q(v, a) = qva;
        Q(a, p) = qpa; Q(a, v) = qva; Q(a, a) = qaa;
    }
    return Q;
}

void DirectMotionModel::predictStep(double dt) {
    MatrixXd F = buildF(dt);
    MatrixXd Q = buildQ(dt);
    x_ = F * x_;
    P_ = F * P_ * F.transpose() + Q;
}

void DirectMotionModel::measurementUpdate(double px, double py, double pz) {
    MatrixXd H = MatrixXd::Zero(3, STATE_DIM);
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    H(2, 2) = 1.0;

    VectorXd z(3);
    z << px, py, pz;

    VectorXd z_pred = H * x_;
    MatrixXd S = H * P_ * H.transpose() + R_;
    MatrixXd K = P_ * H.transpose() * S.inverse();
    x_ = x_ + K * (z - z_pred);

    MatrixXd I = MatrixXd::Identity(STATE_DIM, STATE_DIM);
    P_ = (I - K * H) * P_;
    P_ = 0.5 * (P_ + P_.transpose()); // 保持对称，防止数值误差累积
}

double DirectMotionModel::wrapAngle(double a) {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

void DirectMotionModel::update(ObservedData& observedData) {
    double dt = observedData.t - last_update_time_;
    if (dt < 0.0) dt = 0.0;
    if (dt > MAX_DT) dt = MAX_DT;
    if (dt > 0.0) predictStep(dt);

    measurementUpdate(observedData.x, observedData.y, observedData.z);

    // yaw 单独做低通平滑，不参与位置预测（角度变化不代表位置变化）
    double yaw_diff = wrapAngle(observedData.yaw - yaw_smoothed_);
    yaw_smoothed_ = wrapAngle(yaw_smoothed_ + yaw_alpha_ * yaw_diff);

    last_update_time_ = observedData.t;
    update_frames_ += 1;
}

void DirectMotionModel::emptyUpdate(double update_time) {
    double dt = update_time - last_update_time_;
    if (dt < 0.0) dt = 0.0;
    if (dt > MAX_DT) dt = MAX_DT;
    if (dt > 0.0) predictStep(dt);
    last_update_time_ = update_time;
    // 不喂回预测值、不计数，状态只做时间推进
}

PredictResult DirectMotionModel::predict(double predictTime) const {
    if (predictTime < 0.0) predictTime = 0.0;

    PredictResult result;
    double px = x_(0), py = x_(1), pz = x_(2);
    double vx = x_(3), vy = x_(4), vz = x_(5);
    double ax = x_(6), ay = x_(7), az = x_(8);

    result.center_x = px + vx * predictTime + 0.5 * ax * predictTime * predictTime;
    result.center_y = py + vy * predictTime + 0.5 * ay * predictTime * predictTime;
    result.center_z = pz + vz * predictTime + 0.5 * az * predictTime * predictTime;
    result.z_another = result.center_z;
    result.r_now = 0.0;
    result.r_another = 0.0;
    result.yaw = yaw_smoothed_;
    result.rotation_direction = 1;

    // 单块板：直接预测它的位置
    result.armors.push_back(SimpleArmor({
        result.center_x,
        result.center_y,
        result.center_z,
        0.0,
        result.yaw
    }));
    return result;
}

DirectMotionState DirectMotionModel::getState() const {
    DirectMotionState state;
    state.x = x_(0);
    state.y = x_(1);
    state.z = x_(2);
    state.vx = x_(3);
    state.vy = x_(4);
    state.vz = x_(5);
    state.ax = x_(6);
    state.ay = x_(7);
    state.az = x_(8);
    state.yaw = yaw_smoothed_;
    state.update_frames = update_frames_;
    return state;
}
