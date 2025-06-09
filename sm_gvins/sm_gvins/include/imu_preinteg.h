#pragma once

#include <memory>
#include "imu.h"
#include "nav_state.h"

class IMUPreintegration
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    struct Options {
        Options() {}
        Vec3d init_ba_ = Vec3d::Zero();  // 初始零偏
        Vec3d init_bg_ = Vec3d::Zero();  // 初始零偏
        double noise_gyro_ = 1e-2;       // 陀螺噪声，标准差
        double noise_acce_ = 1e-1;       // 加计噪声，标准差
    };

    IMUPreintegration(Options options = Options());
    void AddIMU(double dt, const IMU& imu);
    NavStated Predict(const NavStated& start, const Vec3d &grav = Vec3d(0,0,-9.81)) const;
    SO3 GetDeltaRotation(const Vec3d &bg);
    Vec3d GetDeltaVelocity(const Vec3d &bg, const Vec3d &ba);
    Vec3d GetDeltaPosition(const Vec3d &bg, const Vec3d &ba);

    double dt_ = 0;                          // 整体预积分时间
    Mat9d cov_ = Mat9d::Zero();              // 累计噪声矩阵
    Mat6d noise_gyro_acce_ = Mat6d::Zero();  // 测量噪声矩阵

    // 零偏
    Vec3d bg_ = Vec3d::Zero();
    Vec3d ba_ = Vec3d::Zero();

    // 预积分观测量
    SO3 dR_;
    Vec3d dv_ = Vec3d::Zero();
    Vec3d dp_ = Vec3d::Zero();

    // 雅可比矩阵
    Mat3d dR_dbg_ = Mat3d::Zero();
    Mat3d dV_dbg_ = Mat3d::Zero();
    Mat3d dV_dba_ = Mat3d::Zero();
    Mat3d dP_dbg_ = Mat3d::Zero();
    Mat3d dP_dba_ = Mat3d::Zero();
};

using IMUPreintegrationPtr = std::shared_ptr<IMUPreintegration>;