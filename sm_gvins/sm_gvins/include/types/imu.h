#pragma once

#include <memory>
#include "eigen_types.h"

struct IMU {
    IMU() = default;
    IMU(double t, const Vec3d& gyro, const Vec3d& accel) : timestamp_(t), gyro_(gyro), accel_(accel) {}

    double timestamp_ = 0.0;
    Vec3d gyro_  = Vec3d::Zero();
    Vec3d accel_ = Vec3d::Zero();
};

using IMUPtr = std::shared_ptr<IMU>;

