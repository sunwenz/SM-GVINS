#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include "imu.h"
#include "eigen_types.h"
#include "nav_state.h"

class utils{
public:
    // 零速阈值, rad/s, m/s^2
    static constexpr double ZERO_VELOCITY_GYR_THRESHOLD = 0.002;
    static constexpr double ZERO_VELOCITY_ACC_THRESHOLD = 0.1;

    static bool detectZeroVelocity(const std::vector<IMU>& imu_buffer) {
        // 检查输入
        if (imu_buffer.empty()) {
            // LOG(WARNING) << "Empty IMU buffer in detectZeroVelocity";
            return false;
        }

        auto size = static_cast<double>(imu_buffer.size());
        double size_invert = 1.0 / size;

        // 计算均值
        std::vector<double> average(6, 0.0);
        for (const auto& imu : imu_buffer) {
            average[0] += imu.gyro_.x();
            average[1] += imu.gyro_.y();
            average[2] += imu.gyro_.z();
            average[3] += imu.accel_.x();
            average[4] += imu.accel_.y();
            average[5] += imu.accel_.z();
        }
        for (auto& avg : average) {
            avg *= size_invert;
        }

        // 计算标准差
        double sum[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (const auto& imu : imu_buffer) {
            sum[0] += (imu.gyro_.x() - average[0]) * (imu.gyro_.x() - average[0]);
            sum[1] += (imu.gyro_.y() - average[1]) * (imu.gyro_.y() - average[1]);
            sum[2] += (imu.gyro_.z() - average[2]) * (imu.gyro_.z() - average[2]);
            sum[3] += (imu.accel_.x() - average[3]) * (imu.accel_.x() - average[3]);
            sum[4] += (imu.accel_.y() - average[4]) * (imu.accel_.y() - average[4]);
            sum[5] += (imu.accel_.z() - average[5]) * (imu.accel_.z() - average[5]);
        }

        double std[6];
        for (int i = 0; i < 6; ++i) {
            std[i] = std::sqrt(sum[i] * size_invert); // 直接计算速率标准差（rad/s, m/s²）
        }

        // 检查时间跨度
        double dt = imu_buffer.back().timestamp_ - imu_buffer.front().timestamp_;
        if (dt < 0.1) {
            // LOG(WARNING) << "IMU buffer time span too short: " << dt << " s";
            return false;
        }

        // 零速判断
        bool is_zero_velocity = (std[0] < ZERO_VELOCITY_GYR_THRESHOLD) &&
                                (std[1] < ZERO_VELOCITY_GYR_THRESHOLD) &&
                                (std[2] < ZERO_VELOCITY_GYR_THRESHOLD) &&
                                (std[3] < ZERO_VELOCITY_ACC_THRESHOLD) &&
                                (std[4] < ZERO_VELOCITY_ACC_THRESHOLD) &&
                                (std[5] < ZERO_VELOCITY_ACC_THRESHOLD);

        // if (options_.verbose_) {
        //     LOG(INFO) << "Zero velocity detection: " << (is_zero_velocity ? "True" : "False")
        //             << ", std: [" << std[0] << ", " << std[1] << ", " << std[2] << ", "
        //             << std[3] << ", " << std[4] << ", " << std[5] << "]";
        // }

        return is_zero_velocity;
    }

    static void save_vec3(std::ofstream& fout, const Vec3d& v){ 
        fout << v[0] << " " << v[1] << " " << v[2] << " "; 
    };

    static void save_quat(std::ofstream& fout, const Quatd& q) {
        fout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " ";
    };

    static void save_result(std::ofstream& fout, const NavStated& save_state) {
        fout << std::setprecision(18) << save_state.timestamp_ << " " << std::setprecision(9);
        save_vec3(fout, save_state.p_);
        save_quat(fout, save_state.R_.unit_quaternion());
        save_vec3(fout, save_state.v_);
        save_vec3(fout, save_state.bg_);
        save_vec3(fout, save_state.ba_);
        fout << std::endl;
    };
};

