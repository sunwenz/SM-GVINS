#pragma once
#include <iostream>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>

#include "types/image.h"
#include "types/imu.h"
#include "types/nav_state.h"
#include "visual/tracker.h"
#include "imu_preinteg.h"

class Estimator{
   public:
    Estimator();
    
    void AddImage(const Image& image);

    void AddIMU(const IMU& imu);

    void SetCameras(Camera::Ptr camera_left, Camera::Ptr camera_right);

   private:
    void Process();
    void ProcessImage(FramePtr frame);
    void Optimize();

    std::shared_ptr<IMUPreintegration> imu_preinteg_ = nullptr;
    std::deque<NavStatedPtr> state_window_;
    std::deque<FramePtr> frame_window_;
    Mat3d Rbc_[2];
    Vec3d tbc_[2];

    // std::mutex buf_mtx_;
    std::deque<IMU> imu_window_;
    
    std::atomic_bool process_running_{true};
    std::mutex buf_mtx_;
    std::condition_variable con_;
    std::thread process_;

    bool initlized_flag_ = false;
    bool first_imu_flag_ = true;

    // 数据队列
    std::queue<FramePtr> frame_buf_;
    std::queue<IMU> imu_buf_;
    std::queue<IMUPreintegrationPtr> imu_preinteg_buf_;

    TrackerPtr tracker_ = nullptr;
    MapPtr map_ = nullptr;
    Camera::Ptr camera_left_ = nullptr, camera_right_ = nullptr;

    std::ofstream f_out_;
};