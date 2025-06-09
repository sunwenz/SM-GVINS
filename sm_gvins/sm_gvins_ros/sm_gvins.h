#pragma once

#include <vector>
#include <queue>
#include <string>
#include <iostream>
#include <mutex>
#include <thread>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>

#include "estimator.h"
#include "drawer_rviz.h"

class SM_GVINS{
   public:
    SM_GVINS(ros::NodeHandle& nh);

    ~SM_GVINS();

    void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
    {
        std::lock_guard<std::mutex> lock(m_buf_);
        img0_buf_.push(img_msg);
    }

    void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
    {
        std::lock_guard<std::mutex> lock(m_buf_);
        img1_buf_.push(img_msg);
    }

    void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
    {
        double t = imu_msg->header.stamp.toSec();
        double dx = imu_msg->linear_acceleration.x;
        double dy = imu_msg->linear_acceleration.y;
        double dz = imu_msg->linear_acceleration.z;
        double rx = imu_msg->angular_velocity.x;
        double ry = imu_msg->angular_velocity.y;
        double rz = imu_msg->angular_velocity.z;

        estimator_.AddIMU(IMU(t, Vec3d(rx, ry, rz), Vec3d(dx, dy, dz)));
    }

   private:
    void sync_process();

    cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

    ros::Subscriber image0_sub_;
    ros::Subscriber image1_sub_;

    std::mutex m_buf_;
    std::queue<sensor_msgs::ImageConstPtr> img0_buf_;
    std::queue<sensor_msgs::ImageConstPtr> img1_buf_;

    std::atomic<bool> running_{true};
    std::thread sync_thread_;

    Estimator estimator_;
    DrawerRviz drawer_;
};