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
#include <cv_bridge/cv_bridge.h>

#include "estimator.h"
#include "drawer_rviz.h"

class SM_GVINS{
   public:
    struct Options{
        std::string image0_topic_;
        std::string image1_topic_;
        std::string output_path_;

        Estimator::Options estimator_options_;
    };
    
    SM_GVINS(ros::NodeHandle& nh, const Options& options = Options());

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

   private:
    void sync_process();

    cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

    void save_vec3(std::ofstream& fout, const Vec3d& v){ 
        fout << v[0] << " " << v[1] << " " << v[2] << " "; 
    };

    void save_quat(std::ofstream& fout, const Quatd& q) {
        fout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " ";
    };

    void save_result(std::ofstream& fout, const NavStated& save_state) {
        fout << std::setprecision(18) << save_state.timestamp_ << " " << std::setprecision(9);
        save_vec3(fout, save_state.p_);
        save_quat(fout, save_state.R_.unit_quaternion());
        save_vec3(fout, save_state.v_);
        save_vec3(fout, save_state.bg_);
        save_vec3(fout, save_state.ba_);
        fout << std::endl;
    };

    Options options_;

    ros::Subscriber image0_sub_;
    ros::Subscriber image1_sub_;

    std::mutex m_buf_;
    std::queue<sensor_msgs::ImageConstPtr> img0_buf_;
    std::queue<sensor_msgs::ImageConstPtr> img1_buf_;

    std::atomic<bool> running_{true};
    std::thread sync_thread_;

    
    Estimator estimator_;
    DrawerRviz drawer_;

    std::ofstream f_out_;
};