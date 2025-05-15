#include "drawer_rviz.h"
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>

DrawerRviz::DrawerRviz(ros::NodeHandle &nh){
    frame_id_ = "world";

    pose_pub_           = nh.advertise<nav_msgs::Odometry>("pose", 2);
    path_pub_           = nh.advertise<nav_msgs::Path>("path", 2);
    track_image_pub_    = nh.advertise<sensor_msgs::Image>("tracking", 2);
    fixed_points_pub_   = nh.advertise<sensor_msgs::PointCloud>("fixed", 2);
    current_points_pub_ = nh.advertise<sensor_msgs::PointCloud>("current", 2);

    drawer_ = std::thread(&DrawerRviz::run, this);
}

void DrawerRviz::setFinished() {
    isfinished_ = true;
    update_sem_.notify_one();
}

void DrawerRviz::run() {

    while (!isfinished_) {
        // 等待绘图更新信号
        std::unique_lock<std::mutex> lock(update_mutex_);
        update_sem_.wait(lock);

        // 发布跟踪的图像
        if (isframerdy_) {
            publishTrackingImage();

            isframerdy_ = false;
        }

        // // 发布轨迹和地图点
        // if (ismaprdy_) {
        //     publishOdometry();

        //     publishMapPoints();

        //     ismaprdy_ = false;
        // }
    }
}

void DrawerRviz::updateFrame(const Image& image) {
    std::unique_lock<std::mutex> lock(image_mutex_);

    image.img_.copyTo(raw_image_);

    isframerdy_ = true;
    update_sem_.notify_one();
}

void DrawerRviz::publishTrackingImage() {
    std::unique_lock<std::mutex> lock(image_mutex_);

    if (raw_image_.empty()) {
        ROS_WARN("raw_image_ is empty, skipping publish.");
        return;
    }

    cv_bridge::CvImage out_msg;
    out_msg.header.stamp = ros::Time::now();
    out_msg.header.frame_id = frame_id_;
    out_msg.encoding = sensor_msgs::image_encodings::MONO8;
    out_msg.image = raw_image_;  // or your processed image

    track_image_pub_.publish(out_msg.toImageMsg());
}
