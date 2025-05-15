#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <Eigen/Core>

#include "sophus/se3.hpp"

#include "keyframe.h"

using Eigen::Vector3d;
using std::string;
using std::vector;

class DrawerRviz{

public:
    explicit DrawerRviz(ros::NodeHandle &nh);

    void run();

    void setFinished();

    // 地图
    void addNewFixedMappoint(Vector3d point);

    void updateMap(const Eigen::Matrix4d &pose);

    // 跟踪图像
    void updateFrame(FramePtr frame);
    // void updateTrackedMapPoints(vector<cv::Point2f> map, vector<cv::Point2f> matched,
    //                             vector<MapPointType> mappoint_type);
    void updateTrackedRefPoints(vector<cv::Point2f> ref, vector<cv::Point2f> cur);

private:
    void publishTrackingImage();

    void publishMapPoints();

    void publishOdometry();

private:
    // 多线程
    std::condition_variable update_sem_;
    std::mutex update_mutex_;
    std::mutex map_mutex_;
    std::mutex image_mutex_;

    // 标志
    std::atomic<bool> isfinished_;
    std::atomic<bool> isframerdy_;
    std::atomic<bool> ismaprdy_;

    // 跟踪
    cv::Mat raw_image_;
    vector<Vector3d> fixed_mappoints_;

    Sophus::SE3d pose_;
    nav_msgs::Path path_;

    ros::Publisher path_pub_;
    ros::Publisher pose_pub_;
    ros::Publisher track_image_pub_;
    ros::Publisher fixed_points_pub_;
    ros::Publisher current_points_pub_;

    string frame_id_;
};

