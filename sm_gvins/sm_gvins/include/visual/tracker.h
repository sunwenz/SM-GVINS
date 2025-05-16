#pragma once

#include <iostream>
#include <memory>
#include <map>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "camera.h"
#include "image.h"
#include "frame.h"
#include "map.h"
#include "mappoint.h"

class Tracker{ 
public:
    static constexpr int MAX_CNT               = 200;
    static constexpr int MIN_DIST              = 30;
    static constexpr int MIN_FEATURES          = 50; // 最小特征点数量
    static constexpr int MIN_FEATURES_INIT     = 100; // 最小特征点数量
    static constexpr double PARALLAX_THRESHOLD = 10.0; // 平均视差阈值（像素）
    static constexpr double MIN_TIME_GAP       = 0.5; // 关键帧最小时间间隔（秒）

    struct Options{
        int image_width_;
        int image_height_;
        cv::Mat K_;  // Camera intrinsic matrix
        cv::Mat D_;  // Distortion coeffs
        SE3 Tc0c1_;
    };
    
    Tracker(const Options& options = Options());

    bool TrackFrame(const Image& image);

    bool Initilize(const Image& image);
private:
    int TrackLastFrame();

    int EstimatorCurrentPose();

    int DetectFeatures();

    int FindFeaturesInRight();

    bool BuildInitMap();

    FramePtr curr_frame_      = nullptr;
    FramePtr last_frame_      = nullptr;
    Camera::Ptr camera_left_  = nullptr;
    Camera::Ptr camera_right_ = nullptr;

    MapPtr map_ = nullptr;

    Options options_;
};
using TrackerPtr = std::shared_ptr<Tracker>;



