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
        cv::Mat K0_, K1_;
        cv::Mat D0_, D1_;
        SE3 Tc0c1_;
    };
    
    Tracker(MapPtr map, const Options& options = Options());

    void SetCameras(Camera::Ptr camera_left, Camera::Ptr camera_right);

    bool TrackFrame(const Image& image);

    bool Initilize(const Image& image);
private:
    bool IsKeyframe();
    
    int TrackLastFrame();

    bool EstimatorCurrentPose();

    void SetObservationsForKeyFrame();

    int DetectFeatures();

    int FindFeaturesInRight();

    bool BuildInitMap();

    void TriangulateNewPoints();

    FramePtr curr_frame_      = nullptr;
    FramePtr last_frame_      = nullptr;
    Camera::Ptr camera_left_  = nullptr;
    Camera::Ptr camera_right_ = nullptr;

    SE3 relative_motion_;  // 当前帧与上一帧的相对运动，用于估计当前帧pose初值
    
    MapPtr map_ = nullptr;

    Options options_;
};
using TrackerPtr = std::shared_ptr<Tracker>;



