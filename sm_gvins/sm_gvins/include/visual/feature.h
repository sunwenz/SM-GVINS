#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include "eigen_types.h"

struct Frame;
struct MapPoint;
struct Feature {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    std::weak_ptr<Frame> frame_;         
    cv::KeyPoint pixel_pt_;      
    Vec3d normlize_pt_;
    std::weak_ptr<MapPoint> map_point_;  

    bool is_outlier_ = false;       
    bool is_on_left_image_ = true;

   public:
    Feature() = default;

    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
        : frame_(frame), pixel_pt_(kp) {}
};
using FeaturePtr = std::shared_ptr<Feature>;