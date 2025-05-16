#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>

#include "sophus/se3.hpp"
#include "image.h"
#include "feature.h"

struct MapPoint;
struct Feature;
struct Frame {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    unsigned long id_ = 0;           // id of this frame
    unsigned long keyframe_id_ = 0;  // id of key frame
    bool is_keyframe_ = false;       // 是否为关键帧
    double timestamp_;              // 时间戳，暂不使用
    SE3 pose_;                       // Tcw 形式Pose
    cv::Mat left_img_, right_img_;   // stereo images

    // extracted features in left image
    std::vector<std::shared_ptr<Feature>> features_left_;
    // corresponding features in right image, set to nullptr if no corresponding
    std::vector<std::shared_ptr<Feature>> features_right_;

   public:  // data members
    Frame() {}

    Frame(long id, const Image& image);

    void SetKeyFrame();

    static std::shared_ptr<Frame> CreateFrame(const Image& image);
};
using FramePtr = std::shared_ptr<Frame>;

