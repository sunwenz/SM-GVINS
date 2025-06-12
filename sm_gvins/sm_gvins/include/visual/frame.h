#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "eigen_types.h"
#include "image.h"
#include "feature.h"
#include "orb_extractor.h"

class Frame : public std::enable_shared_from_this<Frame>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Frame();
    Frame(int id, double timestamp, cv::Mat left_img, cv::Mat right_img = cv::Mat());
    ~Frame() = default;

    static std::shared_ptr<Frame> createFrame(double stamp, cv::Mat left_img, cv::Mat right_img); // 创建一帧

    void ExtractKeyPointsAndDescriptors(); // 提取关键点

    void UndistKeyPoints(); // 根据相机畸变参数校正关键点坐标

    void MatchFromeStereo(); // 双目相机左右目特征点匹配

    void CreateFeatures(); // 生成当前帧的特征点

    void SetKeyframe();
    
    ulong id_;                // 帧ID
    ulong keyframe_id_;
    double timestamp_;     // 时间戳

    SE3 Twc_;   // 前端跟踪得到的位姿

    cv::Mat left_img_, right_img_; // 左图,右图
    cv::Mat mask_;         // 目标掩码（背景为0）
    cv::Mat scimg_;        // 缩放图

    std::vector<FeaturePtr> features_;        // 当前帧特征点
    ORBextractor::Ptr orbleft_, orbright_;    // 左右目ORB特征
    
    bool is_good_ = false;    // 估计的位姿是否合理
    bool is_keyframe_ = false;     // 是否是关键帧的标志

    std::vector<cv::KeyPoint> keypoints_l_; // 左目中的所有关键点（去除畸变后）
    std::vector<cv::KeyPoint> keypoints_r_; // 右目中的所有关键点
    cv::Mat descriptors_l_;                 // 左目所有关键点的描述子
    cv::Mat descriptors_r_;                 // 右目所有关键点的描述子
    std::vector<float> keypts_depth_;       // RGBD相机中每个特征点对应的深度
    std::vector<float> left_to_right_;      // 左目特征点对应的右目特征点
};
using FramePtr = std::shared_ptr<Frame>;

