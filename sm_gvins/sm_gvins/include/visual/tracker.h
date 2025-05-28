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
#include "parameters.h"
#include "orb_extractor.h"

class Tracker{ 
public:
    static constexpr int MAX_LOST = 10;
    
    Tracker(MapPtr map);

    void SetCameras(Camera::Ptr camera_left, Camera::Ptr camera_right);

    bool TrackFrame(FramePtr frame);

private:
    void ProcessCurrentFrame();
    bool BuildInitMap();
    bool TriangulateNewPoints();
    bool MatchWithLastframe();
    bool MatchWithReferenceframe();
    bool MatchFeatures(FramePtr frame, int th);
    bool MatchFeaturesByProjection(FramePtr frame, int th);
    bool MatchFeaturesByBruteForce(FramePtr frame, int th);
    void CheckRotConsistency(FramePtr frame, vector<cv::Point2f> &fea_mat, vector<int> &index);
    bool CalcPoseByPnP(const vector<cv::Point3d>& points_3d, const vector<cv::Point2d>& pixels_2d);

    bool InBorder(const float ptx, const float pty, int border_size = 1)
    {
        return (Parameters::min_X_ + border_size) < ptx && ptx < (Parameters::max_X_ - border_size) && (Parameters::min_Y_ + border_size) < pty && pty < (Parameters::max_Y_ - border_size);
    }

    FramePtr ref_frame_       = nullptr;
    FramePtr curr_frame_      = nullptr;
    FramePtr last_frame_      = nullptr;
    Camera::Ptr camera_left_  = nullptr;
    Camera::Ptr camera_right_ = nullptr;

    std::vector<cv::Point2f> features_matched_;

    SE3 relative_motion_;  // 当前帧与上一帧的相对运动，用于估计当前帧pose初值
    
    MapPtr map_ = nullptr;

    int num_lost_ = 0;
    
    bool initilize_flag_ = true;

    float normalpose_max_t_ = 0.5; // 与匀速模型的最大位移差
    float normalpose_max_R_ = 0.2; // 与匀速模型的最大旋转差
};
using TrackerPtr = std::shared_ptr<Tracker>;