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
#include "keyframe.h"
#include "map.h"

class Tracker{ 
public:
    static constexpr int MAX_CNT = 200.0;
    static constexpr int MIN_DIST = 30.0;
    static constexpr int MIN_FEATURES = 50; // 最小特征点数量
    static constexpr double PARALLAX_THRESHOLD = 10.0; // 平均视差阈值（像素）
    static constexpr double MIN_TIME_GAP = 0.5; // 关键帧最小时间间隔（秒）

    Tracker() = default;
    Tracker(const std::string& config_file);
    
    bool TrackFrame(const Image& image);

    cv::Mat img_tracking() const {
        return img_track_;
    }

    bool StereoInit(const Image& image);

    void SetCameras(const std::vector<Camera::Ptr>& cameras){
        cameras_ = std::move(cameras);
    }

    void SetMap(const MapPtr& map){
        map_ = map;
    }
private:
    void SetMask();
    void DetectNewFeatures();
    FramePtr CreateNewKeyFrame(double stamp);
    void EstimateCurrentPose(FramePtr current_frame_);
    void EstimateStateInWindow();
    void TrackByLK();
    void TrackByStereoLK(const cv::Mat &img_right);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    bool inBorder(const cv::Point2f &pt);
    template <typename T> void reduceVector(T &vec, const std::vector<uchar>& status);
    bool isKeyframe();
    std::vector<cv::Point2f> PixelToCamera(std::vector<cv::Point2f> &pts, Camera::Ptr cam);
    std::vector<cv::Point2f> ptsVelocity(
        std::vector<unsigned long long> &ids, std::vector<cv::Point2f> &pts, 
        std::map<int, cv::Point2f> &cur_id_pts, std::map<int, cv::Point2f> &prev_id_pts);
    void DrawTrack(
        const cv::Mat &imLeft, const cv::Mat &imRight, 
        std::vector<unsigned long long> &curLeftIds,
        std::vector<cv::Point2f> &curLeftPts, 
        std::vector<cv::Point2f> &curRightPts,
        std::map<int, cv::Point2f> &prevLeftPtsMap);
private:
    double curr_time_, prev_time_;
    int row_, col_;
    cv::Mat img_track_;
    cv::Mat mask_;
    cv::Mat prev_img_, curr_img_; 
    std::vector<cv::Point2f> n_pts_;
    std::vector<cv::Point2f> prev_pts_, curr_pts_, curr_right_pts_;
    std::vector<cv::Point2f> prev_un_pts_, curr_un_pts_, curr_un_right_pts_;
    std::vector<cv::Point2f> pts_velocity_, right_pts_velocity_;
    std::vector<unsigned long long> ids_, ids_right_;
    std::vector<int> track_cnt_;
    std::map<int, cv::Point2f> curr_un_pts_map_, prev_un_pts_map_; 
    std::map<int, cv::Point2f> curr_un_right_pts_map_, prev_un_right_pts_map_;
    std::map<int, cv::Point2f> prev_left_pts_map_;

    double last_keyframe_time_ = 0.0;
    unsigned long long n_id = 0;
    
    MapPtr map_ = nullptr;
    std::vector<Camera::Ptr> cameras_;
};
using TrackerPtr = std::shared_ptr<Tracker>;



