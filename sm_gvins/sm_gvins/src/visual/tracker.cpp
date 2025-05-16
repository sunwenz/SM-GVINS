#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>
#include <unordered_map>

#include "tracker.h"
#include "timer.h"
#include "math.h"

Tracker::Tracker(const Options& options)
    : options_(std::move(options))
{
    camera_left_  = Camera::createCamera(options_.K_, options_.D_, cv::Size(options_.image_width_, options_.image_width_));
    camera_right_ = Camera::createCamera(options_.K_, options_.D_, cv::Size(options_.image_width_, options_.image_width_));

    map_ = std::make_shared<Map>();
}


bool Tracker::TrackFrame(const Image& image){    
    
}

bool Tracker::Initilize(const Image& image){
    curr_frame_ = Frame::CreateFrame(image);

    int num_features_left = DetectFeatures();
    int num_coor_features = FindFeaturesInRight();
    if (num_coor_features < MIN_FEATURES_INIT) {
        return false;
    }

    bool build_map_success = BuildInitMap();
    if (build_map_success) {
        last_frame_ = curr_frame_;
        return true;
    }
    return false;
}

int Tracker::DetectFeatures(){
    cv::Mat mask(curr_frame_->left_img_.size(), CV_8UC1, cv::Scalar(255));
    for(auto& feature : curr_frame_->features_left_){
        cv::rectangle(mask, feature->pixel_pt_.pt - cv::Point2f(10, 10), feature->pixel_pt_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
    }

    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(curr_frame_->left_img_, points, MAX_CNT - curr_frame_->features_left_.size(), 0.01, MIN_DIST, mask);
    
    int cnt_detected = 0;
    std::vector<cv::KeyPoint> keypoints;
    cv::KeyPoint::convert(points, keypoints);
    for (auto &kp : keypoints) {
        curr_frame_->features_left_.push_back(
            FeaturePtr(new Feature(curr_frame_, kp)));
        cnt_detected++;
    }

    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

int Tracker::FindFeaturesInRight(){
    std::vector<cv::Point2f> kps_left, kps_right;
    for (auto &kp : curr_frame_->features_left_) {
        kps_left.push_back(kp->pixel_pt_.pt);
        auto mp = kp->map_point_.lock();
        if (mp) {
            // use projected points as initial guess
            auto px =
                camera_right_->world2pixel(mp->pos_, curr_frame_->pose_);
            kps_right.push_back(cv::Point2f(px[0], px[1]));
        } else {
            // use same pixel in left iamge
            kps_right.push_back(kp->pixel_pt_.pt);
        }
    }

    std::vector<uchar> status;
    cv::Mat error;
    cv::calcOpticalFlowPyrLK(
        curr_frame_->left_img_, curr_frame_->right_img_, kps_left,
        kps_right, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_right[i], 7);
            FeaturePtr feat(new Feature(curr_frame_, kp));
            feat->is_on_left_image_ = false;
            curr_frame_->features_right_.push_back(feat);
            num_good_pts++;
        } else {
            curr_frame_->features_right_.push_back(nullptr);
        }
    }
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    return num_good_pts;
}

bool Tracker::BuildInitMap(){
    std::vector<SE3> poses{curr_frame_->pose_, curr_frame_->pose_ * options_.Tc0c1_.inverse()};
    size_t cnt_init_landmarks = 0;
    for (size_t i = 0; i < curr_frame_->features_left_.size(); ++i) {
        if (curr_frame_->features_right_[i] == nullptr) continue;
        // create map point from triangulation
        std::vector<Vec3d> points{
            camera_left_->pixel2camera(
                Vec2d(curr_frame_->features_left_[i]->pixel_pt_.pt.x,
                     curr_frame_->features_left_[i]->pixel_pt_.pt.y)),
            camera_right_->pixel2camera(
                Vec2d(curr_frame_->features_right_[i]->pixel_pt_.pt.x,
                     curr_frame_->features_right_[i]->pixel_pt_.pt.y))};
        Vec3d pworld = Vec3d::Zero();

        if (math::triangulatePoint(poses, points, pworld) && pworld[2] > 0) {
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->pos_ = pworld;
            new_map_point->AddObservation(curr_frame_->features_left_[i]);
            new_map_point->AddObservation(curr_frame_->features_right_[i]);
            curr_frame_->features_left_[i]->map_point_ = new_map_point;
            curr_frame_->features_right_[i]->map_point_ = new_map_point;
            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }
    curr_frame_->SetKeyFrame();
    map_->InsertKeyFrame(curr_frame_);

    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}