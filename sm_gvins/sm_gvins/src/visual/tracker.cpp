#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>
#include <unordered_map>

#include "tracker.h"
#include "timer.h"
#include "math.h"

#include "g2o_types.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

Tracker::Tracker(MapPtr map, const Options& options)
    : map_(std::move(map)), options_(std::move(options))
{
    gftt_ =
        cv::GFTTDetector::create(150, 0.01, 20);
}

void Tracker::SetCameras(Camera::Ptr camera_left, Camera::Ptr camera_right){
    camera_left_  = std::move(camera_left);
    camera_right_ = std::move(camera_right);
}

bool Tracker::TrackFrame(const Image& image){    
    curr_frame_ = Frame::CreateFrame(image);
    if(last_frame_){
        curr_frame_->pose_ = relative_motion_ * last_frame_->pose_;
    }

    int num_track_last = TrackLastFrame();
    int tracking_inliers = OptimizeInitPose();
    if(tracking_inliers <= 20){
        return false;
    }
    // if(!EstimatorCurrentPose()){
    //     return false;
    // }

    // bool is_keyframe = IsKeyframe();
    // if(is_keyframe == false){
    //     return false;
    // }
    
    InsertKeyframe();

    relative_motion_ = curr_frame_->pose_ * last_frame_->pose_.inverse();
    last_frame_ = curr_frame_;

    return true;
}

void Tracker::InsertKeyframe(){
    curr_frame_->SetKeyFrame();

    map_->InsertKeyFrame(curr_frame_);

    LOG(INFO) << "Set frame " << curr_frame_->id_ << " as keyframe "
              << curr_frame_->keyframe_id_;

    SetObservationsForKeyFrame();

    DetectFeatures();
    
    FindFeaturesInRight();
    
    TriangulateNewPoints();
}

void Tracker::SetObservationsForKeyFrame() {
    for (auto &feat : curr_frame_->features_left_) {
        auto mp = feat->map_point_.lock();
        if (mp) mp->AddObservation(feat);
    }
}

bool Tracker::IsKeyframe(){
    if(curr_frame_->features_left_.size() < MIN_FEATURES){
        LOG(INFO) << "be selected of the keyframe because the features not enough";
        return true;
    }

    if (last_frame_) {
        double total_parallax = 0.0;
        int valid_pairs = 0;

        for (auto& feat_curr : curr_frame_->features_left_) {
            if (!feat_curr || feat_curr->map_point_.expired()) continue;

            auto mp = feat_curr->map_point_.lock();

            // 在上一帧中找是否有这个 MapPoint 的观测
            for (auto& feat_last : last_frame_->features_left_) {
                if (!feat_last || feat_last->map_point_.expired()) continue;
                if (feat_last->map_point_.lock() == mp) {
                    // 计算两帧的像素视差（欧氏距离）
                    double dx = feat_curr->pixel_pt_.pt.x - feat_last->pixel_pt_.pt.x;
                    double dy = feat_curr->pixel_pt_.pt.y - feat_last->pixel_pt_.pt.y;
                    double parallax = std::sqrt(dx * dx + dy * dy);
                    total_parallax += parallax;
                    valid_pairs++;
                    break; // 匹配一次就够了
                }
            }
        }

        double avg_parallax = (valid_pairs > 0) ? total_parallax / valid_pairs : 0.0;
        if (avg_parallax > PARALLAX_THRESHOLD) {
            LOG(INFO) << "be selected of the keyframe because avg_parallax (" 
                    << avg_parallax << " > " << PARALLAX_THRESHOLD << ")";
            return true;
        }
    }

    if (last_frame_->timestamp_ >= 0 && (curr_frame_->timestamp_ - last_frame_->timestamp_) > MIN_TIME_GAP) {
        LOG(INFO) << "be selected of the keyframe because (" << (curr_frame_->timestamp_ - last_frame_->timestamp_) << " > " << MIN_TIME_GAP << ")";
        return true;
    }
    return false;
}

int Tracker::TrackLastFrame(){
    std::vector<cv::Point2f> kps_last, kps_curr;
    for(auto& kp : last_frame_->features_left_){
        if(kp->map_point_.lock()){
            auto mp = kp->map_point_.lock();
            auto px = 
                camera_left_->world2pixel(mp->pos_, curr_frame_->pose_);
            kps_last.push_back(kp->pixel_pt_.pt);
            kps_curr.push_back(cv::Point2f(px[0], px[1]));
        }else{
            kps_last.push_back(kp->pixel_pt_.pt);
            kps_curr.push_back(kp->pixel_pt_.pt);
        }
    }

    std::vector<uchar> status;
    cv::Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame_->left_img_, curr_frame_->left_img_, 
        kps_last, kps_curr, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    
    int good_num_pts = 0;
    for(size_t i = 0; i < status.size(); ++i){
        if(status[i]){
            cv::KeyPoint kp(kps_curr[i], 7);
            FeaturePtr feature(new Feature(curr_frame_, kp));
            feature->map_point_ = last_frame_->features_left_[i]->map_point_;
            curr_frame_->features_left_.push_back(feature);
            good_num_pts++;
        }
    }

    LOG(INFO) << "find " << good_num_pts << " in the last image";
    return good_num_pts;
}

int Tracker::OptimizeInitPose(){
    // setup g2o
    using BlockSolverType = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
    
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(curr_frame_->pose_);
    optimizer.addVertex(vertex_pose);

    // K
    Mat3d K = camera_left_->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<FeaturePtr> features;
    for (size_t i = 0; i < curr_frame_->features_left_.size(); ++i) {
        auto mp = curr_frame_->features_left_[i]->map_point_.lock();
        if (mp) {
            features.push_back(curr_frame_->features_left_[i]);
            EdgeProjectionPoseOnly *edge =
                new EdgeProjectionPoseOnly(mp->pos_, K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(
                Vec2d(curr_frame_->features_left_[i]->pixel_pt_.pt.x,
                     curr_frame_->features_left_[i]->pixel_pt_.pt.y));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }

    // estimate the Pose the determine the outliers
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        vertex_pose->setEstimate(curr_frame_->pose_);
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        // count the outliers
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            if (features[i]->is_outlier_) {
                e->computeError();
            }
            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                e->setLevel(1);
                cnt_outlier++;
            } else {
                features[i]->is_outlier_ = false;
                e->setLevel(0);
            };

            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;
    // Set pose and outlier
    curr_frame_->pose_ = vertex_pose->estimate();

    LOG(INFO) << "Current Pose = \n" << curr_frame_->pose_.matrix();

    for (auto &feat : features) {
        if (feat->is_outlier_) {
            feat->map_point_.reset();
            feat->is_outlier_ = false;  // maybe we can still use it in future
        }
    }
    return features.size() - cnt_outlier;
}

bool Tracker::EstimatorCurrentPose() {
    std::vector<cv::Point3f> obj_points;
    std::vector<cv::Point2f> img_points;

    for (auto& feat : curr_frame_->features_left_) {
        if (feat && !feat->map_point_.expired()) {
            auto mp = feat->map_point_.lock();
            obj_points.emplace_back(mp->pos_[0], mp->pos_[1], mp->pos_[2]);
            img_points.push_back(feat->pixel_pt_.pt);
        }
    }

    if (obj_points.size() < 4) {
        LOG(WARNING) << "Not enough points for PnP estimation: " << obj_points.size();
        return false;
    }

    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    bool pnp_success = cv::solvePnPRansac(obj_points, img_points, camera_left_->cvK(), camera_left_->cvD(),
                                          rvec, tvec, true, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_EPNP);
    if (!pnp_success || inliers.size() < 4) {
        LOG(WARNING) << "PnPRansac failed!";
        return false;
    }

    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    for (int i = 0; i < 3; ++i) {
        t(i) = tvec.at<double>(i);
        for (int j = 0; j < 3; ++j)
            R(i, j) = R_cv.at<double>(i, j);
    }
    curr_frame_->pose_ = SE3(R, t);

    // reprojection error
    double total_error = 0.0;
    for (int i : inliers) {
        std::vector<cv::Point2f> projected;
        cv::projectPoints(std::vector<cv::Point3f>{obj_points[i]}, rvec, tvec, camera_left_->cvK(), camera_left_->cvD(), projected);
        double dx = projected[0].x - img_points[i].x;
        double dy = projected[0].y - img_points[i].y;
        total_error += std::sqrt(dx * dx + dy * dy);
    }

    LOG(INFO) << "PnP inliers: " << inliers.size() << "/" << obj_points.size();
    LOG(INFO) << "Average reprojection error: " << (total_error / inliers.size()) << " pixels";
    return true;
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
    cv::Mat mask(curr_frame_->left_img_.size(), CV_8UC1, 255);
    for (auto &feat : curr_frame_->features_left_) {
        cv::rectangle(mask, feat->pixel_pt_.pt - cv::Point2f(10, 10),
                      feat->pixel_pt_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
    }

    std::vector<cv::KeyPoint> keypoints;
    gftt_->detect(curr_frame_->left_img_, keypoints, mask);
    int cnt_detected = 0;
    for (auto &kp : keypoints) {
        curr_frame_->features_left_.push_back(
            FeaturePtr(new Feature(curr_frame_, kp)));
        cnt_detected++;
    }

    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
/* 
    cv::Mat mask(curr_frame_->left_img_.size(), CV_8UC1, cv::Scalar(255));
    for(auto& feature : curr_frame_->features_left_){
        cv::rectangle(mask, feature->pixel_pt_.pt - cv::Point2f(10, 10), feature->pixel_pt_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
    }

    int cnt_detected = 0;
    std::vector<cv::Point2f> points;
    int n_max_cnt = MAX_CNT - static_cast<int>(curr_frame_->features_left_.size());
    if(n_max_cnt > 0){
        cv::goodFeaturesToTrack(curr_frame_->left_img_, points, MAX_CNT - curr_frame_->features_left_.size(), 0.01, MIN_DIST, mask);
        std::vector<cv::KeyPoint> keypoints;
        cv::KeyPoint::convert(points, keypoints);
        for (auto &kp : keypoints) {
            curr_frame_->features_left_.push_back(
                FeaturePtr(new Feature(curr_frame_, kp)));
            cnt_detected++;
        }

        LOG(INFO) << "Detect " << cnt_detected << " new features";
    }
    
    return cnt_detected;
     */
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
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
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

void Tracker::TriangulateNewPoints(){
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    SE3 current_pose_Twc = curr_frame_->pose_.inverse();
    int cnt_triangulated_pts = 0;
    for (size_t i = 0; i < curr_frame_->features_left_.size(); ++i) {
        if (curr_frame_->features_left_[i]->map_point_.expired() &&
            curr_frame_->features_right_[i] != nullptr) {
            // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
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
                pworld = current_pose_Twc * pworld;
                new_map_point->pos_ = pworld;
                new_map_point->AddObservation(
                    curr_frame_->features_left_[i]);
                new_map_point->AddObservation(
                    curr_frame_->features_right_[i]);

                curr_frame_->features_left_[i]->map_point_ = new_map_point;
                curr_frame_->features_right_[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
}