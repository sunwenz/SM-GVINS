#include "estimator.h"
#include "g2o_types.h"
#include <glog/logging.h>
#include <opencv2/core/eigen.hpp>  
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

Estimator::Estimator(const Options& options)
    : options_(std::move(options))
{
    camera_left_  = std::make_shared<Camera>(options_.tracker_options_.K_, options_.tracker_options_.D_, cv::Size(options_.tracker_options_.image_width_, options_.tracker_options_.image_height_));
    camera_right_ = std::make_shared<Camera>(options_.tracker_options_.K_, options_.tracker_options_.D_, cv::Size(options_.tracker_options_.image_width_, options_.tracker_options_.image_height_));
    camera_right_->pose_ = options_.tracker_options_.Tc0c1_;
    camera_right_->pose_inv_ = options_.tracker_options_.Tc0c1_.inverse();

    map_ = std::make_shared<Map>();
    tracker_ = std::make_shared<Tracker>(map_, options_.tracker_options_);

    tracker_->SetCameras(camera_left_, camera_right_);
}

void Estimator::AddImage(const Image& image){
    Image un_image;
    un_image.timestamp_ = image.timestamp_;
    camera_left_->undistortImage(image.img_, un_image.img_);
    camera_right_->undistortImage(image.img_right_, un_image.img_right_);

    if(is_first_image_){
        if(tracker_->Initilize(un_image)){
            is_first_image_ = false;
        }
        return;
    }

    bool is_keyframe = tracker_->TrackFrame(un_image);
    if(is_keyframe){
        Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
        Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
        Optimize(active_kfs, active_landmarks);
    }else{
        LOG(INFO) << "the image in stamp: " << image.timestamp_ << " is not keyframe";
    }
}

void Estimator::Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks){
    // setup g2o
    using BlockSolverType = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
    
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // pose 顶点，使用Keyframe id
    std::map<unsigned long, VertexPose *> vertices;
    unsigned long max_kf_id = 0;
    for (auto &keyframe : keyframes) {
        auto kf = keyframe.second;
        VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
        vertex_pose->setId(kf->keyframe_id_);
        vertex_pose->setEstimate(kf->pose_);
        optimizer.addVertex(vertex_pose);
        if (kf->keyframe_id_ > max_kf_id) {
            max_kf_id = kf->keyframe_id_;
        }

        vertices.insert({kf->keyframe_id_, vertex_pose});
    }

    // 路标顶点，使用路标id索引
    std::map<unsigned long, VertexXYZ *> vertices_landmarks;

    Mat3d K = camera_left_->K();
    SE3 left_ext = camera_left_->pose();
    SE3 right_ext = camera_right_->pose();

    // edges
    int index = 1;
    double chi2_th = 5.991;  // robust kernel 阈值
    std::map<EdgeProjection *, FeaturePtr> edges_and_features;
    for (auto &landmark : landmarks) {
        if (landmark.second->is_outlier_) continue;
        unsigned long landmark_id = landmark.second->id_;
        auto observations = landmark.second->observations_;
        for (auto &obs : observations) {
            if (obs.lock() == nullptr) continue;
            auto feat = obs.lock();
            if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue;

            auto frame = feat->frame_.lock();
            EdgeProjection *edge = nullptr;
            if (feat->is_on_left_image_) {
                edge = new EdgeProjection(K, left_ext);
            } else {
                edge = new EdgeProjection(K , right_ext);
            }

            // 如果landmark还没有被加入优化，则新加一个顶点
            if (vertices_landmarks.find(landmark_id) ==
                vertices_landmarks.end()) {
                VertexXYZ *v = new VertexXYZ;
                v->setEstimate(landmark.second->pos_);
                v->setId(landmark_id + max_kf_id + 1);
                v->setMarginalized(true);
                vertices_landmarks.insert({landmark_id, v});
                optimizer.addVertex(v);
            }


            if (vertices.find(frame->keyframe_id_) !=
                vertices.end() && 
                vertices_landmarks.find(landmark_id) !=
                vertices_landmarks.end()) {
                    edge->setId(index);
                    edge->setVertex(0, vertices.at(frame->keyframe_id_));    // pose
                    edge->setVertex(1, vertices_landmarks.at(landmark_id));  // landmark
                    edge->setMeasurement(Vec2d(feat->pixel_pt_.pt.x, feat->pixel_pt_.pt.y));
                    edge->setInformation(Mat2d::Identity());
                    auto rk = new g2o::RobustKernelHuber();
                    rk->setDelta(chi2_th);
                    edge->setRobustKernel(rk);
                    edges_and_features.insert({edge, feat});
                    optimizer.addEdge(edge);
                    index++;
                }
            else delete edge;
                
        }
    }

    // do optimization and eliminate the outliers
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    while (iteration < 5) {
        cnt_outlier = 0;
        cnt_inlier = 0;
        // determine if we want to adjust the outlier threshold
        for (auto &ef : edges_and_features) {
            if (ef.first->chi2() > chi2_th) {
                cnt_outlier++;
            } else {
                cnt_inlier++;
            }
        }
        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
        if (inlier_ratio > 0.5) {
            break;
        } else {
            chi2_th *= 2;
            iteration++;
        }
    }

    for (auto &ef : edges_and_features) {
        if (ef.first->chi2() > chi2_th) {
            ef.second->is_outlier_ = true;
            // remove the observation
            ef.second->map_point_.lock()->RemoveObservation(ef.second);
        } else {
            ef.second->is_outlier_ = false;
        }
    }

    LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
              << cnt_inlier;

    // Set pose and lanrmark position
    for (auto &v : vertices) {
        keyframes.at(v.first)->pose_ = v.second->estimate();
    }
    for (auto &v : vertices_landmarks) {
        landmarks.at(v.first)->pos_ = v.second->estimate();
    }
}