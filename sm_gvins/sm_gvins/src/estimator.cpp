#include "estimator.h"
// #include "g2o_types.h"
#include "utils.h"

#include <glog/logging.h>
#include <opencv2/core/eigen.hpp>  
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

Estimator::Estimator()
{
    camera_left_  = std::make_shared<Camera>(
        Parameters::fx_, Parameters::fy_, Parameters::cx_, 
        Parameters::cy_, Parameters::base_, SE3()
    );

    camera_right_  = std::make_shared<Camera>(
        Parameters::fx_, Parameters::fy_, Parameters::cx_, 
        Parameters::cy_, Parameters::base_, Parameters::Tc0c1_.inverse()
    );

    map_ = std::make_shared<Map>();
    tracker_ = std::make_shared<Tracker>(map_);
    tracker_->SetCameras(camera_left_, camera_right_);

    process_ = std::thread(&Estimator::Process, this);
}

void Estimator::AddImage(const Image& image){
    auto new_frame = Frame::createFrame(image.timestamp_, image.img_, image.img_right_);
    if(!tracker_->TrackFrame(new_frame)){
        return;
    }

    frame_buf_.push(new_frame);
    imu_preinteg_buf_.push(imu_preinteg_);
    imu_preinteg_ = std::make_shared<IMUPreintegration>();
}

void Estimator::AddIMU(const IMU& imu){
    if(first_imu_flag_){
        first_imu_flag_ = false;
        imu_buf_.push(imu);
        return;
    }
    imu_preinteg_->AddIMU(imu.timestamp_ - imu_buf_.back().timestamp_, imu);
    imu_buf_.push(imu);
}

void Estimator::SetCameras(Camera::Ptr camera_left, Camera::Ptr camera_right){
    camera_left_  = std::move(camera_left);
    camera_right_ = std::move(camera_right);
    tracker_->SetCameras(camera_left_, camera_right_);
}

void Estimator::Process(){
    while(process_running_){
        if(!frame_buf_.empty()){
            buf_mtx_.lock();
            auto frame = frame_buf_.front();
            frame_buf_.pop();
            buf_mtx_.unlock();

            ProcessImage(frame);
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void Estimator::ProcessImage(FramePtr frame){
    if(initlized_flag_ == false){
        
    }
}

void Estimator::Optimize(){
    /* // setup g2o
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
    } */
}