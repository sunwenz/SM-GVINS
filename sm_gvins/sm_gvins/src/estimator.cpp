#include "estimator.h"
#include "g2o_types.h"
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
    Rbc_[0] = Parameters::Tbc0_.rotationMatrix();
    Rbc_[1] = Parameters::Tbc1_.rotationMatrix();
    tbc_[0] = Parameters::Tbc0_.translation();
    tbc_[1] = Parameters::Tbc1_.translation();

    f_out_ = std::ofstream(Parameters::output_path_);
    process_ = std::thread(&Estimator::Process, this);
}

void Estimator::AddImage(const Image& image){
    auto new_frame = Frame::createFrame(image.timestamp_, image.img_, image.img_right_);

    std::lock_guard<std::mutex> lk(buf_mtx_);
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
            FramePtr frame = nullptr;
            {   
                std::lock_guard<std::mutex> lk(buf_mtx_);
                frame = frame_buf_.front();
                frame_buf_.pop();
            }

            ProcessImage(frame);

            utils::save_result(f_out_, NavStated(frame->timestamp_, frame->Twc_));
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void Estimator::ProcessImage(FramePtr frame){
    if(!tracker_->TrackFrame(frame))    return;

    state_window_.push_back(
        std::make_shared<NavStated>(
            frame->timestamp_, frame->Twc_
        )
    );

    if (state_window_.size() > 10) {
        state_window_.pop_front();  // 移除最旧的帧
    }

    if(state_window_.size() > 1)
        Optimize();
}

void Estimator::Optimize(){
    // setup g2o
    using BlockSolverType = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
    
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // extrincs
    VertexRot* vertex_rbc = new VertexRot();
    vertex_rbc->setId(0);
    vertex_rbc->setEstimate(Rbc_[0]);
    vertex_rbc->setFixed(true);
    optimizer.addVertex(vertex_rbc);

    VertexPosition* vertex_tbc = new VertexPosition();
    vertex_tbc->setId(1);
    vertex_tbc->setEstimate(tbc_[0]);
    vertex_tbc->setFixed(true);
    optimizer.addVertex(vertex_tbc);

    // pose 顶点，使用Keyframe  id
    std::map<double, VertexRot *> vertices_rot;
    std::map<double, VertexPosition *> vertices_pos;
    unsigned long max_kf_id = 2;
    for (size_t i = 0; i < state_window_.size(); ++i) {
        auto state = state_window_[i];

        auto rot = state->R_;
        VertexRot *v_rot = new VertexRot();  // camera vertex_pose
        v_rot->setId(max_kf_id++);
        v_rot->setEstimate(state->R_);
        optimizer.addVertex(v_rot);
        vertices_rot.insert({state->timestamp_, v_rot});

        auto pos = state->p_;
        VertexPosition* v_pos = new VertexPosition();
        v_pos->setId(max_kf_id++);
        v_pos->setEstimate(pos);
        optimizer.addVertex(v_pos);
        vertices_pos.insert({state->timestamp_, v_pos});
    }

    // 路标顶点，使用路标id索引
    Mat3d K = camera_left_->K();
    unsigned long max_mp_id = 0;
    int edge_idx = 1;
    double chi2_th = 5.991;
    
    std::unordered_map<unsigned long, VertexInverseDepth*> inverse_depth_vertices;
    std::unordered_map<unsigned long, FeaturePtr> landmark_ref_features; // Stores the feature that initializes the inverse depth
    std::vector<EdgeProjection*> edges;
    for(auto& keyframe : map_->keyframes()){
        for(auto& ft : keyframe->features_){
            if(ft == nullptr || ft->map_point_.expired())   continue;
            auto mp = ft->map_point_.lock();
            if(landmark_ref_features.find(mp->id_) != landmark_ref_features.end()){
                VertexInverseDepth* v = new VertexInverseDepth;
                auto ref_ft = landmark_ref_features[mp->id_];
                auto ref_kf = ref_ft->frame_.lock();
                double depth = (ref_kf->Twc_ * mp->pos_).z();
                v->setEstimate(1 / depth);
                v->setId(max_kf_id + max_mp_id + 1);
                v->setMarginalized(true);
                max_mp_id++;
                inverse_depth_vertices[mp->id_] = v;
                optimizer.addVertex(v);

                EdgeProjection* edge = new EdgeProjection(
                    camera_left_->pixel2camera(
                        Vec2d(ref_ft->pixel_pt_.pt.x, ref_ft->pixel_pt_.pt.y)
                        ), 
                    camera_left_->pixel2camera(
                        Vec2d(ft->pixel_pt_.pt.x, ft->pixel_pt_.pt.y)
                        )
                );
                
                edge->setId(edge_idx);
                edge->setVertex(0, vertices_rot[ref_kf->timestamp_]);
                edge->setVertex(1, vertices_pos[ref_kf->timestamp_]);
                edge->setVertex(2, vertices_rot[keyframe->timestamp_]);
                edge->setVertex(3, vertices_pos[keyframe->timestamp_]);
                edge->setVertex(4, vertex_rbc);
                edge->setVertex(5, vertex_tbc);
                edge->setVertex(6, v);
                edge->setInformation(Mat2d::Identity()/* EdgeProjection::sqrt_info */);
                auto rk = new g2o::RobustKernelHuber();
                rk->setDelta(chi2_th);
                edge->setRobustKernel(rk);
                optimizer.addEdge(edge);
                edges.push_back(edge);
                edge_idx++;

            }else
                landmark_ref_features[mp->id_] = ft;
        }
    }

    // do optimization and eliminate the outliers
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // Set pose and lanrmark position
    for (size_t i = 0; i < state_window_.size(); ++i) {
        const double t = state_window_[i]->timestamp_;
        map_->SetFramePose(i, SE3(vertices_rot[t]->estimate(), vertices_pos[t]->estimate()));
        state_window_[i]->R_ = vertices_rot[t]->estimate();
        state_window_[i]->p_ = vertices_pos[t]->estimate();
    }

    for (auto &v : inverse_depth_vertices) {
        auto ref_ft = landmark_ref_features[v.first];
        auto ref_kf = ref_ft->frame_.lock();
        map_->SetLandmarkDepth(
            v.first, 
            ref_kf->Twc_.inverse() * camera_left_->pixel2camera(Vec2d(ref_ft->pixel_pt_.pt.x, ref_ft->pixel_pt_.pt.y)) / v.second->estimate()
        );
    }
}