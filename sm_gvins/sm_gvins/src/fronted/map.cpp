#include "map.h"
#include "../utils/math.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>  
#include <glog/logging.h>

void Map::AddKeyFrame(const FramePtr &frame) {
    // New keyframe
    keyframes_.push_back(frame);

    // New landmarks
    TriangulateNewPoints();

    if (keyframes_.size() > window_size_) {
        this->pop_front();
    }
}

void Map::AddMapPoint(unsigned long long id, const Vec3d& point){
     if (landmarks_.find(id) == landmarks_.end()) {
        landmarks_.insert(std::make_pair(id, point));
    } else {
        // landmarks_[id] = point;
    }
}

void Map::pop_front() {
    auto& frame = keyframes_.front();
    // 移除与关键帧关联的所有路标点
    const auto& features = frame->features();
    for (const auto &[id, feature] : features) {
        if(landmarks_.find(id) != landmarks_.end()){
            landmarks_.erase(id);
        }

        for(size_t i = 1; i < window_size_ + 1; ++i){
            if(keyframes_[i]->IsFeatureInFrame(id)){
                keyframes_[i]->RemoveFeature(id);
            }
        }   
    }
    
    frame->clearFeatures();
    keyframes_.pop_front();
}

void Map::TriangulateNewPoints(){
    auto frame = keyframes_.back();
    std::vector<unsigned long long> to_delete;
    auto& features = frame->features();
    for(auto& [id, feature] :features){
        // triangulate the features of this frame
        // 左相机世界位姿
        Sophus::SE3d Twc0 = frame->pose();     
        Sophus::SE3d Tbc0 = Tbc_[0];          
        Sophus::SE3d Tbc1 = Tbc_[1];           
        Sophus::SE3d Tc0c1 = Tbc0.inverse() * Tbc1;
        Sophus::SE3d Twc1 = Twc0 * Tc0c1;        // 世界 -> 右目
        std::vector<Sophus::SE3d> poses{
            // Tbc0, Tbc1
            Twc0.inverse(), Twc1.inverse()
        };

        std::vector<Vec3d> points{
            feature->normlize_pt_, feature->normlize_pt_right_
        };

        // 三角化
        Eigen::Vector3d point3d;
        if(math::triangulatePoint(poses, points, point3d)){
            this->AddMapPoint(feature->id(), point3d);
            feature->world_pt_ = point3d;
        }else{
            to_delete.push_back(feature->id());
        }; 

        // Eigen::Matrix<double, 3, 4> left_pose  = Twc0.inverse().matrix().block<3,4>(0, 0);
        // Eigen::Matrix<double, 3, 4> right_pose = Twc1.inverse().matrix().block<3,4>(0, 0);
        // Eigen::Vector3d point0 =       feature->normlize_pt_;
        // Eigen::Vector3d point1 = feature->normlize_pt_right_;

        // Eigen::Vector3d pw;
        // math::triangulatePoint(left_pose, right_pose, point0, point1, pw);

        // Eigen::Vector3d Pc = Twc0 * pw;  // T.inverse() * P_world
        // double depth = Pc.z();

        // if (depth > 0){
        //     feature->world_pt_ = pw;
        //     this->AddMapPoint(feature->id(), pw);
        // }else{
        //     to_delete.push_back(feature->id());
        // }
    }
    
    for(auto id :to_delete){
        landmarks_.erase(id);
        for(auto frame : keyframes_){
            if(frame->IsFeatureInFrame(id)){
                frame->RemoveFeature(id);
            }
        } 
    }
}

