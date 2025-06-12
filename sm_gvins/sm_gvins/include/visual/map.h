#pragma once

#include "frame.h"
#include "feature.h"
#include "nav_state.h"
#include "mappoint.h"

#include <memory>
#include <deque>
#include <unordered_map>

class Map {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Map() {}
    
    const std::deque<FramePtr>& keyframes(){
        return frame_window_;
    }

    const std::unordered_map<unsigned long, MapPointPtr>& landmarks(){
        return landmarks_;
    }

    void SetFramePose(int idx, SE3 pose){
        std::lock_guard<std::mutex> lk(map_mtx_);
        frame_window_[idx]->Twc_ = pose;
    }

    void SetLandmarkDepth(unsigned long id, Vec3d pos){
        std::lock_guard<std::mutex> lk(map_mtx_);
        landmarks_[id]->pos_ = pos;
    }

    /// 增加一个关键帧
    void InsertKeyFrame(FramePtr frame);
    /// 增加一个地图顶点
    void InsertMapPoint(MapPointPtr map_point);

    /// 清理map中观测数量为零的点
    void CleanMap();

    FramePtr current_frame_ = nullptr;
   private:
    void RemoveOldKeyframe();

    std::mutex map_mtx_;
    std::deque<FramePtr> frame_window_;
    std::unordered_map<unsigned long, MapPointPtr> landmarks_;

    // settings
    const int window_size_ = 10;  // 激活的关键帧数量
};
using MapPtr = std::shared_ptr<Map>;