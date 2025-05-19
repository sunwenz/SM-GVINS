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
    using LandmarksType = std::unordered_map<unsigned long, MapPointPtr> ;
    using KeyframesType = std::unordered_map<unsigned long, FramePtr>    ;

    Map() {}

    /// 增加一个关键帧
    void InsertKeyFrame(FramePtr frame);
    /// 增加一个地图顶点
    void InsertMapPoint(MapPointPtr map_point);

    /// 获取所有地图点
    LandmarksType GetAllMapPoints() {
        return landmarks_;
    }
    /// 获取所有关键帧
    KeyframesType GetAllKeyFrames() {
        return keyframes_;
    }

    /// 获取激活地图点
    LandmarksType GetActiveMapPoints() {
        return active_landmarks_;
    }

    /// 获取激活关键帧
    KeyframesType GetActiveKeyFrames() {
        return active_keyframes_;
    }

    /// 清理map中观测数量为零的点
    void CleanMap();

    FramePtr current_frame_ = nullptr;
   private:
    // 将旧的关键帧置为不活跃状态
    void RemoveOldKeyframe();

    LandmarksType landmarks_;         // all landmarks
    LandmarksType active_landmarks_;  // active landmarks
    KeyframesType keyframes_;         // all key-frames
    KeyframesType active_keyframes_;  // all key-frames

    // settings
    int num_active_keyframes_ = 7;  // 激活的关键帧数量
};
using MapPtr = std::shared_ptr<Map>;