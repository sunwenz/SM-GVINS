#include "map.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>  
#include <glog/logging.h>

void Map::InsertKeyFrame(FramePtr frame) {
    current_frame_ = frame;
    frame_window_.push_back(frame);

    if (frame_window_.size() > window_size_) {
        RemoveOldKeyframe();
    }
}

void Map::InsertMapPoint(MapPointPtr map_point) {
    if (landmarks_.find(map_point->id_) == landmarks_.end()) {
        landmarks_.insert(make_pair(map_point->id_, map_point));
    } else {
        landmarks_[map_point->id_] = map_point;
    }
}

void Map::RemoveOldKeyframe() {
    if (current_frame_ == nullptr) return;

    auto frame_to_remove = frame_window_.front();
    frame_window_.pop_front(); 
    for (auto feat : frame_to_remove->features_) {
        if(feat == nullptr) continue;
        auto mp = feat->map_point_.lock();
        if (mp) {
            mp->RemoveObservation(feat);
        }
    }
    CleanMap();
}

void Map::CleanMap() {
    int cnt_landmark_removed = 0;
    for (auto iter = landmarks_.begin();
         iter != landmarks_.end();) {
        if (iter->second->observed_times_ == 0) {
            iter = landmarks_.erase(iter);
            cnt_landmark_removed++;
        } else {
            ++iter;
        }
    }
    LOG(INFO) << "Removed " << cnt_landmark_removed << " active landmarks";
}