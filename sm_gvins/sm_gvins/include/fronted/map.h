#pragma once

#include "keyframe.h"
#include "feature.h"
#include "nav_state.h"

#include <memory>
#include <deque>
#include <unordered_map>

class Map {

public:
    Map() = delete;
    explicit Map(size_t size)
        : window_size_(size) {
    }

    void resetWindowSize(size_t size) {
        window_size_ = size;
    }

    NavStated GetState(){
        return NavStated(keyframes_.back()->stamp(), keyframes_.back()->pose());
    }

    FramePtr BackFrame(){
        return keyframes_.back();
    }

    Vec3d PointInId(unsigned long long id){
        return landmarks_[id];
    }

    size_t windowSize() const {
        return window_size_;
    }

    bool isWindowFull() const {
        return is_window_full_;
    }

    bool isWindowNormal() const {
        return keyframes_.size() == window_size_;
    }

    void SetTbc(const std::vector<SE3>& Tbc){
        Tbc_ = Tbc;
    }

    bool IsPointInMap(unsigned long long id){
        return landmarks_.find(id) != landmarks_.end();
    }
    
    void AddKeyFrame(const FramePtr &frame);
    void AddMapPoint(unsigned long long id, const Vec3d& point);
    void pop_front();

    void TriangulateNewPoints();

    std::deque<FramePtr> keyframes_;
    std::unordered_map<unsigned long long, Vec3d> landmarks_;

    size_t window_size_ = 10;
    bool is_window_full_ = false;

    std::vector<SE3> Tbc_;
};
using MapPtr = std::shared_ptr<Map>;