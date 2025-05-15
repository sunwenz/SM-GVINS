#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <opencv2/opencv.hpp>

#include "sophus/se3.hpp"
#include "image.h"
#include "feature.h"

class Frame {
public:
    using SE3 = Sophus::SE3d;
    using SE3f = Sophus::SE3f;
    using SO3 = Sophus::SO3d;

    Frame() = delete;
    Frame(ulong id, double timestamp);

    static std::shared_ptr<Frame> createFrame(double stamp);

    SE3 pose() {
        return Twc_;
    }

    void setPose(const SE3& pose) {
        Twc_ = pose;
    }

    std::unordered_map<unsigned long long, FeaturePtr>& features() {
        return features_;
    }

    void addFeature(unsigned long long id, const FeaturePtr &features) {
        features_.insert(std::make_pair(id, features));
    }

    void RemoveFeature(unsigned long long id){
        features_.erase(id);
    }
    
    bool IsFeatureInFrame(unsigned long long id){
        return features_.find(id) != features_.end();
    }

    void clearFeatures() {
        features_.clear();
    }

    size_t numFeatures() {
        return features_.size();
    }

    double stamp() const {
        return timestamp_;
    }

    void setStamp(double stamp) {
        timestamp_ = stamp;
    }

    double timeDelay() const {
        return td_;
    }

private:
    double timestamp_;
    double td_{0};

    SE3 Twc_;

    std::unordered_map<unsigned long long, FeaturePtr> features_;
    // vector<std::shared_ptr<MapPoint>> unupdated_mappoints_;
};
using FramePtr = std::shared_ptr<Frame>;

