#pragma once

#include "frame.h"
#include "feature.h"

struct Frame;
struct Feature;
struct MapPoint {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    unsigned long id_ = 0;  // ID
    bool is_outlier_ = false;
    Vec3d pos_ = Vec3d::Zero();  // Position in world
    int observed_times_ = 0;  // being observed by feature matching algo.
    std::list<std::weak_ptr<Feature>> observations_;

    MapPoint() {}

    MapPoint(long id, Vec3d position);

    void AddObservation(std::shared_ptr<Feature> feature) {
        observations_.push_back(feature);
        observed_times_++;
    }

    void RemoveObservation(std::shared_ptr<Feature> feat);

    // factory function
    static std::shared_ptr<MapPoint> CreateNewMappoint();
};
using MapPointPtr = std::shared_ptr<MapPoint>;
