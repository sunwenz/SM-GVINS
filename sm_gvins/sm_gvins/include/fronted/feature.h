#pragma once

#include <memory>

#include "eigen_types.h"

class Feature {
public:
    Feature() = delete;
    Feature(unsigned long long id)   
        : id_(id) {}

    static std::shared_ptr<Feature> createFeature(unsigned long long id) {
        return std::make_shared<Feature>(id);
    }

    unsigned long long id() const {
        return id_;
    }

    Vec3d world_pt_;
    Vec3d normlize_pt_, normlize_pt_right_;
    Vec2d pixel_pt_, pixel_pt_right_;
    Vec2d vel_, vel_right_;
    
    bool is_outlier_ = false;
private:
    unsigned long long id_;
};
using FeaturePtr = std::shared_ptr<Feature>;