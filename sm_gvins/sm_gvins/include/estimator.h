#pragma once

#include "types/image.h"
#include "types/nav_state.h"
#include "visual/tracker.h"

class Estimator{
   public:
    struct Options{
        Mat3d K_[2];
        Vec3d t_[2];
        SE3 Tbc0_;
        SE3 Tbc1_;
        Tracker::Options tracker_options_;
    };

    Estimator(std::shared_ptr<std::ofstream> f_out, const Options& options = Options());
    
    void AddImage(const Image& image);

    NavStated GetNavState() const {
        return NavStated(map_->current_frame_->timestamp_, map_->current_frame_->pose_);
    }
   private:
    void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);

    Options options_;
    bool is_first_image_ = true;
    TrackerPtr tracker_ = nullptr;
    MapPtr map_ = nullptr;
    Camera::Ptr camera_left_ = nullptr, camera_right_ = nullptr;
    std::shared_ptr<std::ofstream> f_out_;
};
