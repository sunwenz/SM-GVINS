#pragma once

#include "types/image.h"
#include "visual/tracker.h"

class Estimator{
   public:
    struct Options{
        SE3 Tbc0_;
        SE3 Tbc1_;
        Tracker::Options tracker_options_;
    };

    Estimator(const Options& options = Options());
    void AddImage(const Image& image);
   private:
    Options options_;
    bool is_first_image_ = true;
    TrackerPtr tracker_ = nullptr;
};
