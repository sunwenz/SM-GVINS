#include "estimator.h"
#include <glog/logging.h>

Estimator::Estimator(){
    tracker_ = std::make_shared<Tracker>(options_.tracker_options_);
}

void Estimator::AddImage(const Image& image){
    if(is_first_image_){
        if(tracker_->Initilize(image)){
            is_first_image_ = false;
        }
        return;
    }

    // bool is_keyframe = tracker_->TrackFrame(image);
    // if(is_keyframe){
        
    // }else{
    //     LOG(INFO) << "the image in stamp: " << image.timestamp_ << " is not keyframe";
    // }
}