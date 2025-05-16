#include "frame.h"

Frame::Frame(long id, const Image& image)
    : id_(id), timestamp_(image.timestamp_), left_img_(image.img_), right_img_(image.img_right_) {}

std::shared_ptr<Frame>  Frame::CreateFrame(const Image& image) {
    static long factory_id = 0;
    return std::make_shared<Frame>(factory_id++, image);
}

void Frame::SetKeyFrame() {
    static long keyframe_factory_id = 0;
    is_keyframe_ = true;
    keyframe_id_ = keyframe_factory_id++;
}