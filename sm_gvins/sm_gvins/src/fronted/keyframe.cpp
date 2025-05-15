#include "keyframe.h"

Frame::Frame(ulong id, double timestamp)
    : timestamp_(timestamp)
{
    features_.clear();
    // unupdated_mappoints_.clear();
}

std::shared_ptr<Frame> Frame::createFrame(double stamp) {
    static ulong factory_id = 0;

    return std::make_shared<Frame>(factory_id++, stamp);
}
