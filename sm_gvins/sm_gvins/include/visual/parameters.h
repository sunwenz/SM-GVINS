#pragma once
#include "eigen_types.h"
#include <opencv2/opencv.hpp>

struct Parameters
{
    static int width_, height_; // 图像的宽和高

    static float fx_;
    static float min_X_, max_X_, min_Y_, max_Y_;
    static float base_;    // 双目基线长度（米）
    static float base_fx_; // 双目基线长度（米）*fx_

    static int fps_; // 帧率

    static float depth_factor_; // 深度比例（仅RGBD相机）
};
