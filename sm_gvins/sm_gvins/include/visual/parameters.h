#pragma once
#include "eigen_types.h"
#include <string>

struct Parameters
{
    static std::string image0_topic_, image1_topic_, output_path_;
    static int width_, height_; // 图像的宽和高

    static double fx_, fy_, cx_, cy_;
    static double k1_, k2_, k3_, p1_, p2_;
    static SE3 Tc0c1_;
    static float min_X_, max_X_, min_Y_, max_Y_;
    static float base_;    // 双目基线长度（米）
    static float base_fx_; // 双目基线长度（米）*fx_

    static int fps_; // 帧率
    static float key_max_t_; // 与参考帧的最大位移差
    static float key_max_R_; // 与参考帧的最大旋转差
};
