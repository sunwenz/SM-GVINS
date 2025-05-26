#include "parameters.h"

int Parameters::width_ = 0, Parameters::height_ = 0; // 图像的宽和高
float Parameters::base_ = 0;    // 双目基线长度（米）
int Parameters::fps_ = 0; // 帧率
float Parameters::depth_factor_ = 0; // 深度比例（仅RGBD相机）

float Parameters::fx_ = 0;
float Parameters::min_X_ = 0, Parameters::max_X_ = 0, Parameters::min_Y_ = 0, Parameters::max_Y_ = 0;
float Parameters::base_fx_ = 0; 