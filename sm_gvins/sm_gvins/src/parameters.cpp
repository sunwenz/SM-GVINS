#include "parameters.h"

std::string Parameters::image0_topic_ = std::string();
std::string Parameters::image1_topic_ = std::string();
std::string Parameters::output_path_ = std::string();
int Parameters::width_ = 0, Parameters::height_ = 0; // 图像的宽和高
int Parameters::image_fps_ = 0; // 帧率
float Parameters::base_ = 0;    // 双目基线长度（米）
float Parameters::base_fx_ = 0; 
float Parameters::min_X_ = 0, Parameters::max_X_ = 0, Parameters::min_Y_ = 0, Parameters::max_Y_ = 0;
double Parameters::fx_ = 0, Parameters::fy_ = 0, Parameters::cx_ = 0, Parameters::cy_ = 0;
double Parameters::k1_ = 0, Parameters::k2_ = 0, Parameters::k3_ = 0, Parameters::p1_ = 0, Parameters::p2_ = 0;

SE3 Parameters::Tc0c1_ = SE3();
SE3 Parameters::Twb_ = SE3();
SE3 Parameters::Tbc0_ = SE3();
SE3 Parameters::Tbc1_ = SE3();
SE3 Parameters::Twn_ = SE3();

int Parameters::window_size_ = 10;

