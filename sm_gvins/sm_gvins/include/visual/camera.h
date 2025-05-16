#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "eigen_types.h"

using cv::Mat;

class Camera {

public:
    using Ptr = std::shared_ptr<Camera>;

    Camera() = delete;

    Camera(const Mat& intrinsic, const Mat& distortion, const cv::Size &size);

    static Camera::Ptr createCamera(const Mat& intrinsic, const Mat& distortion, const cv::Size &size);

    const Mat& cvK() const {
        return intrinsic_;
    }

    Mat3d K() const {
        Mat3d k;
        k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
        return k;
    }

    Vec3d world2camera(const Vec3d &p_w, const SE3 &T_c_w);
    Vec3d camera2world(const Vec3d &p_c, const SE3 &T_c_w);
    Vec2d camera2pixel(const Vec3d &p_c);
    Vec3d pixel2camera(const Vec2d &p_p, double depth = 1);
    Vec3d pixel2world(const  Vec2d &p_p, const SE3 &T_c_w, double depth = 1);
    Vec2d world2pixel(const  Vec3d &p_w, const SE3 &T_c_w);
private:
    Mat distortion_;
    Mat undissrc_, undisdst_;

    double fx_, fy_, cx_, cy_, skew_;
    double k1_, k2_, k3_, p1_, p2_;
    Mat intrinsic_;

    int width_, height_;
};