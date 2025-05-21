#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "eigen_types.h"

using cv::Mat;

class Camera {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Camera> Ptr;

    double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0,
           baseline_ = 0;  // Camera intrinsics
    double k1_, k2_, k3_, p1_, p2_;
    
    Mat intrinsic_;
    Mat distortion_;
    Mat undissrc_, undisdst_;
    int width_, height_;

    SE3 pose_;             // extrinsic, from stereo camera to single camera
    SE3 pose_inv_;         // inverse of extrinsics

    Camera(Mat intrinsic, Mat distortion, const cv::Size &size);

    void undistortImage(const Mat &src, Mat &dst);
    
    SE3 pose() const { return pose_; }

    // return intrinsic matrix
    Mat3d K() const {
        Mat3d k;
        k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
        return k;
    }

    cv::Mat cvK() const {
        return intrinsic_;
    }


    cv::Mat cvD() const {
        return distortion_;
    }


    // coordinate transform: world, camera, pixel
    Vec3d world2camera(const Vec3d &p_w, const SE3 &T_c_w);

    Vec3d camera2world(const Vec3d &p_c, const SE3 &T_c_w);

    Vec2d camera2pixel(const Vec3d &p_c);

    Vec3d pixel2camera(const Vec2d &p_p, double depth = 1);

    Vec3d pixel2world(const Vec2d &p_p, const SE3 &T_c_w, double depth = 1);

    Vec2d world2pixel(const Vec3d &p_w, const SE3 &T_c_w);
};