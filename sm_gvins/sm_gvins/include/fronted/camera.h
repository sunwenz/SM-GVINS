#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

using cv::Mat;
using Eigen::Quaterniond;
using Eigen::Vector2d;
using Eigen::Vector3d;

class Camera {

public:
    using Ptr = std::shared_ptr<Camera>;

    Camera() = delete;

    Camera(const Mat& intrinsic, const Mat& distortion, const cv::Size &size);

    static Camera::Ptr createCamera(const Mat& intrinsic, const Mat& distortion, const cv::Size &size);

    const Mat &cameraMatrix() const {
        return intrinsic_;
    }

    cv::Point2f pixel2cam(const cv::Point2f &pixel) const;

private:
    Mat distortion_;
    Mat undissrc_, undisdst_;

    double fx_, fy_, cx_, cy_, skew_;
    double k1_, k2_, k3_, p1_, p2_;
    Mat intrinsic_;

    int width_, height_;
};