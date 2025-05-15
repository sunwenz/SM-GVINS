#include "camera.h"

Camera::Camera(const Mat& intrinsic, const Mat& distortion, const cv::Size &size)
    : distortion_(std::move(distortion))
    , intrinsic_(std::move(intrinsic)) {

    fx_   = intrinsic_.at<double>(0, 0);
    skew_ = intrinsic_.at<double>(0, 1);
    cx_   = intrinsic_.at<double>(0, 2);
    fy_   = intrinsic_.at<double>(1, 1);
    cy_   = intrinsic_.at<double>(1, 2);

    k1_ = distortion_.at<double>(0);
    k2_ = distortion_.at<double>(1);
    p1_ = distortion_.at<double>(2);
    p2_ = distortion_.at<double>(3);
    k3_ = distortion_.at<double>(4);

    width_  = size.width;
    height_ = size.height;

    // 相机畸变矫正初始化
    initUndistortRectifyMap(intrinsic_, distortion_, Mat(), intrinsic_, size, CV_16SC2, undissrc_, undisdst_);
}

Camera::Ptr Camera::createCamera(const Mat& intrinsic, const Mat& distortion, const cv::Size &size) {
    return std::make_shared<Camera>(intrinsic, distortion, size);
}

cv::Point2f Camera::pixel2cam(const cv::Point2f &pixel) const {
    float y = (pixel.y - cy_) / fy_;
    float x = (pixel.x - cx_ - skew_ * y) / fx_;
    return {x, y};
}



