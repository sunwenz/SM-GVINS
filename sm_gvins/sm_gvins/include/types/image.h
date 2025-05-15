#pragma once

#include <memory>
#include <opencv2/opencv.hpp>

struct Image
{
    Image() = default;
    Image(double timestamp, cv::Mat img, cv::Mat img_right = cv::Mat())
        : timestamp_(timestamp), img_(img), img_right_(img_right) {}

    // unsigned long id_ = 0;
    double timestamp_ = 0.0;
    cv::Mat img_ = cv::Mat();
    cv::Mat img_right_;
};
using ImagePtr = std::shared_ptr<Image>;