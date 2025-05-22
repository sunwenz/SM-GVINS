#include "dataset.h"
#include "image.h"
#include <glog/logging.h>

#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace std;


Dataset::Dataset(const std::string& dataset_path)
    : dataset_path_(dataset_path) {}

bool Dataset::Init() {
    // read camera intrinsics and extrinsics
    ifstream fin(dataset_path_ + "/calib.txt");
    if (!fin) {
        LOG(ERROR) << "cannot find " << dataset_path_ << "/calib.txt!";
        return false;
    }

    for (int i = 0; i < 4; ++i) {
        char camera_name[3];
        for (int k = 0; k < 3; ++k) {
            fin >> camera_name[k];
        }
        double projection_data[12];
        for (int k = 0; k < 12; ++k) {
            fin >> projection_data[k];
        }
        Mat3d K;
        K << projection_data[0], projection_data[1], projection_data[2],
            projection_data[4], projection_data[5], projection_data[6],
            projection_data[8], projection_data[9], projection_data[10];
        Vec3d t;
        t << projection_data[3], projection_data[7], projection_data[11];
        t = K.inverse() * t;
        K = K * 0.5;
        Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                          t.norm(), SE3(SO3(), t)));
        cameras_.push_back(new_camera);
        LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();
    }
    fin.close();
    current_image_index_ = 0;

    fin_time_ = std::ifstream(dataset_path_ + "/times.txt");
    return true;
}

FramePtr Dataset::NextFrame() {
    if (!fin_time_.good()) {
        LOG(WARNING) << "times.txt stream is not good.";
        return nullptr;
    }

    // 构造图像路径（每次 new boost::format，防止状态污染）
    std::string left_path =
        (boost::format("%s/image_%d/%06d.png") % dataset_path_ % 0 % current_image_index_).str();
    std::string right_path =
        (boost::format("%s/image_%d/%06d.png") % dataset_path_ % 1 % current_image_index_).str();

    // 读取图像
    cv::Mat image_left = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
    cv::Mat image_right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);

    if (image_left.empty() || image_right.empty()) {
        LOG(WARNING) << "Cannot find images at index " << current_image_index_;
        return nullptr;
    }

    // 缩放图像
    cv::Mat image_left_resized, image_right_resized;
    cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
    cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);

    // 读取时间戳
    double time = 0;
    if (!(fin_time_ >> time)) {
        LOG(WARNING) << "Cannot read timestamp at index " << current_image_index_;
        return nullptr;
    }

    // 构造图像对象
    Image image;
    image.timestamp_ = time;
    image.img_ = image_left_resized;
    image.img_right_ = image_right_resized;

    // 构造帧对象
    auto new_frame = Frame::CreateFrame(image);
    current_image_index_++;
    return new_frame;
}


