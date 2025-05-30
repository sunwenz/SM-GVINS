#include "frame.h"
#include "math.h"

#include <thread>

Frame::Frame(int id, double timestamp, cv::Mat left_img, cv::Mat right_img)
    : id_(id),
      timestamp_(timestamp),
      left_img_(left_img),
      right_img_(right_img)
{
    mask_ = cv::Mat::zeros(left_img_.size(), CV_8UC1);
    orbleft_.reset(new ORBextractor);
    orbright_.reset(new ORBextractor);
}


std::shared_ptr<Frame> Frame::createFrame(double stamp, cv::Mat left_img, cv::Mat right_img) {
    static ulong factory_id = 0;

    return std::make_shared<Frame>(factory_id++, stamp, left_img, right_img);
}

void Frame::SetKeyframe() {
    static long keyframe_factory_id = 0;
    is_keyframe_ = true;
    keyframe_id_ = keyframe_factory_id++;
}

void Frame::ExtractKeyPointsAndDescriptors()
{
    // 提取ORB特征点
    // TODO: 将描述子的计算部分移到calcDescriptors()函数里
    std::thread orbleft(&ORBextractor::extractORB, orbleft_, left_img_, std::ref(keypoints_l_), std::ref(descriptors_l_));
    std::thread orbright(&ORBextractor::extractORB, orbright_, right_img_, std::ref(keypoints_r_), std::ref(descriptors_r_));
    orbleft.join();
    orbright.join();

    // cv::Mat outimg_left, outimg_right, concat_img;
    // cv::drawKeypoints(left_img_, keypoints_l_, outimg_left, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
    // cv::drawKeypoints(right_img_, keypoints_r_, outimg_right, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);
    // cv::hconcat(outimg_left, outimg_right, concat_img); // 水平拼接
    // cv::imshow("ORB Features - Left | Right", concat_img);
    // cv::waitKey(0);

    // 图像金字塔，只保存两层
    scimg_ = orbleft_->mvImagePyramid[1];

    // 把特征点分配到网格中以加速匹配
    ORBextractor::AssignfeatoGrid(keypoints_l_, orbleft_);
}

void Frame::MatchFromeStereo()
{
    matchLeftRight(keypoints_l_, keypoints_r_, left_to_right_, descriptors_l_, descriptors_r_, orbleft_, orbright_); 
}

void Frame::UndistKeyPoints()
{
    // if (!Camera::k1_)
    // {
    //     return;
    // }

    // // 构建关键点坐标数组
    // int num_kp = keypoints_l_.size();
    // cv::Mat kpts(num_kp, 2, CV_32F);
    // for (int i = 0; i < num_kp; i++)
    // {
    //     kpts.ptr<float>(i)[0] = keypoints_l_[i].pt.x;
    //     kpts.ptr<float>(i)[1] = keypoints_l_[i].pt.y;
    // }

    // // 计算去畸变后的关键点坐标
    // cv::undistortPoints(kpts, kpts, Camera::cvK_, Camera::D_, cv::Mat(), Camera::cvK_);

    // // 坐标赋值
    // for (int i = 0; i < num_kp; i++)
    // {
    //     keypoints_l_[i].pt.x = kpts.ptr<float>(i)[0];
    //     keypoints_l_[i].pt.y = kpts.ptr<float>(i)[1];
    // }
}

void Frame::CreateFeatures()
{
    features_.resize(keypoints_l_.size(), nullptr);
    int num_feats = 0;
    for (int i = 0; i < keypoints_l_.size(); i++)
    {
        if (left_to_right_[i] == 0.0) continue;
        
        FeaturePtr ft(new Feature);
        // 坐标、描述子
        ft->pixel_pt_ = keypoints_l_[i];
        ft->octave_ = keypoints_l_[i].octave;
        ft->descriptor_ = descriptors_l_.row(i);
        ft->x_r_ = left_to_right_[i];
        ft->pixel_pt_right_ = cv::KeyPoint(left_to_right_[i], ft->pixel_pt_.pt.y, ft->pixel_pt_.size);
        // num_features_++;
        num_feats++;
        features_[i] = ft;
    }

    LOG(INFO) << "当前帧可用特征点数量：" << num_feats;
}

