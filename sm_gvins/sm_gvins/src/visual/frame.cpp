#include "frame.h"
#include "math.h"

#include <thread>

Frame::Frame(int id, double timestamp, cv::Mat left_img, cv::Mat right_img = cv::Mat())
    : id_(id),
      timestamp_(timestamp),
      left_img_(left_img),
      right_img_(right_img)
{
    mask_ = cv::Mat::zeros(left_img_.size(), CV_8UC1);
    orbleft_.reset(new ORBextractor);
    orbright_.reset(new ORBextractor);
}


std::shared_ptr<Frame> Frame::createFrame(double stamp, cv::) {
    static ulong factory_id = 0;

    return std::make_shared<Frame>(factory_id++, stamp);
}

void Frame::ExtractKeyPointsAndDescriptors()
{
    // 提取ORB特征点
    // TODO: 将描述子的计算部分移到calcDescriptors()函数里
    std::thread orbleft(&ORBextractor::extractORB, orbleft_, left_img_, std::ref(keypoints_l_), std::ref(descriptors_l_));
    std::thread orbright(&ORBextractor::extractORB, orbright_, right_img_, std::ref(keypoints_r_), std::ref(descriptors_r_));
    orbleft.join();
    orbright.join();
    // Mat outimg1;
    // drawKeypoints(left_, keypoints_l_, outimg1, Scalar(0,0, 255), DrawMatchesFlags::DEFAULT);
    // // cout << "size of ORB Features keypoints of img_1:" << keypoints_1.size() << endl;
    // imshow("ORB features", outimg1);
    // waitKey(0);

    // 图像金字塔，只保存两层
    scimg_ = orbleft_->mvImagePyramid[1];


    // 把特征点分配到网格中以加速匹配
    ORBextractor::AssignfeatoGrid(keypoints_l_, orbleft_);
}

void Frame::MatchFromeStereo()
{
    matchLeftRight(keypoints_l_, keypoints_r_, left_to_right_, descriptors_l_, descriptors_r_, orbleft_, orbright_); //// 有没有防止右目点在左目点右边的
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
    features_.reserve(keypoints_l_.size());
    for (int i = 0; i < keypoints_l_.size(); i++)
    {
        FeaturePtr ft(new Feature);
        // 坐标、描述子
        ft->pixel_pt_ = keypoints_l_[i];
        ft->octave_ = keypoints_l_[i].octave;
        ft->response_ = keypoints_l_[i].response;
        ft->descriptor_ = descriptors_l_.row(i);

        if (left_to_right_[i] == 0)
        {
            ft->type_ = 0;
        }
        else
        {
            ft->type_ = 1;
            ft->pixel_pt_right_ = cv::Point2f(left_to_right_[i], ft->pixel_pt_.pt.y);
            num_features_++;
        }

        features_.push_back(ft);
    }
}

/**
 * @brief 当前帧与上一帧或关键帧进行特征匹配
 * 上一帧中type_不为0的特征与当前帧所有特征进行匹配
 * 关键帧中type_不为0的特征与当前帧所有特征进行暴力匹配
 * 当前帧的初始位姿用匀速模型计算出的
 * @param frame 上一帧或关键帧
 * @param th 阈值
 */
bool Frame::MatchFeatures(std::shared_ptr<Frame> frame, const SE3 &initPose, int th)
{
    if (matchFeaturesByProjection(frame, initPose, th))
    {
        return true;
    }
    else if (matchFeaturesByBruteForce(frame, initPose, th))
    {
        return true;
    }
    return false;

}

bool Frame::MatchFeaturesByProjection(std::shared_ptr<Frame> frame, const SE3 &initPose, int th)
{
    features_matched_.clear();
    fea_matIndex.clear();
    features_matched_ = vector<cv::Point2f>(frame->features_.size(), Point2f(0, 0));
    fea_matIndex = vector<int>(frame->features_.size(),-1);
    num_matched_ = 0;
   
    if(frame->id_ == 0)
        return false;

    int N = frame->keypoints_l_.size();
    cout << "keypoints_l_ size: " << N << endl;
    cout << "features size: " << frame->features_.size() << endl;

    // 判断是前进还是后退
    bool isForward, isBackward;
    SE3 Twc_f;
    math::judgeForOrBackward(frame->Twc_, Twc_f, isForward, isBackward);

    // 将Frame中的点投影到当前帧
    int fea_count = 0;
    vector<cv::Point2f> prjPos(N, cv::Point2f(0, 0));
    vector<float> invzc(N, 0);
    for (int i = 0; i < frame->features_.size(); i++)
    {
        // 如果上一帧的特征点匹配到了地图点，就将该地图点投影到当前帧
        // 否则就用上一帧双目三角化出的坐标
        Vec3d lwld, x3Dc;
        MapPoint::Ptr mp = frame->mpoints_matched_[i];
        if (mp != nullptr)
            lwld = mp->coor_;
        else if (frame->features_[i]->wldcoor_ != Vec3d(0, 0, 0))
            lwld = frame->features_[i]->wldcoor_;
        else
            continue;
        x3Dc = Transform::world2camera(lwld, T_cw_f_);
        Vec2d pixeluv = Transform::camera2pixel(Camera::K_, x3Dc);
        float u = pixeluv(0);
        float v = pixeluv(1);
        if (!Func::inBorder(u, v))
            continue;

        fea_count++;
        invzc[i] = 1.0 / x3Dc(2);
        prjPos[i] = cv::Point2f(u, v);
    }

    /*与Frame帧特征匹配*/
    vector<cv::Point2f> fea_mat(N, Point2f(0, 0));
    vector<int> index_pre(frame->features_.size(), -1); // frame帧第i个特征对应当前帧第几个
    vector<int> fea_dist(features_.size(), 256);        // 当前帧第i个特征的最小匹配距离
    for (int idex = 0; idex < 2; idex++)
    {
        for (int i = 0; i < N; i++)
        {
            if (fea_mat[i] != Point2f(0, 0) || prjPos[i] == Point2f(0, 0))
                continue;

            int nLastOctave = frame->keypoints_l_[i].octave;

            // Search in a window. Size depends on scale
            float radius = 0;
            if (idex == 0)
                radius = th * orbleft_->mvScaleFactor[nLastOctave];
            else
                radius = 2 * th * orbleft_->mvScaleFactor[nLastOctave];
            vector<size_t> vIndices2;

            // 根据前进还是后退在不同尺度上搜索特征点
            float u = prjPos[i].x, v = prjPos[i].y;
            if (isForward)
            {
                vIndices2 = ORBextractor::GetFeaturesInArea(orbleft_, keypoints_l_, u, v, radius, nLastOctave);
            }
            else if (isBackward)
            {
                vIndices2 = ORBextractor::GetFeaturesInArea(orbleft_, keypoints_l_, u, v, radius, 0, nLastOctave);
            }
            else
            {
                vIndices2 = ORBextractor::GetFeaturesInArea(orbleft_, keypoints_l_, u, v, radius, nLastOctave - 1, nLastOctave + 1);
            }

            if (vIndices2.empty())
                continue;

            const cv::Mat dMP = frame->descriptors_l_.row(i);

            int bestDist = 256;
            int bestIdx2 = -1;

            // 遍历满足条件的特征点
            for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end();
                vit != vend; vit++)
            {
                const size_t i2 = *vit;

                if (features_[i2]->type_)
                {
                    const float ur = u - Camera::base_fx_ * invzc[i];
                    const float er = fabs(ur - features_[i2]->x_r_);
                    if (er > radius)
                        continue;
                }

                const cv::Mat &d = descriptors_l_.row(i2);

                const int dist = ORBextractor::DescriptorDistance(dMP, d);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx2 = i2;
                }
            }

            if (bestDist <= std::min(ORBextractor::TH_HIGH, fea_dist[bestIdx2]))
            {
                fea_mat[i] = keypoints_l_[bestIdx2].pt;
                index_pre[i] = bestIdx2;
                fea_dist[bestIdx2] = bestDist;
//                    cout << features_[bestIdx2]->stereo_ << " ";
            }
        }
//            cout << endl;

        // 旋转一致性检验
        checkRotConsistency(frame, fea_mat, index_pre);

        // 统计匹配个数
        num_matched_ = 0;
        for (int i = 0; i < fea_mat.size(); i++)
        {
            if (fea_mat[i] != Point2f(0, 0))
                num_matched_++;
        }

        if (num_matched_ >= (float)0.5 * fea_count)
            break;
    }
    cout << "重投影匹配个数： " << num_matched_ << endl;

    features_matched_ = fea_mat;
    fea_matIndex = index_pre;

    // 如果匹配数量过少，则返回false
    if (num_matched_ < 0.5 * fea_count && num_matched_ < 15)
        return false;
    else
        return true;
}