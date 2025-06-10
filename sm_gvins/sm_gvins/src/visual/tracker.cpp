#include <glog/logging.h>
#include <unordered_map>

#include "tracker.h"
#include "timer.h"
#include "math.h"

using namespace std;

Tracker::Tracker(MapPtr map)
    : map_(std::move(map))
{
    camera_left_  = std::make_shared<Camera>(
        Parameters::fx_, Parameters::fy_, Parameters::cx_, 
        Parameters::cy_, Parameters::base_, SE3()
    );

    camera_right_  = std::make_shared<Camera>(
        Parameters::fx_, Parameters::fy_, Parameters::cx_, 
        Parameters::cy_, Parameters::base_, Parameters::Tc0c1_.inverse()
    );
}

void Tracker::SetCameras(Camera::Ptr camera_left, Camera::Ptr camera_right){
    camera_left_  = std::move(camera_left);
    camera_right_ = std::move(camera_right);
}

bool Tracker::TrackFrame(FramePtr frame){    
    curr_frame_ = frame;
    ProcessCurrentFrame();

    // 初始化匹配地图点
    if(initilize_flag_){
        BuildInitMap();

        initilize_flag_ = false;
        last_frame_ = curr_frame_;
        return true;
    }

    curr_frame_->Twc_ = last_frame_->Twc_ * relative_motion_;
    bool track_success = MatchWithLastframe();
    if(track_success){
        num_lost_ = 0;
        curr_frame_->is_good_ = true;
    }else{
        num_lost_++;
        curr_frame_->is_good_ = false;
        LOG(INFO) << "当前帧匹配失败！";
    }

    if(num_lost_ > MAX_LOST){
        LOG(ERROR) << "里程计跟踪失败！";
        return false;
    }

    TriangulateNewPoints();

    last_frame_ = curr_frame_;
    return true;
}

void Tracker::ProcessCurrentFrame(){
    if (curr_frame_->left_img_.channels() == 3)
    {
        cvtColor(curr_frame_->left_img_, curr_frame_->left_img_, cv::COLOR_RGB2GRAY);
        cvtColor(curr_frame_->right_img_, curr_frame_->right_img_, cv::COLOR_RGB2GRAY);
    }

    curr_frame_->ExtractKeyPointsAndDescriptors();
    curr_frame_->MatchFromeStereo();
    curr_frame_->UndistKeyPoints();
    curr_frame_->CreateFeatures();
    // curr_frame_->calcFeaturesCamCoors();
}

bool Tracker::MatchWithLastframe(){
    features_matched_.clear();

    bool match_success = MatchFeatures(last_frame_, 7);
    if(!match_success){
        return false;
    }

    int i = 0;
    vector<cv::Point3d> points_3d;
    vector<cv::Point2d> pixels_2d;
    for (auto pt : features_matched_) {
        if (pt != cv::Point2f(0, 0) && last_frame_->features_[i]->map_point_.lock()){
            Vec3d last_3d = last_frame_->features_[i]->map_point_.lock()->pos_;
            points_3d.push_back(cv::Point3d(double(last_3d(0, 0)), double(last_3d(1, 0)), double(last_3d(2, 0))));
            pixels_2d.push_back(cv::Point2d(double(pt.x), double(pt.y)));
        } 
        i++;
    }

    if(!CalcPoseByPnP(points_3d, pixels_2d)){
        return false;
    }
    
    if(curr_frame_->id_ != 1){
        relative_motion_ = last_frame_->Twc_.inverse() * curr_frame_->Twc_;

        double T_reltv = relative_motion_.log().norm();         
        double R_reltv = relative_motion_.so3().log().norm();   
        double t_reltv = relative_motion_.translation().norm(); 

        if (isnan(t_reltv) || isnan(R_reltv))
        {
            cout << "位姿估计结果有误" << endl;
            return false;
        }

        if (t_reltv > normalpose_max_t_ || R_reltv > normalpose_max_R_)
        {
            cout << "位移差：" << t_reltv << " 旋转差：" << R_reltv << " 位姿估计结果有误" << endl;
            return false;

        }
    }

    return true;
}

bool Tracker::MatchWithReferenceframe()
{
   /*  curr_frame_->features_matched_Key.clear();
    curr_frame_->fea_matIndex_Key.clear();
    curr_frame_->features_matched_Key = vector<cv::Point2f>(curr_frame_->features_.size(), cv::Point2f(0, 0));
    curr_frame_->fea_matIndex_Key = vector<int>(curr_frame_->features_.size(),-1);

    MatchFeatures(ref_frame_, 7);
    if(curr_frame_->num_matched_ < 15){
        LOG(INFO) << "与关键帧匹配的特征点数量过少: " << curr_frame_->num_matched_Key << ",无法计算位姿";
        return false;
    }
    LOG(INFO) << "关键帧id: " << ref_frame_->id_ << " 当前帧与关键帧匹配上的特征点的数量： " << curr_frame_->num_matched_;

    curr_frame_->features_matched_Key = curr_frame_->features_matched_;
    curr_frame_->fea_matIndex_Key     = curr_frame_->fea_matIndex;
    curr_frame_->num_matched_Key      = curr_frame_->num_matched_;
    
    int i = 0;
    vector<cv::Point3d> points_3d;
    vector<cv::Point2d> pixels_2d;
    for (auto pt : curr_frame_->features_matched_Key)
    {
        if (pt != cv::Point2f(0, 0))
        {
            Vec3d ref_3d = ref_frame_->features_[i]->camcoor_;
            points_3d.push_back(cv::Point3d(double(ref_3d(0, 0)), double(ref_3d(1, 0)), double(ref_3d(2, 0))));
            pixels_2d.push_back(cv::Point2d(double(pt.x), double(pt.y)));
        }
        i++;
    }

    return true; */
}

bool Tracker::MatchFeatures(FramePtr frame, int th){
    /* if (MatchFeaturesByProjection(frame, th))
    {
        return true;
    }
    else  */if (MatchFeaturesByBruteForce(frame, th))
    {
        return true;
    }
    return false;
}

bool Tracker::MatchFeaturesByProjection(FramePtr frame, int th)
{
   /*  
    int N = frame->keypoints_l_.size();

    // 判断是前进还是后退
    bool isForward, isBackward;
    math::judgeForOrBackward(frame->Twc_, curr_frame_->Twc_, isForward, isBackward);

    // 将Frame中的点投影到当前帧
    int fea_count = 0;
    vector<cv::Point2f> prjPos(N, cv::Point2f(0, 0));
    vector<float> invzc(N, 0);
    for (int i = 0; i < frame->features_.size(); i++)
    {
        if(!frame->features_[i] || !frame->features_[i]->map_point_.lock())  continue;

        // 如果上一帧的特征点匹配到了地图点，就将该地图点投影到当前帧
        // 否则就用上一帧双目三角化出的坐标
        auto mp = frame->features_[i]->map_point_.lock();
        Vec3d lwld = mp->pos_;
        Vec3d x3Dc = camera_left_->world2camera(lwld, curr_frame_->Twc_);
        Vec2d pixeluv = camera_left_->world2pixel(lwld, curr_frame_->Twc_);

        float u = pixeluv(0);
        float v = pixeluv(1);
        if (!InBorder(u, v))
            continue;

        fea_count++;
        invzc[i] = 1.0 / x3Dc(2);
        prjPos[i] = cv::Point2f(u, v);
    }

    int num_matched = 0;
    vector<cv::Point2f> fea_mat(N, cv::Point2f(0, 0));
    vector<int> index_pre(frame->features_.size(), -1); // frame帧第i个特征对应当前帧第几个
    vector<int> fea_dist(curr_frame_->features_.size(), 256);        // 当前帧第i个特征的最小匹配距离
    for (int idex = 0; idex < 2; idex++)
    {
        for (int i = 0; i < N; i++)
        {
            if (fea_mat[i] != cv::Point2f(0, 0) || prjPos[i] == cv::Point2f(0, 0))
                continue;

            int nLastOctave = frame->keypoints_l_[i].octave;

            // Search in a window. Size depends on scale
            float radius = 0;
            if (idex == 0)
                radius = th * curr_frame_->orbleft_->mvScaleFactor[nLastOctave];
            else
                radius = 2 * th * curr_frame_->orbleft_->mvScaleFactor[nLastOctave];
            vector<size_t> vIndices2;

            // 根据前进还是后退在不同尺度上搜索特征点
            float u = prjPos[i].x, v = prjPos[i].y;
            if (isForward)
            {
                vIndices2 = ORBextractor::GetFeaturesInArea(curr_frame_->orbleft_, curr_frame_->keypoints_l_, u, v, radius, nLastOctave);
            }
            else if (isBackward)
            {
                vIndices2 = ORBextractor::GetFeaturesInArea(curr_frame_->orbleft_, curr_frame_->keypoints_l_, u, v, radius, 0, nLastOctave);
            }
            else
            {
                vIndices2 = ORBextractor::GetFeaturesInArea(curr_frame_->orbleft_, curr_frame_->keypoints_l_, u, v, radius, nLastOctave - 1, nLastOctave + 1);
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

                if (curr_frame_->features_[i2] != nullptr)
                {
                    const float ur = u - Parameters::base_fx_ * invzc[i];
                    const float er = fabs(ur - curr_frame_->features_[i2]->x_r_);
                    if (er > radius)
                        continue;
                    
                    const cv::Mat &d = curr_frame_->descriptors_l_.row(i2);

                    const int dist = ORBextractor::DescriptorDistance(dMP, d);
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx2 = i2;
                    }
                }
            }

            if (bestDist <= std::min(ORBextractor::TH_HIGH, fea_dist[bestIdx2]))
            {
                fea_mat[i] = curr_frame_->keypoints_l_[bestIdx2].pt;
                index_pre[i] = bestIdx2;
                fea_dist[bestIdx2] = bestDist;
            }
        }

        // 旋转一致性检验
        CheckRotConsistency(frame, fea_mat, index_pre);

        // 统计匹配个数
        for (int i = 0; i < fea_mat.size(); i++)
        {
            if (fea_mat[i] != cv::Point2f(0, 0))
                num_matched++;
        }

        if (num_matched >= (float)0.5 * fea_count)
            break;
    }
    LOG(INFO) << "重投影匹配个数： " << num_matched;

    features_matched_ = fea_mat;

    // 如果匹配数量过少，则返回false
    if (num_matched < 0.5 * fea_count && num_matched < 15)
        return false;
    else
        return true;

 */

    
    int N = frame->keypoints_l_.size();
    vector<int> rotHist[ORBextractor::HISTO_LENGTH];
    for (int i = 0; i < ORBextractor::HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    
    // 判断是前进还是后退
    SE3 T_l_c = frame->Twc_.inverse() * curr_frame_->Twc_;
    Vec3d tlc = T_l_c.translation();

    const bool bForward = tlc(2)   > Parameters::base_;   // 如果大于基线则为前进
    const bool bBackward = -tlc(2) > Parameters::base_; // 如果小于基线则为后退

    // 与上一帧特征匹配
    int num_matched = 0;
    const float factor = ORBextractor::HISTO_LENGTH / 360.0f;
    vector<cv::Point2f> fea_mat(N, cv::Point2f(0, 0));
    vector<int> fea_order(N, -1);
    for (int idex = 0; idex < 2; idex++)
    {
        for (int i = 0; i < N; i++)
        {
            if (!frame->features_[i])
            {
                continue;
            }

            // 如果上一帧的特征点匹配到了地图点，就将该地图点投影到当前帧
            // 否则就用上一帧双目三角化出的坐标
            Vec3d lwld, x3Dc;
            MapPointPtr mp = frame->features_[i]->map_point_.lock();
            if (mp != nullptr)
            {
                lwld = mp->pos_;
            }
            else
                continue;
            
            x3Dc = camera_left_->world2camera(lwld, curr_frame_->Twc_);

            const double invzc = 1.0 / x3Dc(2);
            if (invzc < 0)
                continue;
            
            Vec2d pixeluv = camera_left_->camera2pixel(x3Dc);
            float u = pixeluv(0);
            float v = pixeluv(1);
            if (u < Parameters::min_X_ || u > Parameters::max_X_)
                continue;
            if (v < Parameters::min_Y_ || v > Parameters::max_Y_)
                continue;

            int nLastOctave = frame->keypoints_l_[i].octave;

            // Search in a window. Size depends on scale
            float radius = 0;
            if (idex == 0)
                radius = th * curr_frame_->orbleft_->mvScaleFactor[nLastOctave];
            else
                radius = 2 * th * curr_frame_->orbleft_->mvScaleFactor[nLastOctave];
            vector<size_t> vIndices2;

            // 根据前进还是后退在不同尺度上搜索特征点
            // NOTE 尺度越大,图像越小
            if (bForward)
            {
                vIndices2 = ORBextractor::GetFeaturesInArea(curr_frame_->orbleft_, curr_frame_->keypoints_l_, u, v, radius, nLastOctave);
            }
            else if (bBackward)
            {
                vIndices2 = ORBextractor::GetFeaturesInArea(curr_frame_->orbleft_, curr_frame_->keypoints_l_, u, v, radius, 0, nLastOctave);
            }
            else
            {
                vIndices2 = ORBextractor::GetFeaturesInArea(curr_frame_->orbleft_, curr_frame_->keypoints_l_, u, v, radius, nLastOctave - 1, nLastOctave + 1);
            }
//             vIndices2 = ORBextractor::GetFeaturesInArea(orbleft_, keypoints_l_, u, v, radius);

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
                // if (!features_[i2]->type_)
                //     continue;
                if (i2 >= curr_frame_->descriptors_l_.rows)
                    continue;
                if (curr_frame_->features_[i2])
                {
                    const float ur = u - Parameters::base_fx_ * invzc;
                    const float er = fabs(ur - curr_frame_->features_[i2]->x_r_);
                    if (er > radius)
                        continue;
                }

                const cv::Mat &d = curr_frame_->descriptors_l_.row(i2);

                const int dist = ORBextractor::DescriptorDistance(dMP, d);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx2 = i2;
                }
            }

            // 判断匹配点的相对旋转量和平均旋转量的差异，若较大则为错误匹配
            if (bestDist <= ORBextractor::TH_HIGH)
            {
                // fea_mat.push_back(keypoints_l_[bestIdx2].pt);
                fea_mat[i] = curr_frame_->keypoints_l_[bestIdx2].pt;
                fea_order[i] = bestIdx2;
                num_matched++;
                float rot = frame->keypoints_l_[i].angle - curr_frame_->keypoints_l_[bestIdx2].angle;
                if (rot < 0.0)
                    rot += 360.0f;
                int bin = round(rot * factor); //返回四舍五入的整数值
                if (bin == ORBextractor::HISTO_LENGTH)
                    bin = 0;
                assert(bin >= 0 && bin < ORBextractor::HISTO_LENGTH);
                rotHist[bin].push_back(bestIdx2);
            }
        }
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;
        ORBextractor::ComputeThreeMaxima(rotHist, ORBextractor::HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < ORBextractor::HISTO_LENGTH; i++)
        {
            if (i != ind1 && i != ind2 && i != ind3)
            {
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    for (int c = 0; c < N; c++)
                    {
                        if (fea_mat[c] == curr_frame_->keypoints_l_[rotHist[i][j]].pt)
                        {
                            fea_mat[c] = cv::Point2f(0, 0);
                            fea_order[c] = -1;
                            num_matched--;
                        }
                    }
                }
            }
        }
        if (num_matched >= 0.25 * ORBextractor::nfeatures)
            break;
    }

    features_matched_ = fea_mat;
    LOG(INFO) << "投影匹配数量：" << num_matched;

    // 如果匹配数量过少，则返回false
    if (num_matched < 15)
        return false;
    else
        return true;
    
}

bool Tracker::MatchFeaturesByBruteForce(FramePtr frame, int th)
{
    /*特征点匹配*/
    int num_matched = 0;
    vector<cv::DMatch> matches;
    vector<vector<cv::DMatch>> knnMatches;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming(2)"); // BF匹配
    // 取出frame中可以匹配的特征
    cv::Mat descriptors_1;
    for (int i = 0; i < frame->features_.size(); i++)
    {
        if (!frame->features_[i])
            continue;
        
        descriptors_1.push_back(frame->features_[i]->descriptor_);
    }

    if (!descriptors_1.rows)
    {
        return false;
    }

    // 匹配
    cv::Mat descriptors_2 = curr_frame_->descriptors_l_;
    matcher->match(descriptors_1, descriptors_2, matches); ////mask
    // matcher->knnMatch(descriptors_1, descriptors_2, knnMatches, 2); // knn匹配

    // 计算最小距离
    float min_dis = std::min_element(
                        matches.begin(), matches.end(), [](const cv::DMatch &m1, const cv::DMatch &m2) { return m1.distance < m2.distance; })
                        ->distance;

    // 匹配结果进一步筛选
    int j = 0;
    int count = 0;
    features_matched_.resize(frame->features_.size());
    for (int i = 0; i < frame->features_.size(); i++)
    {
        if (!frame->features_[i] || features_matched_[i] != cv::Point2f(0, 0))
            continue;
        else
        {
            cv::DMatch &m = matches[j];
            int queryIdx = m.queryIdx; // frame帧
            int trainIdx = m.trainIdx; // 当前帧
            if (m.distance < max<float>(min_dis * 2, 30.0))
            {
                count++;
                num_matched++;
                features_matched_[i] = curr_frame_->keypoints_l_[trainIdx].pt;
            }
            j++;
        }
    }

    LOG(INFO) << "暴力匹配数量：" << count;
    // 如果匹配数量过少，则返回false
    if (num_matched < 0.05 * ORBextractor::nfeatures){
        LOG(ERROR) << "暴力匹配数量太少：" << count;
        return false;
    }
    
    return true;
}

void Tracker::CheckRotConsistency(FramePtr frame, vector<cv::Point2f> &fea_mat, vector<int> &index)
{
    const float factor = ORBextractor::HISTO_LENGTH / 360.0f;
    vector<int> rotHist[ORBextractor::HISTO_LENGTH]; // 旋转直方图（检查旋转一致性）
    for (int i = 0; i < ORBextractor::HISTO_LENGTH; i++)
        rotHist[i].reserve(500);

    // 计算旋转直方图
    for (int i = 0; i < frame->features_.size(); i++)
    {
        int cur_idx = index[i];
        if (cur_idx == -1)
            continue;
        float rot = curr_frame_->keypoints_l_[cur_idx].angle - frame->keypoints_l_[i].angle;
        if (rot < 0.0)
            rot += 360.0f;
        int bin = std::round(rot * factor);
        if (bin == ORBextractor::HISTO_LENGTH)
            bin = 0;
        assert(bin >= 0 && bin < ORBextractor::HISTO_LENGTH);
        rotHist[bin].push_back(i);
    }

    // 取出直方图中值最大的三个index
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;
    ORBextractor::ComputeThreeMaxima(rotHist, ORBextractor::HISTO_LENGTH, ind1, ind2, ind3);

    // 剔除直方图中不是三个index的匹配
    for (int i = 0; i < ORBextractor::HISTO_LENGTH; i++)
    {
        if (i != ind1 && i != ind2 && i != ind3)
        {
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                int pre_idx = rotHist[i][j];
                fea_mat[pre_idx] = cv::Point2f(0, 0);
                index[pre_idx] = -1;
            }
        }
    }
}

bool Tracker::CalcPoseByPnP(const vector<cv::Point3d>& points_3d, const vector<cv::Point2d>& pixels_2d)
{
    vector<int> inliers;
    // 计算位姿
    cv::Mat K, r, t, R;
    Mat3d R_eg;
    Vec3d t_eg;
    //rvec - 输出的旋转向量。使坐标点从世界坐标系旋转到相机坐标系
    //tvec - 输出的平移向量。使坐标点从世界坐标系平移到相机坐标系
    if (!cv::solvePnPRansac(points_3d, pixels_2d, camera_left_->cvK(), cv::Mat(), r, t, false, 100, 4.0, 0.98999, inliers/* , cv::SOLVEPNP_EPNP */))
    {
        LOG(INFO) << "特征匹配数量过少，无法计算位姿";
        return false;
    }
    // LOG(INFO) << "\nr:\n" << r;
    // LOG(INFO) << "\nt:\n" << t;
    cv::Rodrigues(r, R);
    for (int i = 0; i < 3; ++i) {
        t_eg(i) = t.at<double>(i);
        for (int j = 0; j < 3; ++j)
            R_eg(i, j) = R.at<double>(i, j);
    }
    // std::cout << "\nR_eg:\n" << R_eg << std::endl;
    // std::cout << "\nt_eg:\n" << t_eg << std::endl;
    
    if(inliers.size() < 5)
    {
        LOG(ERROR) << "RANSAC 内点数量太少：" << inliers.size();
        return false;
    }

    curr_frame_->Twc_ = last_frame_->Twc_ * SE3(R_eg, t_eg);
    
    // 剔除误匹配
    int j = 0, k = 0;
    for (int i = 0; i < features_matched_.size(); i++)
    {
        if (features_matched_[i] != cv::Point2f(0, 0))
        {
            if (j >= inliers.size() || k != inliers[j])
            {
                features_matched_[i] = cv::Point2f(0, 0);
            }
            else
                j++;
            k++;
        }
    }
    
    LOG(INFO) << "RANSAC 内点数量：" << inliers.size();
    return true;
}

bool Tracker::BuildInitMap(){
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    size_t cnt_init_landmarks = 0;
    for (size_t i = 0; i < curr_frame_->features_.size(); ++i) {
        if (curr_frame_->features_[i] == nullptr) continue;
        // create map point from triangulation
        std::vector<Vec3d> points{
            camera_left_->pixel2camera(
                Vec2d(curr_frame_->features_[i]->pixel_pt_.pt.x,
                     curr_frame_->features_[i]->pixel_pt_.pt.y)),
            camera_right_->pixel2camera(
                Vec2d(curr_frame_->features_[i]->pixel_pt_right_.pt.x,
                     curr_frame_->features_[i]->pixel_pt_right_.pt.y))};
        Vec3d pworld = Vec3d::Zero();

        if (math::triangulatePoint(poses, points, pworld) && pworld[2] > 0) {
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->pos_ = pworld;
            new_map_point->AddObservation(curr_frame_->features_[i]);
            new_map_point->AddObservation(curr_frame_->features_[i]);
            curr_frame_->features_[i]->map_point_ = new_map_point;
            LOG(INFO) << "pworld: " << pworld.transpose();
            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }
    curr_frame_->SetKeyframe();
    map_->InsertKeyFrame(curr_frame_);

    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Tracker::TriangulateNewPoints(){
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    SE3 current_pose_Tcw = curr_frame_->Twc_.inverse();
    int cnt_triangulated_pts = 0;
    for (size_t i = 0; i < curr_frame_->features_.size(); ++i) {
        if (curr_frame_->features_[i] != nullptr && 
            curr_frame_->features_[i]->map_point_.expired()) {
            // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
            std::vector<Vec3d> points{
                camera_left_->pixel2camera(
                    Vec2d(curr_frame_->features_[i]->pixel_pt_.pt.x,
                         curr_frame_->features_[i]->pixel_pt_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2d(curr_frame_->features_[i]->pixel_pt_right_.pt.x,
                         curr_frame_->features_[i]->pixel_pt_right_.pt.y))};
            Vec3d pworld = Vec3d::Zero();

            if (math::triangulatePoint(poses, points, pworld) && pworld[2] > 0) {
                auto new_map_point = MapPoint::CreateNewMappoint();
                // pworld = current_pose_Tcw * pworld;
                new_map_point->pos_ = pworld;
                new_map_point->AddObservation(
                    curr_frame_->features_[i]);
                new_map_point->AddObservation(
                    curr_frame_->features_[i]);

                curr_frame_->features_[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
}
