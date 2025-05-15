#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>
#include <unordered_map>
#include "g2o_types.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include "tracker.h"
#include "timer.h"
#include "math.h"

Tracker::Tracker(const std::string& config_file){
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    cv::Mat intrinsic_mat0, intrinsic_mat1, distortion_mat0, distortion_mat1;
    fsSettings["cam0"]["intrinsic"] >> intrinsic_mat0;
    fsSettings["cam0"]["distortion"] >> distortion_mat0;
    fsSettings["cam1"]["intrinsic"] >> intrinsic_mat1;
    fsSettings["cam1"]["distortion"] >> distortion_mat1;

    LOG(INFO) << "\nIntrinsic Matrix:\n" << intrinsic_mat0;
    LOG(INFO) << "\nDistortion Coefficients:\n" << distortion_mat0;
    LOG(INFO) << "\nIntrinsic Matrix:\n" << intrinsic_mat1;
    LOG(INFO) << "\nDistortion Coefficients:\n" << distortion_mat1;

    cameras_.resize(2);
    cameras_[0] = Camera::createCamera(intrinsic_mat0, distortion_mat0, cv::Size(1241, 376));
    cameras_[1] = Camera::createCamera(intrinsic_mat1, distortion_mat1, cv::Size(1241, 376));

}


bool Tracker::TrackFrame(const Image& image){    
    curr_time_ = image.timestamp_;
    curr_img_ = image.img_;
    row_ = curr_img_.rows;
    col_ = curr_img_.cols;

    curr_pts_.clear();

    if(prev_pts_.size() > 0){
        TrackByLK();
    }

    DetectNewFeatures();
    
    if(!image.img_right_.empty()){
        TrackByStereoLK(image.img_right_);
    }
    
    DrawTrack(curr_img_, image.img_right_, ids_, curr_pts_, curr_right_pts_, prev_left_pts_map_);

    bool is_keyframe = isKeyframe();
    if(!is_keyframe){
        return false;
    }

    FramePtr new_frame = CreateNewKeyFrame(image.timestamp_);

    EstimateCurrentPose(new_frame);

    map_->AddKeyFrame(new_frame);

    EstimateStateInWindow();

    prev_img_ = curr_img_;
    prev_pts_ = curr_pts_;
    prev_un_pts_ = curr_un_pts_;
    prev_un_pts_map_ = curr_un_pts_map_;
    prev_time_ = curr_time_;
    prev_left_pts_map_.clear();
    for(size_t i = 0; i < curr_pts_.size(); i++)
        prev_left_pts_map_[ids_[i]] = curr_pts_[i];

    return true;
}

void Tracker::EstimateCurrentPose(FramePtr current_frame_) {
    // pnp

    /* 
    // setup g2o
    using BlockSolverType = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);


    // auto current_frame_ = map_->BackFrame();
    // vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->pose());
    optimizer.addVertex(vertex_pose);

    Mat3d K;
    K << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1;
    
    // std:: cout << K << std::endl;
    // std:: cout << cameras_[0]->cameraMatrix() << std::endl;

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<FeaturePtr> features;
    for(auto& [id, feature] : current_frame_->features()){
        if(map_->landmarks_.find(id) != map_->landmarks_.end()){
            features.push_back(feature);
            EdgeProjectionPoseOnly *edge =
                new EdgeProjectionPoseOnly(map_->PointInId(id), K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(feature->pixel_pt_);
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }
        
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    
    // 示例：输出每条边的误差
    for (g2o::HyperGraph::Edge* e : optimizer.edges()) {
        g2o::OptimizableGraph::Edge* edge = dynamic_cast<g2o::OptimizableGraph::Edge*>(e);
        if (edge) {
            edge->computeError();  // 现在可以调用了
            std::cout << "Edge error: " << edge->chi2() << std::endl;
        } else {
            std::cerr << "Warning: Edge is not an OptimizableGraph::Edge!" << std::endl;
        }
    }

    current_frame_->setPose(vertex_pose->estimate());
 */
}

void Tracker::EstimateStateInWindow(){
    using BlockSolverType = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    std::vector<VertexPose*> vertices;
    int index = 0;
    for(auto frame : map_->keyframes_){
        VertexPose* vertex_pose = new VertexPose();
        vertex_pose->setId(index++);
        vertex_pose->setEstimate(frame->pose());
        if(frame->pose().rotationMatrix().hasNaN()){
            LOG(INFO) << frame->pose().rotationMatrix();
        }
        optimizer.addVertex(vertex_pose);
        vertices.push_back(vertex_pose);
    }

    Mat3d K;
    K << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1;
    int index_edge = 1;
    std::unordered_map<unsigned long long, g2o::VertexPointXYZ*> vertices_landmarks;
    for(auto &[id, pt3d] : map_->landmarks_){
        g2o::VertexPointXYZ* vertex_pt3d = new g2o::VertexPointXYZ();
        vertex_pt3d->setId(index++);
        vertex_pt3d->setEstimate(pt3d);
        vertex_pt3d->setMarginalized(true);
        vertices_landmarks[id] = vertex_pt3d;
        optimizer.addVertex(vertex_pt3d);
        
        for(size_t i = 0; i < map_->keyframes_.size(); ++i){
            auto frame = map_->keyframes_[i];
            if(frame->IsFeatureInFrame(id)){
                EdgeProjectionVO* edge = new EdgeProjectionVO(K);
                edge->setId(index_edge++);
                edge->setVertex(0, vertices[i]);
                edge->setVertex(1, vertex_pt3d);
                edge->setMeasurement(frame->features()[id]->pixel_pt_);
                auto rk = new g2o::RobustKernelHuber();
                rk->setDelta(5.991);
                edge->setRobustKernel(rk);
                optimizer.addEdge(edge);
            }
        }
    }

    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    for(size_t i = 0; i < vertices.size(); ++i){
        map_->keyframes_[i]->setPose(vertices[i]->estimate());
    }

    for(auto &[id, pt3d] : map_->landmarks_){
        pt3d = vertices_landmarks[id]->estimate();
        for(auto frame : map_->keyframes_){
            if(frame->IsFeatureInFrame(id)){
                frame->features()[id]->world_pt_ = pt3d;
            }
        }
    }
}

bool Tracker::StereoInit(const Image& image){
    curr_time_ = image.timestamp_;
    curr_img_ = image.img_;
    row_ = curr_img_.rows;
    col_ = curr_img_.cols;

    DetectNewFeatures();
    
    if(!image.img_right_.empty()){
        TrackByStereoLK(image.img_right_);
    }
    
    DrawTrack(curr_img_, image.img_right_, ids_, curr_pts_, curr_right_pts_, prev_left_pts_map_);

    FramePtr new_frame = CreateNewKeyFrame(image.timestamp_);

    map_->AddKeyFrame(new_frame);

    prev_img_ = curr_img_;
    prev_pts_ = curr_pts_;
    prev_un_pts_ = curr_un_pts_;
    prev_un_pts_map_ = curr_un_pts_map_;
    prev_time_ = curr_time_;
    prev_left_pts_map_.clear();
    for(size_t i = 0; i < curr_pts_.size(); i++)
        prev_left_pts_map_[ids_[i]] = curr_pts_[i];

    return true;
}

void Tracker::DetectNewFeatures(){
    SetMask();

    Timer timer;
    int n_max_cnt = MAX_CNT - static_cast<int>(curr_pts_.size());
    if (n_max_cnt > 0)
    {
        if(mask_.empty())
            LOG(INFO) << "mask is empty ";
        if (mask_.type() != CV_8UC1)
            LOG(INFO) << "mask type wrong ";

        cv::goodFeaturesToTrack(curr_img_, n_pts_, MAX_CNT - curr_pts_.size(), 0.01, MIN_DIST, mask_);
    }
    else
        n_pts_.clear();
    LOG(INFO) << "detect feature costs: " << timer.toc() << " ms";

    for (auto &p : n_pts_)
    {
        curr_pts_.push_back(p);
        ids_.push_back(n_id++);
        track_cnt_.push_back(1);
    }
    // std::cout << "feature cnt after add: " << (int)ids_.size() << std::endl;
    
    curr_un_pts_ = PixelToCamera(curr_pts_, cameras_[0]);
    pts_velocity_ = ptsVelocity(ids_, curr_un_pts_, curr_un_pts_map_, prev_un_pts_map_);
}

FramePtr Tracker::CreateNewKeyFrame(double stamp){
    FramePtr new_frame = Frame::createFrame(stamp);
    for (size_t i = 0; i < curr_pts_.size(); i++) {
        int feature_id = ids_[i];
        auto feature = Feature::createFeature(feature_id);
        double x, y ,z;
        x = curr_un_pts_[i].x;
        y = curr_un_pts_[i].y;
        z = 1;
        feature->normlize_pt_ << x, y, z;

        x = curr_un_right_pts_[i].x;
        y = curr_un_right_pts_[i].y;
        z = 1;
        feature->normlize_pt_right_ << x, y, z;

        double p_u, p_v;
        p_u = curr_pts_[i].x;
        p_v = curr_pts_[i].y;
        feature->pixel_pt_ << p_u, p_v;

        p_u = curr_right_pts_[i].x;
        p_v = curr_right_pts_[i].y;
        feature->pixel_pt_right_ << p_u, p_v;

        double velocity_x, velocity_y;
        velocity_x = pts_velocity_[i].x;
        velocity_y = pts_velocity_[i].y;
        feature->vel_ << velocity_x, velocity_y;

        velocity_x = right_pts_velocity_[i].x;
        velocity_y = right_pts_velocity_[i].y;
        feature->vel_right_ << velocity_x, velocity_y;
        
        new_frame->addFeature(feature_id, feature);
    }

    // map_->AddKeyFrame(new_frame);
    return new_frame;
}


bool Tracker::isKeyframe() {
    bool is_keyframe = false;

    // 检查特征点数量
    if (curr_pts_.size() < MIN_FEATURES) {
        is_keyframe = true;
        LOG(INFO) << "选择为关键帧：特征点不足 (" << curr_pts_.size() << " < " << MIN_FEATURES << ")";
    }

    // 如果不是关键帧，计算平均视差
    if (!is_keyframe && !prev_left_pts_map_.empty()) {
        double total_parallax = 0.0;
        int valid_pairs = 0;
        for (size_t i = 0; i < curr_pts_.size(); ++i) {
            auto it = prev_left_pts_map_.find(ids_[i]);
            if (it != prev_left_pts_map_.end()) {
                total_parallax += distance(curr_pts_[i], it->second);
                valid_pairs++;
            }
        }
        double avg_parallax = valid_pairs > 0 ? total_parallax / valid_pairs : 0.0;
        if (avg_parallax > PARALLAX_THRESHOLD) {
            is_keyframe = true;
            LOG(INFO) << "选择为关键帧：平均视差 (" << avg_parallax << " > " << PARALLAX_THRESHOLD << ")";
        }
    }

    // 检查与上一个关键帧的时间间隔
    if (!is_keyframe && last_keyframe_time_ >= 0 && (curr_time_ - last_keyframe_time_) > MIN_TIME_GAP) {
        is_keyframe = true;
        LOG(INFO) << "选择为关键帧：时间间隔 (" << (curr_time_ - last_keyframe_time_) << " > " << MIN_TIME_GAP << ")";
    }

    // 如果是关键帧，更新 last_keyframe_time_
    if (is_keyframe) {
        last_keyframe_time_ = curr_time_;
    }

    return is_keyframe;
}

void Tracker::TrackByLK(){
    Timer lk_timer;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(prev_img_, curr_img_, prev_pts_, curr_pts_, status, err, cv::Size(21, 21), 3);

    std::vector<uchar> reverse_status;
    std::vector<cv::Point2f> reverse_pts = prev_pts_;
    cv::calcOpticalFlowPyrLK(curr_img_, prev_img_, curr_pts_, reverse_pts, reverse_status, err, cv::Size(21, 21), 1, 
    cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
    for(size_t i = 0; i < status.size(); i++)
    {
        //如果前后都能找到，并且找到的点的距离小于0.5
        if(status[i] && reverse_status[i] && distance(prev_pts_[i], reverse_pts[i]) <= 0.5)
        {
            status[i] = 1;
        }
        else
            status[i] = 0;
    }

    for (int i = 0; i < int(curr_pts_.size()); i++)
        if (status[i] && !inBorder(curr_pts_[i]))// 如果这个点不在图像内，则剔除
            status[i] = 0;
    reduceVector(prev_pts_, status);
    reduceVector(curr_pts_, status);
    reduceVector(ids_, status);
    reduceVector(track_cnt_, status);
    LOG(INFO) << "temporal optical flow costs: %fms", lk_timer.toc();
    LOG(INFO) << "track cnt: " << (int)ids_.size();

    for (auto &n : track_cnt_)
        n++;
}

void Tracker::TrackByStereoLK(const cv::Mat &img_right){
    ids_right_.clear();
    curr_right_pts_.clear();
    curr_un_right_pts_.clear();
    right_pts_velocity_.clear();

    cv::Mat rightImg = img_right;
    if(!curr_pts_.empty())
    {
        //printf("stereo image; track feature on right image\n"); //在右侧图像上追踪特征

        std::vector<cv::Point2f> reverseLeftPts;
        std::vector<uchar> status, statusRightLeft; //左右目的状态
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(curr_img_, rightImg, curr_pts_, curr_right_pts_, status, err, cv::Size(21, 21), 3);

        reverseLeftPts = curr_pts_;
        cv::calcOpticalFlowPyrLK(rightImg, curr_img_, curr_right_pts_, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
        for(size_t i = 0; i < status.size(); i++)
        {
            if(status[i] && statusRightLeft[i] && inBorder(curr_right_pts_[i]) && distance(curr_pts_[i], reverseLeftPts[i]) <= 0.5)
                status[i] = 1;
            else
                status[i] = 0;
        }
        

        ids_right_ = ids_;
        
        reduceVector(prev_pts_, status);
        reduceVector(curr_pts_, status);
        reduceVector(curr_un_pts_, status);
        reduceVector(ids_, status);
        reduceVector(track_cnt_, status);
        reduceVector(curr_right_pts_, status);
        reduceVector(ids_right_, status);
        
        curr_un_right_pts_ = PixelToCamera(curr_right_pts_, cameras_[1]);
        right_pts_velocity_ = ptsVelocity(ids_right_, curr_un_right_pts_, curr_un_right_pts_map_, prev_un_right_pts_map_);
    }
    prev_un_right_pts_map_ = curr_un_right_pts_map_;
}

void Tracker::SetMask(){
    mask_ = cv::Mat(row_, col_, CV_8UC1, cv::Scalar(255)); 

    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cnt_pts_id;
    for (unsigned int i = 0; i < curr_pts_.size(); i++)
        cnt_pts_id.push_back(std::make_pair(track_cnt_[i], std::make_pair(curr_pts_[i], ids_[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), []
        (const std::pair<int, std::pair<cv::Point2f, int>> &a, const std::pair<int, std::pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    curr_pts_.clear();
    ids_.clear();
    track_cnt_.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask_.at<uchar>(it.second.first) == 255)
        {
            curr_pts_.push_back(it.second.first);
            ids_.push_back(it.second.second);
            track_cnt_.push_back(it.first);

            cv::circle(mask_, it.second.first, MIN_DIST, 0, -1);
        }
        // cv::imshow ( "mask", mask );    
        // cv::waitKey ( 0 );              
    }
}

std::vector<cv::Point2f> Tracker::ptsVelocity(std::vector<unsigned long long> &ids, std::vector<cv::Point2f> &pts, 
                                    std::map<int, cv::Point2f> &cur_id_pts, std::map<int, cv::Point2f> &prev_id_pts)
{
    std::vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(std::make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = curr_time_ - prev_time_;
        
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < curr_pts_.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

std::vector<cv::Point2f> Tracker::PixelToCamera(std::vector<cv::Point2f> &pts, Camera::Ptr cam){
    std::vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        un_pts.push_back(cam->pixel2cam(pts[i]));
    }
    return un_pts;
}

void Tracker::DrawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               std::vector<unsigned long long> &curLeftIds,
                               std::vector<cv::Point2f> &curLeftPts, 
                               std::vector<cv::Point2f> &curRightPts,
                               std::map<int, cv::Point2f> &prevLeftPtsMap)
{
    //int rows = imLeft.rows;
    int cols = imLeft.cols;

    // ------------将两幅图像进行拼接
    if (!imRight.empty())
        cv::hconcat(imLeft, imRight, img_track_);
        // 图像凭借hconcat（B,C，A）; // 等同于A=[B  C]
    else
        img_track_ = imLeft.clone();
    cv::cvtColor(img_track_, img_track_, cv::COLOR_GRAY2BGR);
        //将imTrack转换为彩色

    // -------------在左目图像上标记特征点
    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt_[j] / 20);//FIXME: 这个是画圈的颜色问题
        cv::circle(img_track_, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }

    // -------------在右目图像上标记特征点
    if (!imRight.empty())
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(img_track_, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            //画出左右目的匹配直线 curLeftPtsTrackRight找不到啊！！
            // cv::Point2f leftPt = curLeftPtsTrackRight[i];
            // cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
    
    std::map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(img_track_, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
            // 在imTrack上，从curLeftPts到mapIt->second画箭头
        }
    }

    // cv::imshow("img_track_", img_track_);
    // cv::waitKey();

    // cv::Mat imCur2Compress;
    // cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));//调整图像的大小
}

double Tracker::distance(cv::Point2f &pt1, cv::Point2f &pt2){
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

bool Tracker::inBorder(const cv::Point2f &pt){
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col_ - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row_ - BORDER_SIZE;
}

template <typename T> 
void Tracker::reduceVector(T &vec, const std::vector<uchar>& status){
    size_t index = 0;
    for (size_t k = 0; k < vec.size(); k++) {
        if (status[k]) {
            vec[index++] = vec[k];
        }
    }
    vec.resize(index);
}