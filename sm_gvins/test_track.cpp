#include <yaml-cpp/yaml.h>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include "tracker.h"
#include "map.h"
#include "parameters.h"
#include "math.h"
#include "estimator.h"
#include <glog/logging.h>
#include <boost/format.hpp>

inline bool LoadOptionsFromYaml(const std::string& config_file) {
    // 检查文件是否存在
    std::ifstream fin(config_file);
    if (!fin.good()) {
        throw std::runtime_error("YAML config file not found: " + config_file);
    }

    YAML::Node config;
    try {
        config = YAML::LoadFile(config_file);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML parsing failed: " + std::string(e.what()));
    }

    try {
        Parameters::image0_topic_ = config["image0_topic"].as<std::string>();
        Parameters::image1_topic_ = config["image1_topic"].as<std::string>();
        Parameters::output_path_  = config["output_path"].as<std::string>();
        Parameters::width_  = config["image_width"].as<int>();
        Parameters::height_ = config["image_height"].as<int>();

        auto proj = config["projection_parameters"];
        auto dist = config["distortion_parameters"];

        if (!proj || !dist) throw std::runtime_error("Missing projection_parameters or distortion_parameters.");

        Parameters::fx_ = proj["fx"].as<double>();
        Parameters::fy_ = proj["fy"].as<double>();
        Parameters::cx_ = proj["cx"].as<double>();
        Parameters::cy_ = proj["cy"].as<double>();
        // options.K_ = (cv::Mat_<double>(3, 3) << fx, 0, cx,
        //                                         0, fy, cy,
        //                                         0, 0, 1);

        Parameters::k1_ = dist["k1"].as<double>();
        Parameters::k2_ = dist["k2"].as<double>();
        Parameters::p1_ = dist["p1"].as<double>();
        Parameters::p2_ = dist["p2"].as<double>();
        // options.D_ = (cv::Mat_<double>(1, 4) << k1, k2, p1, p2);

        std::vector<double> Tbc0_data = config["body_T_cam0"]["data"].as<std::vector<double>>();
        std::vector<double> Tbc1_data = config["body_T_cam1"]["data"].as<std::vector<double>>();
        if (Tbc0_data.size() != 16 || Tbc1_data.size() != 16)
            throw std::runtime_error("Extrinsic matrices must have 16 elements (4x4).");

        Eigen::Matrix4d Tbc0_mat, Tbc1_mat;
        Tbc0_mat = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(Tbc0_data.data());
        Tbc1_mat = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(Tbc1_data.data());

        Sophus::SE3d Tbc0_(Tbc0_mat.block<3, 3>(0, 0), Tbc0_mat.block<3, 1>(0, 3));
        Sophus::SE3d Tbc1_(Tbc1_mat.block<3, 3>(0, 0), Tbc1_mat.block<3, 1>(0, 3));

        Parameters::Tc0c1_ = Tbc0_.inverse() * Tbc1_;
        Parameters::base_ = Parameters::Tc0c1_.translation().norm();
        Parameters::base_fx_ = Parameters::base_ * Parameters::fx_;
        Parameters::image_fps_ = 10;
        
        cv::Mat mat(4, 2, CV_32F);
        mat.ptr<float>(0)[0] = 0.0; //左上
        mat.ptr<float>(0)[1] = 0.0;
        mat.ptr<float>(1)[0] = Parameters::width_; //右上
        mat.ptr<float>(1)[1] = 0.0;
        mat.ptr<float>(2)[0] = 0.0; //左下
        mat.ptr<float>(2)[1] = Parameters::height_;
        mat.ptr<float>(3)[0] = Parameters::width_; //右下
        mat.ptr<float>(3)[1] = Parameters::height_;

        // 角点去畸变
        // cv::undistortPoints(mat, mat, Camera::cvK_, Camera::D_, cv::Mat(), Camera::cvK_);

        // 选取最小和最大的边界坐标
        Parameters::min_X_ = min(mat.ptr<float>(0)[0], mat.ptr<float>(2)[0]); //左上和左下横坐标最小的
        Parameters::max_X_ = max(mat.ptr<float>(1)[0], mat.ptr<float>(3)[0]); //右上和右下横坐标最大的
        Parameters::min_Y_ = min(mat.ptr<float>(0)[1], mat.ptr<float>(1)[1]); //左上和右上纵坐标最小的
        Parameters::max_Y_ = max(mat.ptr<float>(2)[1], mat.ptr<float>(3)[1]); //左下和右下纵坐标最小的

        ORBextractor::initStaticParam();

    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML value error: " + std::string(e.what()));
    }

    LOG(INFO) << "Load Options Finished.";
}

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    google::SetStderrLogging(google::INFO); 
    FLAGS_colorlogtostderr = true;
    FLAGS_logtostderr = true;

    auto save_vec3 = [](std::ofstream& fout, const Vec3d& v) { fout << v[0] << " " << v[1] << " " << v[2] << " "; };
    auto save_quat = [](std::ofstream& fout, const Quatd& q) {
        fout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " ";
    };

    auto save_result = [&save_vec3, &save_quat](std::ofstream& fout, double timestamp, const SE3& save_state) {
        fout << std::setprecision(18) << timestamp << " " << std::setprecision(9);
        save_vec3(fout, save_state.translation());
        save_quat(fout, save_state.unit_quaternion());
        fout << std::endl;
    };

    LoadOptionsFromYaml("/home/sunwenz/Code/sm_gvins_ws/src/config/kitti_config00-02.yaml");
    
    Estimator estimator;
    std::string dataset_dir = "/media/sunwenz/sunwenzSE/KITTYdatasets/2011_10_03_drive_0027/2011_10_03/2011_10_03_drive_0027_sync/";
    if (dataset_dir.empty())
    {
        cerr << "Please input dataset directory in default.yaml!" << endl;
    }
    ifstream associate,timestamps;
    associate.open(dataset_dir + "/associate.txt");
    // timestamps.open("/media/sunwenz/sunwenzSE/KITTYdatasets/2011_10_03_drive_0027/2011_10_03/2011_10_03_drive_0042_sync/image_00/timestamps.txt",std::ios::in);
    if (!associate)
    {
        cerr << "assciate file does not exist!" << endl;
    }

    // 读取所有帧的两张图片的路径
    while (dataset_dir.back() != '/')
    {
        dataset_dir.pop_back();
    }
    cout << "dataset: " << dataset_dir << endl;

    bool is_first = true;
    std::ofstream fout_v("/home/sunwenz/Code/sm_gvins_ws/src/output/vins_visual.txt");

    while(true){
        if(associate.eof() || !associate.good())
            break;
        
        string time_stamp, img0_file, img1_file;
        associate >> time_stamp >>img0_file >> img1_file;

        cv::Mat image_left  = cv::imread(dataset_dir + img0_file);
        cv::Mat image_right = cv::imread(dataset_dir + img1_file);
        cv::Mat gray;
        if (image_left.channels() == 3) { // 彩色图像
            cv::cvtColor(image_left, image_left, cv::COLOR_BGR2GRAY);
            cv::cvtColor(image_right, image_right, cv::COLOR_BGR2GRAY);
        }

        
        Image image(atof(time_stamp.c_str()), image_left, image_right);
        estimator.AddImage(image);

        // my_vins.TrackFrame(atof(time_stamp.c_str()),image_left,image_right);
    }

    estimator.process_.join();
    return 0;
}