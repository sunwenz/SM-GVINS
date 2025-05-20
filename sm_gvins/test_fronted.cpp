#include <yaml-cpp/yaml.h>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include "sm_gvins.h"
#include <glog/logging.h>

inline SM_GVINS::Options LoadOptionsFromYaml(const std::string& config_file) {
    SM_GVINS::Options options;

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
        options.image0_topic_ = config["image0_topic"].as<std::string>();
        options.image1_topic_ = config["image1_topic"].as<std::string>();
        options.output_path_  = config["output_path"].as<std::string>();

        options.estimator_options_.tracker_options_.image_width_  = config["image_width"].as<int>();
        options.estimator_options_.tracker_options_.image_height_ = config["image_height"].as<int>();

        // === 读取 cam0 内参和畸变参数 ===
        auto cam0_proj = config["cam0"]["projection_parameters"];
        auto cam0_dist = config["cam0"]["distortion_parameters"];
        if (!cam0_proj || !cam0_dist)
            throw std::runtime_error("Missing cam0 projection_parameters or distortion_parameters.");

        options.estimator_options_.tracker_options_.K0_ = (cv::Mat_<double>(3, 3) <<
            cam0_proj["fx"].as<double>(), 0, cam0_proj["cx"].as<double>(),
            0, cam0_proj["fy"].as<double>(), cam0_proj["cy"].as<double>(),
            0, 0, 1);

        options.estimator_options_.tracker_options_.D0_ = (cv::Mat_<double>(1, 4) <<
            cam0_dist["k1"].as<double>(), cam0_dist["k2"].as<double>(),
            cam0_dist["p1"].as<double>(), cam0_dist["p2"].as<double>());

        // === 读取 cam1 内参和畸变参数 ===
        auto cam1_proj = config["cam1"]["projection_parameters"];
        auto cam1_dist = config["cam1"]["distortion_parameters"];
        if (!cam1_proj || !cam1_dist)
            throw std::runtime_error("Missing cam1 projection_parameters or distortion_parameters.");

        options.estimator_options_.tracker_options_.K1_ = (cv::Mat_<double>(3, 3) <<
            cam1_proj["fx"].as<double>(), 0, cam1_proj["cx"].as<double>(),
            0, cam1_proj["fy"].as<double>(), cam1_proj["cy"].as<double>(),
            0, 0, 1);

        options.estimator_options_.tracker_options_.D1_ = (cv::Mat_<double>(1, 4) <<
            cam1_dist["k1"].as<double>(), cam1_dist["k2"].as<double>(),
            cam1_dist["p1"].as<double>(), cam1_dist["p2"].as<double>());

        std::vector<double> Tbc0_data = config["body_T_cam0"]["data"].as<std::vector<double>>();
        std::vector<double> Tbc1_data = config["body_T_cam1"]["data"].as<std::vector<double>>();
        if (Tbc0_data.size() != 16 || Tbc1_data.size() != 16)
            throw std::runtime_error("Extrinsic matrices must have 16 elements (4x4).");

        using Matrix4dRowMajor = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
        Eigen::Matrix4d Tbc0_mat = Matrix4dRowMajor(Tbc0_data.data());
        Eigen::Matrix4d Tbc1_mat = Matrix4dRowMajor(Tbc1_data.data());

        Eigen::Matrix3d Rbc0 = Tbc0_mat.block<3,3>(0,0);
        Eigen::Vector3d tbc0 = Tbc0_mat.block<3,1>(0,3);
        Eigen::Matrix3d Rbc1 = Tbc1_mat.block<3,3>(0,0);
        Eigen::Vector3d tbc1 = Tbc1_mat.block<3,1>(0,3);
        LOG(INFO) << "Tbc0_mat:\n" << Tbc0_mat;
        LOG(INFO) << "Tbc1_mat:\n" << Tbc1_mat;

        options.estimator_options_.Tbc0_ = SE3(SO3(Rbc0), tbc0);
        options.estimator_options_.Tbc1_ = SE3(SO3(Rbc1), tbc1);

        // 计算 cam0_T_cam1 = Tbc0.inverse() * Tbc1
        SE3 Tc0c1 = options.estimator_options_.Tbc0_.inverse() * options.estimator_options_.Tbc1_;
        options.estimator_options_.tracker_options_.Tc0c1_ = Tc0c1;

    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML value error: " + std::string(e.what()));
    }

    LOG(INFO) << "Load Options Finished.";
    return options;
}


int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    google::SetStderrLogging(google::INFO); 
    FLAGS_colorlogtostderr = true;
    FLAGS_logtostderr = true;

    ros::init(argc, argv, "sm_gvins_node");
    ros::NodeHandle nh;

    if (argc < 2) {
        std::cerr << "Usage: rosrun your_pkg sm_gvins_node <config.yaml>" << std::endl;
        return -1;
    }

    std::string config_file = argv[1];
    auto options = LoadOptionsFromYaml(config_file);
    SM_GVINS gvins(nh, options);
    
    ros::spin();
    return 0;
}
