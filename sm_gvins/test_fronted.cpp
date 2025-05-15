#include <yaml-cpp/yaml.h>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include "sm_gvins.h"
#include <glog/logging.h>

inline SM_GVINS::Options LoadOptionsFromYaml(const std::string& config_file) {
    SM_GVINS::Options options;

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
        options.image0_topic_ = config["image0_topic"].as<std::string>();
        options.image1_topic_ = config["image1_topic"].as<std::string>();
        options.output_path_  = config["output_path"].as<std::string>();
        options.image_width_  = config["image_width"].as<int>();
        options.image_height_ = config["image_height"].as<int>();

        auto proj = config["projection_parameters"];
        auto dist = config["distortion_parameters"];

        if (!proj || !dist) throw std::runtime_error("Missing projection_parameters or distortion_parameters.");

        double fx = proj["fx"].as<double>();
        double fy = proj["fy"].as<double>();
        double cx = proj["cx"].as<double>();
        double cy = proj["cy"].as<double>();
        options.K_ = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                                                0, fy, cy,
                                                0, 0, 1);

        double k1 = dist["k1"].as<double>();
        double k2 = dist["k2"].as<double>();
        double p1 = dist["p1"].as<double>();
        double p2 = dist["p2"].as<double>();
        options.D_ = (cv::Mat_<double>(1, 4) << k1, k2, p1, p2);

        std::vector<double> Tbc0_data = config["body_T_cam0"]["data"].as<std::vector<double>>();
        std::vector<double> Tbc1_data = config["body_T_cam1"]["data"].as<std::vector<double>>();
        if (Tbc0_data.size() != 16 || Tbc1_data.size() != 16)
            throw std::runtime_error("Extrinsic matrices must have 16 elements (4x4).");

        options.Tbc0_ = cv::Mat(4, 4, CV_64F, Tbc0_data.data()).clone();
        options.Tbc1_ = cv::Mat(4, 4, CV_64F, Tbc1_data.data()).clone();
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML value error: " + std::string(e.what()));
    }

    LOG(INFO) << "Load Options Finished.";
    return options;
}

int main(int argc, char** argv)
{
    // google::InitGoogleLogging(argv[0]);

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
