#include <vector>
#include <queue>
#include <string>
#include <iostream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "sm_gvins.h"
#include "visual/dataset.h"

using namespace std;

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
        
        auto calib_path  = config["calib_path"].as<std::string>();
        ifstream fin(calib_path);
        if (!fin) {
            LOG(ERROR) << "cannot find " << calib_path << "/calib.txt!";
        }

        for (int i = 0; i < 2; ++i) {
            char camera_name[3];
            for (int k = 0; k < 3; ++k) {
                fin >> camera_name[k];
            }
            double projection_data[12];
            for (int k = 0; k < 12; ++k) {
                fin >> projection_data[k];
            }
            Eigen::Matrix3d K;
            K << projection_data[0], projection_data[1], projection_data[2],
                projection_data[4], projection_data[5], projection_data[6],
                projection_data[8], projection_data[9], projection_data[10];
            Eigen::Vector3d t;
            t << projection_data[3], projection_data[7], projection_data[11];
            t = K.inverse() * t;
            K = K * 0.5;

            options.estimator_options_.K_[i] = K;
            options.estimator_options_.t_[i] = t;

            LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();
        }
        fin.close();

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

        LOG(INFO) << "options.estimator_options_.tracker_options_.K0_:\n" << options.estimator_options_.tracker_options_.K0_;
        LOG(INFO) << "options.estimator_options_.tracker_options_.K1_:\n" << options.estimator_options_.tracker_options_.K1_;
        LOG(INFO) << "options.estimator_options_.tracker_options_.D0_:\n" << options.estimator_options_.tracker_options_.D0_;
        LOG(INFO) << "options.estimator_options_.tracker_options_.D1_:\n" << options.estimator_options_.tracker_options_.D1_;

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
        LOG(INFO) << "Tc0c1:\n" << Tc0c1.rotationMatrix() << "\n" << Tc0c1.translation().transpose();

        options.estimator_options_.tracker_options_.Tc0c1_ = Tc0c1;
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML value error: " + std::string(e.what()));
    }

    LOG(INFO) << "Load Options Finished.";
    return options;
}

// 主函数
int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::SetStderrLogging(google::INFO); 
    FLAGS_colorlogtostderr = true;
    FLAGS_logtostderr = true;

    std::string config_file = "./config/kitti_config00-02.yaml";
    std::string dataset_dir = "/media/shentao/sunwenzSE/KITTI/dataset/sequences/05";
    Dataset::Ptr dataset_(new Dataset(dataset_dir));
    dataset_->Init();

    auto options = LoadOptionsFromYaml(config_file);
    std::shared_ptr<std::ofstream> f_out = std::make_shared<std::ofstream>(options.output_path_);
    std::shared_ptr<Estimator> estimator = std::make_shared<Estimator>(f_out, options.estimator_options_);

    estimator->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));
    while (true)
    {
        auto new_image = dataset_->NextFrame();
        if(new_image == nullptr){
            break;
        }

        estimator->AddImage(*new_image);
    }
    
/* 
    std::string config_file = "./config/kitti_config00-02.yaml";
    auto options = LoadOptionsFromYaml(config_file);

    std::shared_ptr<std::ofstream> f_out = std::make_shared<std::ofstream>(options.output_path_);
    std::shared_ptr<Estimator> estimator = std::make_shared<Estimator>(f_out, options.estimator_options_);
    
    std::string dataset_dir = "/media/shentao/sunwenzSE/KITTYdatasets/2011_10_03_drive_0027/2011_10_03/2011_10_03_drive_0027_sync/";
    if (dataset_dir.empty())
    {
        cerr << "Please input dataset directory in default.yaml!" << endl;
    }
    ifstream associate,timestamps;
    associate.open(dataset_dir + "associate.txt");
    // timestamps.open("/media/sunwenz/sunwenzSE/KITTYdatasets/2011_10_03_drive_0027/2011_10_03/2011_10_03_drive_0042_sync/image_00/timestamps.txt",std::ios::in);
    if (!associate)
    {
        cerr << "assciate file does not exist!" << endl;
    }


    std::string time_stamp, img0_file, img1_file;
    while (associate >> time_stamp >> img0_file >> img1_file) {
        std::string path_left  = dataset_dir + img0_file;
        std::string path_right = dataset_dir + img1_file;

        cv::Mat image_left  = cv::imread(path_left,  cv::IMREAD_GRAYSCALE);
        cv::Mat image_right = cv::imread(path_right, cv::IMREAD_GRAYSCALE);

        if (image_left.empty() || image_right.empty()) {
            std::cerr << "Failed to load image: " << path_left << " or " << path_right << std::endl;
            continue;
        }
        cv::Mat image_left_resized, image_right_resized;
        cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5,
                cv::INTER_NEAREST);
        cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5,
                cv::INTER_NEAREST);

        Image image(atof(time_stamp.c_str()), image_left_resized, image_right_resized);
        estimator->AddImage(image);
        std::cout << "new image added: " << time_stamp << std::endl;
    }
     */
    return 0;
}