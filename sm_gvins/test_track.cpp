#include <yaml-cpp/yaml.h>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include "tracker.h"
#include "map.h"
#include "parameters.h"
#include "math.h"
#include <glog/logging.h>

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    google::SetStderrLogging(google::INFO); 
    FLAGS_colorlogtostderr = true;
    FLAGS_logtostderr = true;

    Tracker::Options options;
    MapPtr map = std::make_shared<Map>();
    TrackerPtr tracker = std::make_shared<Tracker>(map);


    ros::spin();
    return 0;
}