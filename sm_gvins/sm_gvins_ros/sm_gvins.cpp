#include "sm_gvins.h"
#include <glog/logging.h>

SM_GVINS::SM_GVINS(ros::NodeHandle& nh)
    : drawer_(nh)
{
    image0_sub_ = nh.subscribe(Parameters::image0_topic_, 100, &SM_GVINS::img0_callback, this);
    image1_sub_ = nh.subscribe(Parameters::image1_topic_, 100, &SM_GVINS::img1_callback, this);

    sync_thread_ = std::thread(&SM_GVINS::sync_process, this);

    LOG(INFO) << "sm_gvins_node start";
}

SM_GVINS::~SM_GVINS() {
    running_ = false;
    if (sync_thread_.joinable()) {
        sync_thread_.join();
    }
}

void SM_GVINS::sync_process()
{
    while(running_)
    {
        cv::Mat image0, image1;
        std_msgs::Header header;
        double time = 0;
        m_buf_.lock();
        if (!img0_buf_.empty() && !img1_buf_.empty())
        {
            double time0 = img0_buf_.front()->header.stamp.toSec();
            double time1 = img1_buf_.front()->header.stamp.toSec();
            // 0.003s sync tolerance
            if(time0 < time1 - 0.003)
            {
                img0_buf_.pop();
                LOG(INFO) << "throw img0";
            }
            else if(time0 > time1 + 0.003)
            {
                img1_buf_.pop();
                LOG(INFO) << "throw img1";
            }
            else
            {
                time = img0_buf_.front()->header.stamp.toSec();
                header = img0_buf_.front()->header;
                image0 = getImageFromMsg(img0_buf_.front());
                img0_buf_.pop();
                image1 = getImageFromMsg(img1_buf_.front());
                img1_buf_.pop();
                LOG(INFO) << "find img0 and img1";
            }
        }
        m_buf_.unlock();

        if(!image0.empty()){
            Image image(time, image0, image1);
            drawer_.updateFrame(image);
            estimator_.AddImage(image);
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

cv::Mat SM_GVINS::getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}