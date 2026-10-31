/****************************************************************************************
*  FA-Harris corner detection. 
*
*  This method will save corner events to `/corner/scene_/fa_harris.txt` 
*  with (x, y, t, p). Modify `corner.launch` file to choose scene before `launch`.
*
*  Author: Ruoxiang Li
*  Date: 2019.1.20
*
****************************************************************************************/

#include <chrono>
#include <fstream>
#include <cmath>
#include "fa_harris_detector_cpu.h"
#include "fa_harris_detector_gpu.h"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>


#include "../../chen_resource_controller/controller/ApplicationMonitor.h"
#include "../../chen_resource_controller/controller/MappingConsts.h"
#include "ros/time.h"
using namespace std;

typedef std::chrono::high_resolution_clock Clock;

fa_harris_cpu::FAHarrisDetector fa_harris_detector_cpu = fa_harris_cpu::FAHarrisDetector();
fa_harris_gpu::FAHarrisDetector fa_harris_detector_gpu = fa_harris_gpu::FAHarrisDetector();

ros::Publisher corner_pub;

// global stats
double total_time_ = 0.;
int total_events_ = 0, total_corners_ = 0;
double total_time_for_add_ = 0.;
auto t_init_last_batch = Clock::now();

data_t *data;
double req_thr = 25;
int mapping = 0;
// used to save corner info
ofstream outfile, logfile, statfile;

std::chrono::seconds::rep clock_to_s(std::chrono::time_point<std::chrono::system_clock> tp) {
    return std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch()).count();
}

std::chrono::nanoseconds::rep clock_to_ns(std::chrono::time_point<std::chrono::system_clock> tp) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
}

std::string clock_to_s_str(std::chrono::time_point<std::chrono::system_clock> tp) {
    std::time_t curr_time_c = std::chrono::system_clock::to_time_t(tp);
    std::string curr_time_s = std::string(std::ctime(&curr_time_c));
    curr_time_s = curr_time_s.substr(0, curr_time_s.size()-1);
    return curr_time_s;
}
void EventMsgCallback(const dvs_msgs::EventArray::ConstPtr &event_msg_in) 
{
    auto t_init = Clock::now();
    dvs_msgs::EventArray event_msg;
    event_msg.header = event_msg_in->header;
    event_msg.width = event_msg_in->width;
    event_msg.height = event_msg_in->height;
    for (auto& e : event_msg_in->events) {
        event_msg.events.push_back(e);
        event_msg.events.push_back(e);
        // event_msg.events.push_back(e);
        // event_msg.events.push_back(e);
    }

    const int n_event = event_msg.events.size();
    if (n_event == 0) {return;}
    dvs_msgs::EventArray corner_msg;
    corner_msg.header = event_msg.header;
    corner_msg.width = event_msg.width;
    corner_msg.height = event_msg.height;
    int cnt = 0;

    int width = event_msg.width;
    int height = event_msg.height;

    // // Create a blank grayscale image
    // cv::Mat event_image = cv::Mat::zeros(height, width, CV_8UC1);

    int use_gpu = 0;
    if (use_gpu == 0) {
        fa_harris_detector_gpu.initializeCudaMemory(width, height);
        fa_harris_detector_gpu.processEventsOnGpu(event_msg);
    } else {
        for (const auto& e : event_msg.events)
        {
            if (fa_harris_detector_cpu.isCornerWithoutFilter(e)) {
                corner_msg.events.push_back(e);
            }
            cnt++;
            // if(cnt > 40000) break;
            // if (e.x >= 0 && e.x < width && e.y >= 0 && e.y < height) {
            //     if (e.polarity) {
            //         event_image.at<uchar>(e.y, e.x) = std::min(255, event_image.at<uchar>(e.y, e.x) + 50);
            //     } else {
            //         event_image.at<uchar>(e.y, e.x) = std::max(0, event_image.at<uchar>(e.y, e.x) - 50);
            //     }
            // }
        }
    }
    
    // // Normalize image for better visibility
    // cv::normalize(event_image, event_image, 0, 255, cv::NORM_MINMAX);
    
    // // Send detected corner events
    corner_pub.publish(corner_msg);

    // save corner events to txt file
    for (const auto& e : corner_msg.events)
    {
        outfile << e.x << " " << e.y << " " << e.ts << " ";
        const int pol = e.polarity ? 1 : 0;
        outfile << pol << "\n";
    }

    const int n_corner = corner_msg.events.size();
    
    ros::Time first_event_time = ros::Time::now();
    ros::Time last_event_time = ros::Time::now();

    if(event_msg.events.size() > 0) {
        first_event_time = event_msg.events[0].ts;
        last_event_time = event_msg.events[event_msg.events.size()-1].ts;
    }

    auto t_init_ns = clock_to_ns(t_init);
    auto t_init_s = clock_to_s(t_init);
    auto t_init_s_str = clock_to_s_str(t_init);
    
    auto t_end = Clock::now();
    auto t_end_ns = clock_to_ns(t_end);

    // TIME,UTIME,PRC_INIT_UTIME_NS,PRC_END_UTIME_NS,EVENT_FIRST_UTIME_NS,EVENT_LAST_UTIME_NS,N_EVENTS,N_CORNERS
    // statfile << t_init_s_str << "," << t_init_s << "," << t_init_ns << "," << t_end_ns << "," 
    //          << first_event_time.toNSec() << ","  << last_event_time.toNSec() << "," 
    //          << n_event << "," << n_corner << std::endl;
    float processing_time = (1.0e-9*(t_end_ns - t_init_ns));
    monitorTick(data);
    setUnitThProcessingTime(data, processing_time);
}

int main(int argc, char **argv)
{
    std::cout << "Throughput " << req_thr << std::endl;
    data = monitorAttach(argv[0], req_thr, mapping, 1, true);
	
    ros::init(argc, argv, "fa_harris");
    ros::NodeHandle nh;

    // load parameter
    string scene_;
    ros::param::param<string>("~scene", scene_, "shapes_6dof");

    string log_dir;
    if(!nh.hasParam("/chen/log_dir")) {
        std::cout << "log_dir parameter not set! paramset /chen/log_dir /path/to/logdir" << std::endl;
        exit(0);
    }
    nh.getParam("/chen/log_dir", log_dir);
    
    std::cout << scene_ << std::endl << log_dir << std::endl;

    outfile.open(log_dir + "/" + "fa_harris_output_" + scene_ +".txt", ios::out);
    logfile.open(log_dir + "/"+ "fa_harris_log.txt", ios::out);
    // statfile.open(log_dir + "/"+ "fa_harris_stat.csv", ios::out);

    if (!outfile.is_open())
    {
        std::cout<< "Error opening file. " << std::endl;
        return 0;
    }
    
    // statfile << "TIME,UTIME,PRC_INIT_UTIME_NS,PRC_END_UTIME_NS,EVENT_FIRST_UTIME_NS,EVENT_LAST_UTIME_NS,N_EVENTS,N_CORNERS" << std::endl;
    
    t_init_last_batch = Clock::now();

    corner_pub = nh.advertise<dvs_msgs::EventArray>("corners", 1);
    ros::Subscriber event_sub = nh.subscribe(
        "/dvs/events", 1, &EventMsgCallback);

    while (ros::ok()) {ros::spinOnce();} // Preferred over ros::spin() for performance

    // print overall statistics
    logfile << "Global Statistics: " << std::endl
    << " Total time [ns]: " << total_time_ << std::endl
    << " Total number of events: " << total_events_ << std::endl
    << " Total number of corners: " << total_corners_ << std::endl
    << " Time/event [ns]: " << total_time_ / (double)total_events_ << std::endl
    << " Events/s: " << total_events_ / total_time_ *  1e9  << std::endl
    << " Reduction (%): " << (1.  - total_corners_  / (double)total_events_) * 100 << std::endl
    << std::endl;
	
    outfile.close();
    logfile.close();
    // statfile.close();
    return 0;
}

