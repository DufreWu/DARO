#include <chrono>
#include <fstream>
#include <cmath>

#include "../../chen_resource_controller/controller/ApplicationMonitor.h"
#include "../../chen_resource_controller/controller/MappingConsts.h"

#include "ros/ros.h"
#include "ros/time.h"
#include "dvs_msgs/Event.h"
#include "dvs_msgs/EventArray.h"
#include <std_msgs/String.h>

using namespace std;

typedef std::chrono::high_resolution_clock Clock;

// global stats
double total_time_ = 0.;
int total_events_ = 0;
auto t_init_last_batch = Clock::now();
bool first_batch;

data_t *data;
double req_thr = 30;
int mapping = 0;

float publish_rate = 0;
float n_event_filtered = 0;
// used to save corner info
ofstream outfile;

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
void EventMsgCallback(const dvs_msgs::EventArray::ConstPtr &event_msg) 
{
    monitorTick(data);
    setUnitThProcessingTime(data, 0.033);
    // // if(req_thr > 0)
	// //     autosleep(data, req_thr);
    
    // auto t_init = Clock::now();

    // const int n_event = event_msg->events.size();
    // const long long ns_from_prev_batch = clock_to_ns(t_init) - clock_to_ns(t_init_last_batch);
    // publish_rate = 0.8*publish_rate + 0.2*(1e9 / ns_from_prev_batch);
    // n_event_filtered = 0.8*n_event_filtered + 0.2*(n_event);
    
    // auto t_init_ns = clock_to_ns(t_init);
    // auto t_init_s = clock_to_s(t_init);
    // auto t_init_s_str = clock_to_s_str(t_init);

    // if(!first_batch) {
    //     outfile << t_init_s_str << "," << t_init_s << "," << t_init_ns << "," 
    //          << n_event << "," << ns_from_prev_batch << "," 
    //          << publish_rate << "," << n_event_filtered << std::endl;
    // } else {
    //     first_batch = false;
    // }
    // t_init_last_batch = t_init;
}

int main(int argc, char **argv)
{
    //Antonio platform
    std::cout << "Throughput " << req_thr << std::endl;
    data = monitorAttach(argv[0], req_thr, mapping, 1, true);
	
    ros::init(argc, argv, "event_counter");
    ros::NodeHandle nh;

    string log_dir;
    if(!nh.hasParam("/chen/log_dir")) {
        std::cout << "log_dir parameter not set! paramset /chen/log_dir /path/to/logdir" << std::endl;
        exit(0);
    }
    nh.getParam("/chen/log_dir", log_dir);
    
    std::cout << "Logs in " << log_dir << std::endl;

    outfile.open(log_dir + "/" + "event_counter.csv", ios::out);
    if (!outfile.is_open()) {
        std::cout << "Error opening file. " << std::endl;
        return 0;
    }
    // logfile << t_init_s_str << "," << t_init_s << "," << t_init_ns << "," 
    //         << n_event << "," << ns_from_prev_batch << "," << publish_rate << std::endl;
    outfile << "TIME,UTIME,UTIME_NS,N_EVENT,NS_FROM_PREV_BATCH,PUBLISH_RATE,N_EVENT_FILTERED" << std::endl;
    
    first_batch = true;

    ros::Subscriber event_sub = nh.subscribe("events", 100, &EventMsgCallback);
    std::cout << ros::ok() << " ------------- \n"; 
    while (ros::ok()) {ros::spinOnce();} // Preferred over ros::spin() for performance

    outfile.close();
    return 0;
}

