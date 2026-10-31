#include <chrono>
#include <thread>
#include <fstream>
#include <cmath>
#include <map>
#include "ros/ros.h"
#include "ros/time.h"
#include "dvs_msgs/Event.h"
#include "dvs_msgs/EventArray.h"
#include <std_msgs/String.h>
#include "argparse.hpp"

#define FOR(i,m,n) for(int i=m; i<n; i++)

using namespace std;

typedef std::chrono::high_resolution_clock Clock;

ofstream logfile;

struct MyArgs : public argparse::Args {
    int &track_length = kwarg("track-length", "length of track in meter").set_default(100);    
    float &init_speed = kwarg("init-speed", "speed at the beginning, when its param is not set").set_default(1.0);
    int &n_events     = kwarg("n-events", "sum of events occurring in the whole track").set_default(100000);
    int &hz                = kwarg("hz", "how many batches per second?").set_default(30);
    std::string &log_dir = kwarg("log-dir", "path to dir where csv log is written to");
    bool &loop         = flag("l,loop", "Loop over the track, and publish indefinitely");
    bool &verbose         = flag("v,verbose", "A flag to toggle verbose");
};


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
void publish_events(MyArgs args, ros::NodeHandle nh) {
    ros::Publisher pub = nh.advertise<dvs_msgs::EventArray>("/dvs/events", -1);

    float speed_val = args.init_speed;
    float t_interval = (1.0 / args.hz);
    float x = 0.0;
    float x_interval;

    while( x < args.track_length) {
        auto t = Clock::now();
        auto next_time = t + std::chrono::duration<long int, std::nano>((long int)(t_interval*1e9));
        auto t_begin_window = ros::Time::now();
        auto t_mid_window = t_begin_window + ros::Duration(t_interval / 2);
        auto t_end_window = t_begin_window + ros::Duration(t_interval);
        
        if(nh.hasParam("/chen/speed")) {
            nh.getParam("/chen/speed", speed_val);
            cout << "speed: " << speed_val << "\n"; 
        } else {
            cout << "Speed param not set! \n";
        }
        x_interval = (t_interval * speed_val);
        int bs = int((args.n_events / args.track_length) * x_interval);

        cout << fixed << clock_to_ns(t) << "\n";

        dvs_msgs::EventArray e_array;
        e_array.width=127; 
        e_array.height=127;
        FOR(j, 0, bs) {
            dvs_msgs::Event e;
            e.x = rand() % 127; 
            e.y = rand() % 127;
            e.ts = t_mid_window;
            e.polarity = rand() % 2;
            e_array.events.push_back(e);
        }
        if (bs > 0) {
            e_array.events[0].ts = t_begin_window;
            e_array.events[e_array.events.size() - 1].ts = t_end_window;
        }
        
        this_thread::sleep_until(next_time);
        pub.publish(e_array);

        auto now = Clock::now();
        auto preparation_time = clock_to_ns(now) - clock_to_ns(t);
        logfile << clock_to_s_str(now) << "," << t_begin_window.toNSec() << "," << t_end_window.toNSec() << "," << e_array.events.size() << "," 
                << preparation_time << "," << clock_to_ns(now) << "," << clock_to_ns(next_time) << ","
                << speed_val << "," << x << "," << x_interval                
                << "\n";

        cout << "creating and publishing an event array with bs: " << bs << " events took " << preparation_time*1e-6 << " milliseconds\n"; 

        x += x_interval;
        if(args.loop && x >= args.track_length) x = 0;
    }
}

int main(int argc, char **argv) {
    MyArgs args = argparse::parse<MyArgs>(argc, argv);

    if (args.verbose)
        args.print();       // prints all variables 

    ros::init(argc, argv, "camera_rand");
    ros::NodeHandle nh;
    
    logfile.open(args.log_dir + "/"+ "event_publish.csv", ios::out);
    logfile << "TIME,TIME_WINDOW_BEGIN_NS,TIME_WINDOW_END_NS,N_EVENTS,PREPARATION_TIME,PUBLISH_UTIME_NS,NEXT_PUBLISH_TIME,SPEED,X,dX\n";
    
    // std::thread worker_thread(publish_events, args);
    // worker_thread.detach();

    publish_events(args, nh);
    
    logfile.close();
    return 0;
}