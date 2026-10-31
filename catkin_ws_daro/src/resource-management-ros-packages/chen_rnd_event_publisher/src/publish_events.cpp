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

void SpeedMsgCallback(const std_msgs::StringConstPtr &sp_msg) {
    cout << "Hi4\n";
    cout << sp_msg << "\n";
    string sp = sp_msg->data;
    cout << "Recieved speed: " << sp << "\n";
}

struct MyArgs : public argparse::Args {
    std::string &batch_sizes = kwarg("batch-sizes", "csv batch sizes").set_default("100,100000");
    int &duration_each     = kwarg("duration-each", "duration of playing each batch-size").set_default(2);
    int &hz                = kwarg("hz", "how many batches per second?").set_default(30);
    std::string &log_dir = kwarg("log-dir", "path to dir where csv log is written to");
    bool &loop         = flag("l,loop", "Loop over batch-sizes, and publish indefinitely");
    bool &verbose         = flag("v,verbose", "A flag to toggle verbose");
};

int bs_sample_size = 2;

vector<int> tokenize(string s, string del = " ")
{
    int start = 0;
    int end = s.find(del);
    string tk;
    std::vector<int> tokens;
    while (end != -1) {
        tk = s.substr(start, end - start);
        tokens.push_back(stoi(tk));
        start = end + del.size();
        end = s.find(del, start);
    }
    tk = s.substr(start, end - start);
    tokens.push_back(stoi(tk));
    return tokens;
}

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
    ros::Publisher pub = nh.advertise<dvs_msgs::EventArray>("/dvs/events", 10);

    vector<int> batch_sizes = tokenize(args.batch_sizes, ","); 
    map<int, int> sec_to_noev;

    float speed_val = 1;
    if(nh.hasParam("/chen/speed")) {
        nh.getParam("/chen/speed", speed_val);
        cout << "speed_val: " << speed_val << "\n";
    } else {
        cout << "Speed param not set! \n";
    }

    float t_interval = (1.0 / args.hz);
    int n_batches = int((args.duration_each / speed_val) / t_interval);
    int bs_i = 0;
    while(bs_i < batch_sizes.size()) {
        int bs = batch_sizes[bs_i];
        int k = 0;
        while(k < n_batches) {
            auto t = Clock::now();
            auto next_time = t + std::chrono::duration<long int, std::nano>((long int)(t_interval*1e9));
            auto t_begin_window = ros::Time::now();
            auto t_mid_window = t_begin_window + ros::Duration(t_interval / 2);
            auto t_end_window = t_begin_window + ros::Duration(t_interval);
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
            int t_sec = clock_to_s(t);
            map<int,int>::iterator itr = sec_to_noev.find(t_sec);
            if (itr == sec_to_noev.end()) {
                sec_to_noev.insert(pair<int, int>(t_sec, 1));
            }
            else {
                itr->second += 1;
            }
            
            pub.publish(e_array);
            auto now = Clock::now();
            auto preparation_time = clock_to_ns(now) - clock_to_ns(t);
            logfile << clock_to_s_str(now) << "," << t_begin_window.toNSec() << "," << t_end_window.toNSec() << "," << e_array.events.size() << "," << preparation_time << "," << clock_to_ns(now) << "," << clock_to_ns(next_time) << "\n"; 
            cout << "creating and publishing an event array with bs: " << bs << " events took " << preparation_time*1e-6 << " milliseconds\n"; 
            
            if(nh.hasParam("/chen/speed")) {
                nh.getParam("/chen/speed", speed_val);
                cout << "speed: " << speed_val << "\n"; 
            } else {
                cout << "Speed param not set! \n";
            }
            n_batches = int((args.duration_each / speed_val) / t_interval);
            
            cout << k << "/" << n_batches << " | " 
                << "duration original: " << args.duration_each 
                << ", curr speed_val: " << speed_val
                << ", duration actual: "<< (args.duration_each / speed_val) <<"\n"; 
            
            k++;
            this_thread::sleep_until(next_time);
        }
        bs_i += 1;
        if(args.loop) bs_i %= batch_sizes.size();
    }
    for(map<int,int>::iterator itr = sec_to_noev.begin(); itr != sec_to_noev.end(); ++itr){
        cout << "Sec " << itr->first << ": NoE => " << itr->second << endl;      
    }  
}

int main(int argc, char **argv) {
    MyArgs args = argparse::parse<MyArgs>(argc, argv);

    if (args.verbose)
        args.print();       // prints all variables 

    ros::init(argc, argv, "camera_rand");
    ros::NodeHandle nh;
    
    logfile.open(args.log_dir + "/"+ "event_publish.csv", ios::out);
    logfile << "TIME,TIME_WINDOW_BEGIN_NS,TIME_WINDOW_END_NS,N_EVENTS,PREPARATION_TIME,PUBLISH_UTIME_NS,NEXT_PUBLISH_TIME\n";
    
    // std::thread worker_thread(publish_events, args);
    // worker_thread.detach();

    publish_events(args, nh);
    
    logfile.close();
    return 0;
}