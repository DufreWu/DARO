#ifndef TEGRASPEEDPOLICY_H_
#define TEGRASPEEDPOLICY_H_

#include "Policy.h"
#include "ApplicationMonitor.h"
#include "getNode_Tegra.h"
#include "MappingConsts.h"
#include "utilization.h"
#include "Battery_2RC_ECM.h"
#include <map>
#include <iostream>
#include <fstream>
#include <chrono>

#include <ros/ros.h>
#include <std_msgs/String.h>

class TegraSpeedPolicy: public Policy{
  public:
    typedef struct{
      //Throughput info
      std::string name;
      float thr; //last measure of throughput
      float thr_ref; //target throughput
      int mapping; //0=LITTLE, 1=big, 2=GPU
      bool mt;
      float unit_th_processing_time;
    } appl_state;
    std::string log_dir;
    TegraSpeedPolicy();
    TegraSpeedPolicy(std::string log_dir, int freq_b, int freq_l, int freq_g, float speed);
    ~TegraSpeedPolicy();
    void run(int cycle);
    typedef struct {
      float err, prev_err = 0, integ_err = 0, deriv_err;
      float kp = 1, ki = 0.01, kd = 0.1;
    } pid_params;
    pid_params freq_pid_params;
    float exec_time = 0.0; // ms
    
  protected:
    monitor_t* appl_monitor;
    GetNode_Tegra* sensors;
    Utilization* utilization;
    Battery2RCECM* battery;

    std::ofstream* csvSysFile;
    std::string csvSysFilename;
    std::ofstream* csvApplFile;
    std::string csvApplFilename;
  
    std::map<pid_t, TegraSpeedPolicy::appl_state> appl_info;
    
    float speed;
    float distance=0.0;
    float getCurrentMechW();
    float getMechW(float sp);
    float getMechW(float sp, float acc);
    void setFreqLittle(int freq);
    void setFreqBig(int freq);
    void setFreqGpu(int freq);
    
    float calDistance(float sp, float dt);
    void setSpeed(float sp);
    float getMechEnergyPerMeter(float mech_power, float speed);
    float getCompEnergyPerMeter(float comp_power, float speed);
    float getEnergyPerMeter(float comp_power, float mech_power, float speed);

    std::chrono::seconds::rep clock_to_s(std::chrono::time_point<std::chrono::system_clock> tp);
    std::chrono::nanoseconds::rep clock_to_ns(std::chrono::time_point<std::chrono::system_clock> tp);
    std::string clock_to_s_str(std::chrono::time_point<std::chrono::system_clock> tp);
    
    ros::NodeHandle nh;
    ros::Publisher speedPublisher;
};

#endif
