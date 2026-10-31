#ifndef TEGRASPEEDHILLCLIMBINGPOLICY_H_
#define TEGRASPEEDHILLCLIMBINGPOLICY_H_

#include "TegraSpeedPolicy.h"
#include "ApplicationMonitor.h"
#include "getNode_Tegra.h"
#include "MappingConsts.h"
#include "utilization.h"
#include <map>
#include <iostream>
#include <fstream>

#include <ros/ros.h>
#include <std_msgs/String.h>

#define SPEED_MAX 5
#define SPEED_MIN 0.25
#define SPEED_STEP 0.25

class TegraSpeedHillClimbingPolicy: public TegraSpeedPolicy{
  public:
    enum HCState{
      IDLE = 0,
      EXPLORE = 1,
      APPLY_BEST = 2,
      TOP = 3,
      MAX_FREQ = 4
    };
    
    TegraSpeedHillClimbingPolicy(std::string log_dir, int freq_b, int freq_l, int freq_g, float speed);
    void run(int cycle);
    typedef struct {
      float speed;
      int freq;
      float thr;
      float comp_power;
      float mech_power;
    } obs_config;

    typedef struct{
      int index;
      TegraSpeedHillClimbingPolicy::obs_config center_obs;
      TegraSpeedHillClimbingPolicy::obs_config neighbors_obs[4];
      int count_neighbors;
    } explore_data_structure;

    int state;
    int init_freq;
    float init_speed;
    TegraSpeedHillClimbingPolicy::explore_data_structure explore_state;
    void reset_explore_state(float speed_center, int freq_center, float thr_center, float mech_power_center, float comp_power_center);
    void print_explore_state();
    std::vector<std::pair<float, int> > get_neighbors(float speed, int freq);

    std::ofstream* csvHCFile;

  private:
    
};

#endif
