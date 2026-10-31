#ifndef TEGRASPEEDCONSTANTFREQINTERACTIVEPOLICY_H_
#define TEGRASPEEDCONSTANTFREQINTERACTIVEPOLICY_H_

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

class TegraSpeedConstantFreqInteractivePolicy : public TegraSpeedPolicy{
  public:
    TegraSpeedConstantFreqInteractivePolicy(std::string log_dir, int freq_b, int freq_l, int freq_g, float speed);
    void run(int cycle);
    float constant_speed;
};

#endif
