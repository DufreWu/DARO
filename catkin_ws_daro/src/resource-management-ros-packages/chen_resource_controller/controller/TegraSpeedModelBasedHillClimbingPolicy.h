#ifndef TEGRASPEEDMODELBASEDHILLCLIMBINGPOLICY_H_
#define TEGRASPEEDMODELBASEDHILLCLIMBINGPOLICY_H_

#include "TegraSpeedHillClimbingPolicy.h"
#include "ApplicationMonitor.h"
#include "getNode_Tegra.h"
#include "MappingConsts.h"
#include "utilization.h"
#include <map>
#include <iostream>
#include <fstream>


class TegraSpeedModelBasedHillClimbingPolicy: public TegraSpeedHillClimbingPolicy{
  public:
    TegraSpeedModelBasedHillClimbingPolicy(std::string log_dir, std::string model_dir, int freq_b, int freq_l, int freq_g, float speed);
    std::string model_dir;
    float linear_model[11][11][2];
    void load_linear_model();
    void run(int cycle);
    float estimate_computation_power(int freq_b, int freq_l, std::vector<int> u);

  private:
};


#endif
