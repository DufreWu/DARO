#ifndef TEGRASPEEDALLCONSTSPOLICY_H_
#define TEGRASPEEDALLCONSTSPOLICY_H_

#include "TegraSpeedPolicy.h"
#include "ApplicationMonitor.h"
#include "getNode_Tegra.h"
#include "MappingConsts.h"
#include "utilization.h"
#include <map>
#include <iostream>
#include <fstream>
#include <chrono>

#include <ros/ros.h>
#include <std_msgs/String.h>

#define SPEED_MAX 5.0
#define SPEED_MIN 1.0
#define SPEED_STEP 1.0

class TegraSpeedAllConstsPolicy: public TegraSpeedPolicy{
  public:
    TegraSpeedAllConstsPolicy(std::string log_dir, int freq_b, int freq_l, int freq_g, float speed);
    ~TegraSpeedAllConstsPolicy();
    void run(int cycle);

  private:
    std::ofstream* csvSysThrFile;

    // Define CPU and GPU frequency lists
    std::vector<int> cpu_fre_list = {499200, 652800, 806400, 960000, 1113600, 1267200, 1420800, 1574400, 1728000, 1881600, 2035200};
    std::vector<int> gpu_fre_list = {229500, 318750, 408000, 497250, 586500, 675750, 765000, 854250, 943500, 1032750, 1122000, 1211250, 1300500};

    // Generate all combinations of CPU and GPU frequency indices
    std::vector<std::pair<int, int>> clk_action_list;

    void initialize_clk_action_list() {
        for (int i = 0; i < cpu_fre_list.size(); ++i) {
            for (int j = 0; j < gpu_fre_list.size(); ++j) {
                clk_action_list.emplace_back(i, j); // Add the pair (i, j) to the list
            }
        }
    }

    int freq_index = 112;

    // *this->csvSysThrFile << "ITERATION,FREQ_B,FREQ_L,FREQ_G,THR,COMPW,SPEED,MECHW,EXECTIME" << std::endl;
    float totel_exec_time = 0.0;
    float total_comp_power = 0.0;
    float total_mech_power = 0.0;
    float total_thr = 0.0;
    int cycle_count = 0;

    float critical_thr = 0.0;
};

#endif
