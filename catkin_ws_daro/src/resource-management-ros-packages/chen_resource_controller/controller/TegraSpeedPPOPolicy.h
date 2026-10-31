#ifndef TEGRASPEEDPPOPOLICY_H_
#define TEGRASPEEDPPOPOLICY_H_

#include "TegraSpeedPolicy.h"
#include "ApplicationMonitor.h"
#include "getNode_Tegra.h"
#include "MappingConsts.h"
#include "utilization.h"
#include "Battery_2RC_ECM.h"
#include "../agents/ppo_agent.h"
#include <map>
#include <iostream>
#include <fstream>
#include <torch/script.h>

#include <ros/ros.h>
#include <std_msgs/String.h>

class TegraSpeedPPOPolicy : public TegraSpeedPolicy{
  public:
    TegraSpeedPPOPolicy(std::string log_dir, std::string model_dir, int freq_b, int freq_l, int freq_g, float speed);
    void run(int cycle);

  private:
    std::string model_dir;
    // Define CPU and GPU frequency lists
    std::vector<int> cpu_fre_list = {499200, 652800, 806400, 960000, 1113600, 1267200, 1420800, 1574400, 1728000, 1881600, 2035200};
    std::vector<int> gpu_fre_list = {229500, 318750, 408000, 497250, 586500, 675750, 765000, 854250, 943500, 1032750, 1122000, 1211250, 1300500};
    // std::vector<int> gpu_fre_list = {229500};
    std::vector<float> speed_levels = {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 
                                       2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0};
    std::vector<std::tuple<int, int, int>> config_list;

    std::vector<std::tuple<int, int, int>> generate_clk_action_list() {
        std::vector<std::tuple<int, int, int>> list;
        for (int i = 0; i < cpu_fre_list.size(); ++i) {
            for (int j = 0; j < gpu_fre_list.size(); ++j) {
                for (int k = 0; k < speed_levels.size(); ++k) {
                    list.emplace_back(i, j, k);
                }
            }
        }
        return list;
    }

    float target_speed = 0.5;
    int cpu_freq_index = 0; // Index for CPU frequency
    int gpu_freq_index = 0; // Index for GPU frequency

    std::ofstream* csvPPOFile;
    PPOAgent agent;
    
    int episode = 0;
    float total_reward = 0.0, min_reward = 10.0, max_reward = 0.0;
    std::vector<float> state_vector = {0.0, 0.5, 1.7, 0.0, 0.0, 0.0}; // Initial state vector
    std::vector<float> next_state_vector = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // Next state vector
    int action_index = 0;
    float reward = 0.0; // Initial reward
    
    float get_reward(float mech_power, float comp_power, int thr, int thr_req, float speed);
};

#endif
