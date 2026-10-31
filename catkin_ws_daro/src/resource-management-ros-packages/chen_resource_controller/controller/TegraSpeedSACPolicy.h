#ifndef TEGRASPEEDSACPOLICY_H_
#define TEGRASPEEDSACPOLICY_H_

#include "TegraSpeedPolicy.h"
#include "ApplicationMonitor.h"
#include "getNode_Tegra.h"
#include "MappingConsts.h"
#include "utilization.h"
#include "Battery_2RC_ECM.h"
#include "../agents/sac_agent.h"
#include <map>
#include <iostream>
#include <fstream>
#include <torch/script.h>

#include <ros/ros.h>
#include <std_msgs/String.h>

class TegraSpeedSACPolicy : public TegraSpeedPolicy{
  public:
    TegraSpeedSACPolicy(std::string log_dir, std::string model_dir, int freq_b, int freq_l, int freq_g, float speed);
    void run(int cycle);

  private:
    std::string model_dir;
    // Define CPU and GPU frequency lists
    std::vector<int> cpu_fre_list = {499200, 652800, 806400, 960000, 1113600, 1267200, 1420800, 1574400, 1728000, 1881600, 2035200};
    // std::vector<int> gpu_fre_list = {229500, 318750, 408000, 497250, 586500, 675750, 765000, 854250, 943500, 1032750, 1122000, 1211250, 1300500};
    std::vector<int> gpu_fre_list = {229500};
    std::vector<int> mapping_list = {0};
    // Generate all combinations of CPU, GPU, and an additional dimension (e.g., memory frequency indices)
    std::vector<std::tuple<int, int, int>> clk_action_list;

    std::vector<std::tuple<int, int, int>> generate_clk_action_list() {
        std::vector<std::tuple<int, int, int>> list;
        for (int i = 0; i < cpu_fre_list.size(); ++i) {
            for (int j = 0; j < gpu_fre_list.size(); ++j) {
                for (int k = 0; k < mapping_list.size(); ++k) {
                    list.emplace_back(i, j, k);
                }
            }
        }
        return list;
    }

    float target_speed = 0.5;
    int cpu_freq_index = 0; // Index for CPU frequency
    int gpu_freq_index = 0; // Index for GPU frequency
    float soh_prev = 0.8; // Previous State of Health (SoH)

// SAC
    // Load the TorchScript model
    // torch::jit::script::Module actor_module;
    // torch::jit::script::Module critic_module;

    std::ofstream* csvSACFile;
    SACAgent agent;
    float total_reward = 0.0, min_reward = 10.0, max_reward = 0.0;
    std::vector<float> state_vector = {0.0, 0.5, 1.7, 0.0, 0.0, 0.0}; // Initial state vector
    std::vector<float> next_state_vector = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // Next state vector
    std::vector<float> action_vector = {0.0, 0.0}; // Initial action vector
    float reward = 0.0; // Initial reward
    
    float get_reward(int thr, float delta_soc, float delta_soh, int target_thr, float speed);
};

#endif
