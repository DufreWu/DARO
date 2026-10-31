#ifndef TEGRASPEEDPDQNPOLICY_H_
#define TEGRASPEEDPDQNPOLICY_H_

#include "TegraSpeedPolicy.h"
#include "ApplicationMonitor.h"
#include "getNode_Tegra.h"
#include "MappingConsts.h"
#include "utilization.h"
#include <map>
#include <vector>
#include <iostream>
#include <fstream>
#include <torch/script.h>
#include "../agents/pdqn_agent.h"

#include <ros/ros.h>
#include <std_msgs/String.h>

const int MAPPING_SIZE = std::pow(NUM_OF_CORES, NUM_OF_APPLICATIONS);
const int ARM_FREQ_RANGE = 11;
const int DENVER_FREQ_RANGE = 11;
const int GPU_FREQ_RANGE = 13;
const int TOTAL_COMBINATIONS = ARM_FREQ_RANGE * DENVER_FREQ_RANGE * GPU_FREQ_RANGE;
const float PDQN_SPEED_MIN = 0.5f;
const float PDQN_SPEED_MAX = 5.0f;
const int STATE_SIZE = 3 + NUM_OF_CORES + NUM_OF_APPLICATIONS * 3;

class TegraSpeedPDQNPolicy : virtual public TegraSpeedPolicy {
  public:
    TegraSpeedPDQNPolicy(std::string log_dir, std::string model_dir, int freq_b, int freq_l, int freq_g, float speed);
    void run(int cycle);

  private:
    std::string model_dir;
    // Define CPU and GPU frequency lists
    std::vector<int> cpu_fre_list = {499200, 652800, 806400, 960000, 1113600, 1267200, 1420800, 1574400, 1728000, 1881600, 2035200};
    std::vector<int> gpu_fre_list = {229500, 318750, 408000, 497250, 586500, 675750, 765000, 854250, 943500, 1032750, 1122000, 1211250, 1300500};

    float target_speed;
    std::vector<int> freq_index_list = {0, 0, 0};  // ARM, Denver, GPU
    
    std::ofstream* csvPDQNFile;

    float get_reward(float mech_power, float comp_power, float speed, std::vector<float> thr, std::vector<float> qos);

    PDQNAgent agent;
    std::vector<float> state_vector = {
        0.0f,   // mech_power
        0.5f,   // comp_power
        1.7f,   // speed
        0.0f, 0.0f, 0.0f,                   // freq_index
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,       // mapping_index
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,       // throughput for 5 apps
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,       // QoS for 5 apps
    };

    // Next state vector (same dimension, initialized to zero)
    std::vector<float> next_state_vector = std::vector<float>(STATE_SIZE, 0.0f);
    int action = 0;
    std::vector<float> action_params = {0.0f, -0.0f};
    float reward = 0.0; // Initial reward
    bool pending_action = false;
    int pending_cycle = -1;
    bool inference_enable= false;
    ros::NodeHandle nh_private;
};

#endif
