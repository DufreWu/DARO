#ifndef TEGRASPEEDPREDICTIVESEARCHPOLICY_H_
#define TEGRASPEEDPREDICTIVESEARCHPOLICY_H_


#include "TegraSpeedHillClimbingPolicy.h"
#include "ApplicationMonitor.h"
#include "getNode_Tegra.h"
#include "MappingConsts.h"
#include "utilization.h"
#include <map>
#include <iostream>
#include <fstream>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <Eigen/Dense>

#define SPEED_MAX 5
#define SPEED_MIN 0.25
#define SPEED_STEP 0.25
#define N_SPEEDS 20
#define N_FREQS 11

class TegraSpeedPredictiveSearchPolicy: public TegraSpeedHillClimbingPolicy{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    enum SearchState {
      IDLE = 0,
      FIND_BEST = 1,
      APPLY_BEST = 2,
      COLLECT_DATA = 3,
      UPDATE_MODEL = 4,
      RANDOM_CONFIG = 5,
      TOP = 6
    };

    TegraSpeedPredictiveSearchPolicy(std::string log_dir, std::string model_dir, int freq_b, int freq_l, int freq_g, float speed);
    void run(int cycle);
    void add_data_point(int freq_b, int freq_l, std::vector<int> u, float comp_p);
    bool update_computation_model(int i, int j);

    std::ofstream* csvModelFile;

    typedef struct{
      Eigen::Matrix<float, 7, 1> coefs;
      Eigen::Matrix<float, 7, 1> norm_mu;
      Eigen::Matrix<float, 7, 1> norm_scale;
    } linear_model;
    linear_model models[11][11];
    

    typedef struct{
      float u[7];
      float power;
    } data_point;
    std::vector<data_point> data[11][11];

    typedef struct{
      int freq;
      float speed;
      float t_process = 0;
      float p_m = 0, p_c = 0, epd_c = 0, epd_m = 0, epd = 0;
    } config_info;
    config_info config_map[N_FREQS][N_SPEEDS];
    float config_map_gamma = 1;

    std::ofstream* data_file;
    void save_model();

  private:
    std::string model_dir;
    void load_linear3_model();
    float estimate_computation_power(int freq_b, int freq_l, std::vector<int> u);
    int start_cycle;
};

#endif
