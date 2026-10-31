#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <thread>
#include <set>
#include <iomanip>
#include <getopt.h>

#include "ApplicationMonitor.h"
#include "CGroupUtils.h"
#include "Policy.h"
#include "TegraPolicy.h"
#include "TegraSpeedPolicy.h"
#include "TegraSpeedHillClimbingPolicy.h"
#include "TegraSpeedPredictiveSearchPolicy.h"
#include "TegraSpeedModelBasedHillClimbingPolicy.h"
#include "TegraSpeedModelBasedLin3HillClimbingPolicy.h"
#include "TegraSpeedConstantPolicy.h"
#include "TegraSpeedConstantFreqInteractivePolicy.h"
#include "TegraSpeedAllConstsPolicy.h"
#include "TegraSpeedPDQNPolicy.h"
#include "TegraSpeedSACPolicy.h"
#include "TegraSpeedDQNPolicy.h"
#include "TegraSpeedPPOPolicy.h"

#include "InputArgs.h"
#include <ros/ros.h>

//#define PROFILING_CALLS

#define WAKEUP_PERIOD 1000 //milliseconds
// #define WAKEUP_PERIOD 500 //milliseconds

typedef std::chrono::high_resolution_clock Clock;

enum {
  NO_POLICY = 0,
  ONLY_SENSORS = 1,
  ODROID_POLICY = 2,
  ODROID_LOG_POLICY = 3,
  ODROID_PROF_POLICY = 4,
  TEGRA_POLICY = 5,
  TEGRA_LOG_POLICY = 6,
  RPI3_POLICY = 11,
  TEGRA_SPEED_LOG_POLICY = 13,
  TEGRA_SPEED_HILLCLIMBING_LOG_POLICY = 14,
  TEGRA_SPEED_CONSTANT_LOG_POLICY = 15,
  TEGRA_SPEED_MODELBASED_HILLCLIMBING_LOG_POLICY = 16,
  TEGRA_SPEED_MODELBASED_LIN3_HILLCLIMBING_LOG_POLICY = 17,
  TEGRA_SPEED_CONSTANT_FREQ_INTERACTIVE=18,
  TEGRA_SPEED_PREDICTIVE_SEARCH_LOG_POLICY = 19,
  TEGRA_SPEED_ALL_CONSTS_POLICY_LOG_POLICY = 20,
  TEGRA_SPEED_PDQN_POLICY=22,
  TEGRA_SPEED_SAC_POLICY=23,
  TEGRA_SPEED_DQN_POLICY=24,
  TEGRA_SPEED_PPO_POLICY=25,
  MAX_POLICY = 27,
};

void sig_handler (int sig, siginfo_t *info, void *extra);
bool end_loop = false; //states if the controller has to be terminated
ros::Publisher speed;

int main (int argc, char* argv[]) {
#ifdef PROFILING_CALLS
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#endif
  ros::init(argc, argv, "chen_resource_controller_node");

  ////////////////////////////////////////////////////////////////////////////////
  //parsing input arguments
  ////////////////////////////////////////////////////////////////////////////////

  MyArgs args = argparse::parse<MyArgs>(argc, argv);
  args.print();       // prints all variables 

  if(args.policy_id >= MAX_POLICY || args.policy_id < 0){
    std::cerr << std::endl << "USAGE: " << argv[0] << " [-p POLICY_ID]" << std::endl;
    std::cerr << std::endl << "Supported modes:" << std::endl;
    std::cerr << "5 - Tegra policy for fake multiboard exps" << std::endl;
    std::cerr << "6 - Tegra policy for fake multiboard exps + logs in CSV" << std::endl;
    std::cerr << "7 - Tegra Profiling policy for fake multiboard exps + logs in CSV" << std::endl;
    std::cerr << "8 - Tegra Nano policy for fake multiboard exps" << std::endl;
    std::cerr << "9 - Tegra Nano policy for fake multiboard exps + logs in CSV" << std::endl;
    std::cerr << "10- Tegra Nano Profiling policy for fake multiboard exps + logs in CSV" << std::endl;
    std::cerr << "12 - Tegra + Robot Speed policy for fake multiboard exps" << std::endl;
    std::cerr << "13 - Tegra + Robot Speed policy for fake multiboard exps + logs in CSV" << std::endl;
    std::cerr << "14 - Tegra + Robot Speed HillClimbing policy for fake multiboard exps + logs in CSV" << std::endl;
    std::cerr << "15 - Tegra + Robot Speed Constant policy for fake multiboard exps + logs in CSV" << std::endl;
    exit(EXIT_FAILURE);
  }

  //setup interrupt handler
  struct sigaction action;
  action.sa_flags = SA_SIGINFO;
  action.sa_sigaction = &sig_handler;
  if (sigaction(SIGINT, &action, NULL) == -1) {
    std::cout << "Error registering interrupt handler\n" << std::endl;
    exit(EXIT_FAILURE);
  }

  //set output precision
  std::cout << std::setprecision(10);
  std::cerr << std::setprecision(10);
  std::cout << args.policy_id << std::endl;
  
  //setup monitors and policies
  Policy* policy = NULL;
  if(args.policy_id == TEGRA_POLICY){
    policy = new TegraPolicy();
  } else if(args.policy_id == TEGRA_LOG_POLICY){
    policy = new TegraPolicy("sys.csv", "appls.csv");
  } else if(args.policy_id == TEGRA_SPEED_LOG_POLICY){
    policy = new TegraSpeedPolicy(args.log_dir, args.init_freq_b, args.init_freq_l, args.init_freq_g, args.init_speed);
  } else if(args.policy_id == TEGRA_SPEED_HILLCLIMBING_LOG_POLICY){
    policy = new TegraSpeedHillClimbingPolicy(args.log_dir, args.init_freq_b, args.init_freq_l, args.init_freq_g, args.init_speed);
  } else if(args.policy_id == TEGRA_SPEED_PREDICTIVE_SEARCH_LOG_POLICY){
    policy = new TegraSpeedPredictiveSearchPolicy(args.log_dir, args.model_dir, args.init_freq_b, args.init_freq_l, args.init_freq_g, args.init_speed);
  } else if(args.policy_id == TEGRA_SPEED_ALL_CONSTS_POLICY_LOG_POLICY){
    policy = new TegraSpeedAllConstsPolicy(args.log_dir, args.init_freq_b, args.init_freq_l, args.init_freq_g, args.init_speed);
  } else if(args.policy_id == TEGRA_SPEED_CONSTANT_LOG_POLICY){
    policy = new TegraSpeedConstantPolicy(args.log_dir, args.init_freq_b, args.init_freq_l, args.init_freq_g, args.init_speed);
  } else if(args.policy_id == TEGRA_SPEED_MODELBASED_HILLCLIMBING_LOG_POLICY){
    policy = new TegraSpeedModelBasedHillClimbingPolicy(args.log_dir, args.model_dir, args.init_freq_b, args.init_freq_l, args.init_freq_g, args.init_speed);
  } else if(args.policy_id == TEGRA_SPEED_MODELBASED_LIN3_HILLCLIMBING_LOG_POLICY){
    policy = new TegraSpeedModelBasedLin3HillClimbingPolicy(args.log_dir, args.model_dir, args.init_freq_b, args.init_freq_l, args.init_freq_g, args.init_speed);
  } else if(args.policy_id == TEGRA_SPEED_CONSTANT_FREQ_INTERACTIVE){
    policy = new TegraSpeedConstantFreqInteractivePolicy(args.log_dir, args.init_freq_b, args.init_freq_l, args.init_freq_g, args.init_speed);
  } else if(args.policy_id == TEGRA_SPEED_PDQN_POLICY) {
    policy = new TegraSpeedPDQNPolicy(args.log_dir, args.model_dir, args.init_freq_b, args.init_freq_l, args.init_freq_g, args.init_speed);
  } else if(args.policy_id == TEGRA_SPEED_SAC_POLICY) {
    policy = new TegraSpeedSACPolicy(args.log_dir, args.model_dir, args.init_freq_b, args.init_freq_l, args.init_freq_g, args.init_speed);
  } else if(args.policy_id == TEGRA_SPEED_DQN_POLICY) {
    policy = new TegraSpeedDQNPolicy(args.log_dir, args.model_dir, args.init_freq_b, args.init_freq_l, args.init_freq_g, args.init_speed);
  } else if(args.policy_id == TEGRA_SPEED_PPO_POLICY) {
    policy = new TegraSpeedPPOPolicy(args.log_dir, args.model_dir, args.init_freq_b, args.init_freq_l, args.init_freq_g, args.init_speed);
  } else {
    //unreachable else
    std::cout << "unreachable else"<< std::endl;
    exit(EXIT_FAILURE);
  }

#ifdef PROFILING_CALLS
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span1 = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
    std::cout << "[PROFILING] Init controller: " << time_span1.count() << " s." << std::endl;
#endif

  std::cout << "press Ctrl+C in order to stop monitoring" << std::endl;
  std::cout << "monitoring start..." << std::endl;
  int i = 0;

  while (!end_loop) {
    auto t = Clock::now();
    auto next_time = t + std::chrono::duration<long int, std::nano>((long int)(WAKEUP_PERIOD*1e6));
    std::cout << std::endl << "iteration " << i << std::endl;
#ifdef PROFILING_CALLS
    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
#endif
    //run policy
    policy->run(i);
#ifdef PROFILING_CALLS
    std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span2 = std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3);
    std::cout << "[PROFILING] Run policy: " << time_span2.count() << " s." << std::endl;
#endif

    //sleep for the specified period
    i++;
    std::this_thread::sleep_until(next_time);
  }

  delete policy;

  return 0;
}

//interrupt handler that finalizes the execution
void sig_handler (int sig, siginfo_t *info, void *extra) {
  if (sig == SIGINT) {
    std::cout << "\nmonitoring stop." << std::endl;
    end_loop = true;
  }
}
