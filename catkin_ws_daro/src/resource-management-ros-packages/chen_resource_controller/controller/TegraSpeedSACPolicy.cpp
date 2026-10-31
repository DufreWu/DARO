#include "TegraSpeedSACPolicy.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <set>
#include <vector>
#include <thread>
#include <utility> // For std::pair
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sstream> // For std::ostringstream
#include "CGroupUtils.h"
#include <torch/optim.h> 
#include <torch/nn/functional.h>

#include <ros/ros.h>
#include <std_msgs/String.h>


#define SPEED_MAX 5.0
#define SPEED_MIN 2.0

#define INFERENCE_ENABLE true

typedef std::chrono::high_resolution_clock Clock;

#define BLOCKWIFI

float TegraSpeedSACPolicy::get_reward(int thr, float delta_soc, float delta_soh, int target_thr, float speed) {
  float delta_soc_percentage = abs(delta_soc*100);
  float delta_soh_percentage = abs(delta_soh*100);
  float beta_1 = 550.0;
  float beta_2 = 500.0;
  float u = 1.0;

  if (thr >= target_thr || target_thr == 0)
      u = 1.0;
  else
      u = pow(static_cast<float>(thr) / std::max(target_thr, 1), 3.0f);

  float soc_d = std::abs(delta_soc_percentage * beta_1) / speed;
  float soh_d = std::abs(std::exp(beta_2*delta_soh_percentage)) / speed;

  float reward = u - soc_d - soh_d;

  std::cout << "Reward calculation: "
            // << ", mech_power=" << mech_power 
            // << ", comp_power=" << comp_power 
            << ", speed=" << speed 
            << ", thr=" << thr 
            << ", target_thr=" << target_thr 
            << ", u=" << u 
            << ", soc_d=" << soc_d
            << ", soh_d=" << soh_d
            << ", reward=" << reward
            << std::endl;

  return reward;
}

TegraSpeedSACPolicy::TegraSpeedSACPolicy(std::string log_dir, std::string model_dir, int freq_b, int freq_l, int freq_g, float speed) : TegraSpeedPolicy(log_dir, freq_b, freq_l, freq_g, speed),
clk_action_list(generate_clk_action_list()),
agent(6, 2) {
  this->model_dir = model_dir;
  if (INFERENCE_ENABLE == true) {
    agent.load_model(this->model_dir);
    std::cout << "Load SAC model" << std::endl;
  }

  this->csvSACFile = new std::ofstream((this->log_dir + "/sac_reward.csv").c_str());
  if (! this->csvSACFile->is_open()) {
    std::cout << "Cannot open SAC CSV file" << std::endl;
    exit(0);
  }
  *this->csvSACFile << "ITERATION,MEAN_REWARD,MIN_REWARD,MAX_REWARD" << std::endl;

  std::cout << "TegraSpeedSACPolicy initialized." << std::endl;
}

void TegraSpeedSACPolicy::run(int cycle){
  auto curr_time = Clock::now();
  auto curr_time_str = this->clock_to_s_str(curr_time);
  auto curr_time_sec = this->clock_to_s(curr_time);

  //update status
  std::vector<pid_t> newAppls = updateAttachedApplications(this->appl_monitor);
  std::vector<int> u_cpu = utilization->getCPUUtilization(); 
  int u_gpu = sensors->getGPUCurUtilization();
  this->sensors->updateSensing();

  int curr_freqs[3]; //DO NOTE 2 clusters 0=LITTLE and 1=big. GPU is not used.
  curr_freqs[MAP_LITTLE_ID] = sensors->getCPUCurScalFreq(TEGRA_LITTLE_CORE_ID);
  curr_freqs[MAP_BIG_ID] = sensors->getCPUCurScalFreq(TEGRA_BIG_CORE_ID);
  curr_freqs[MAP_GPU_ID] = sensors->getGPUCurFreq();
  
  //report current status
  if(newAppls.size()>0){    
    std::cout << "New applications: ";
    for(int i=0; i < newAppls.size(); i++)
      std::cout << newAppls[i] << " ";
    std::cout << std::endl;
  }  
  printAttachedApplications(this->appl_monitor);
    
  std::cout << "CPU usage: ";
  for(int j=0; j < u_cpu.size(); j++){
    std::cout << u_cpu[j] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "CPU Freqs (current): " << " - " << sensors->getCPUCurFreq(TEGRA_LITTLE_CORE_ID) << " - " << sensors->getCPUCurFreq(TEGRA_BIG_CORE_ID) << std::endl;        
  std::cout << "CPU Freqs (scaling): " << " - " << curr_freqs[MAP_LITTLE_ID] << " - " << curr_freqs[MAP_BIG_ID] << std::endl;        

  std::cout << "GPU usage: " << u_gpu << std::endl;
  std::cout << "GPU Freqs (current): " << " - " << sensors->getGPUCurFreq() << std::endl;

  float curr_mech_power = this->getCurrentMechW();
  float curr_comp_power = this->sensors->getVddInW() + this->sensors->getMuxW();
  // float curr_elec_W = (curr_mech_power/0.8 + curr_comp_power)/1000.0;
  float curr_elec_W = this->battery->getBatteryPower((curr_mech_power/0.8 + curr_comp_power)/1000.0);

  float curr_epd = curr_elec_W/this->speed;
  float curr_battery_soc = this->battery->updateBatterySOC(curr_elec_W, 1.0);
  float curr_battery_soh = this->battery->updateBatterySOH(curr_battery_soc, 25.0);
  float curr_distance = this->calDistance(this->speed, 1.0);

  if(this->csvSysFile){
    *this->csvSysFile << curr_time_str << "," << curr_time_sec << "," << cycle << "," << curr_comp_power << "," 
    << curr_freqs[MAP_BIG_ID] << "," << curr_freqs[MAP_LITTLE_ID] << "," << curr_freqs[MAP_GPU_ID] << "," ;
    for(int i=0; i<TEGRA_NUM_OF_CPU_CORES; i++)
      *this->csvSysFile << u_cpu[i] << ",";
    *this->csvSysFile << u_gpu << ",";
    *this->csvSysFile << this->sensors->getVddInW() << "," << this->sensors->getMuxW() << "," << this->sensors->getGpuW() << "," << this->sensors->getSocW() << "," 
                      << this->sensors->getCpuW() << "," << this->sensors->getDdrW() << "," << this->sensors->getWifiW()  << ","
                      << this->speed << "," << curr_mech_power << "," << this->exec_time << "," 
                      << curr_epd << "," << curr_distance << "," << curr_battery_soc << "," << curr_battery_soh;

    *this->csvSysFile << std::endl;
  }
  
  //run resource management policy
  std::set<pid_t> alreadyAnalyzedPid;
  //number of sequential and multithreaded applications per cluster
  std::vector<pid_t> st_appls_count[2];
  std::vector<pid_t> mt_appls_count[2];
  //events
  bool new_appl = false;
  bool ended_appl = false;

  //remove deatched applications info. DO NOTE; do it before updating with new appls 
  //since in the meanwhile I perform some other analysis on the currently running applications 
  
  //get the list of running applications to compute ended once by subtraction of sets
  for(int i=0; i < appl_monitor->nAttached; i++){
    pid_t curr_pid = appl_monitor->appls[i].pid;
    alreadyAnalyzedPid.insert(curr_pid);
  }
  //the iterator is not invalidated by the deletion!
  std::map<pid_t, TegraSpeedSACPolicy::appl_state>::iterator erase_it = appl_info.begin(); 
  while(erase_it != appl_info.end()){
    if (alreadyAnalyzedPid.find(erase_it->first) == alreadyAnalyzedPid.end()) {
      //actually erase data from the map
      erase_it = appl_info.erase(erase_it);      
      ended_appl = true;
    } else {
      ++erase_it;
    }   
  }

  //analyze running applications to update the local data structure and start analyzing 
  //the current situation
  for(int i=0; i < appl_monitor->nAttached; i++){
    //get data from shared memory
    pid_t curr_pid = appl_monitor->appls[i].pid;
    data_t temp_data = monitorRead(appl_monitor->appls[i].segmentId);
    //initialization of newly arrived applications
    if(appl_info.find(curr_pid) == appl_info.end()) {   
      //copy settings from the shared memory 
      appl_info[curr_pid].mapping = appl_monitor->appls[i].mapping;
      if(appl_info[curr_pid].mapping >= MAP_MAX_ID){
        std::cout << "Mapping not implemented yet" << std::endl;
        exit(0);      
      }
      appl_info[curr_pid].mt = (appl_monitor->appls[i].maxThreads >1);
      new_appl = true;
    } 
    
    //update curr throughput and corresponding requirement for ALL applications
    appl_info[curr_pid].thr = getCurrThroughput(&temp_data);
    appl_info[curr_pid].thr_ref = getReqThroughput(&temp_data);
    appl_info[curr_pid].unit_th_processing_time = getUnitThProcessingTime(&temp_data);

    //count how many sequential/multithreaded appls on big and  on LITTLE
    if(appl_info[curr_pid].mt)
      mt_appls_count[appl_info[curr_pid].mapping].push_back(curr_pid);
    else
      st_appls_count[appl_info[curr_pid].mapping].push_back(curr_pid);     

    std::cout << "appl[" << i << "]: " << appl_info[curr_pid].thr_ref << " " << appl_info[curr_pid].thr << " " << curr_freqs[appl_info[curr_pid].mapping] << " " << std::endl;
    // std::cout << "appl2: " << 1.0/appl_info[curr_pid].thr_ref << " " << 1.0/appl_info[curr_pid].thr << " " << temp_data.usleepTime << std::endl;
  }
 
  //In a second csv file we save the data about the applications. they have to be postprocessed to be attached to the other system file
  if(this->csvApplFile){
    *this->csvApplFile << curr_time_str << ","<< curr_time_sec << ","<< cycle;
    for(std::map<pid_t, TegraSpeedSACPolicy::appl_state>::iterator mapIt = appl_info.begin(); mapIt != appl_info.end(); mapIt++) {
      std::cout << "appl[" << mapIt->first << "]: " << mapIt->second.thr_ref << " " << mapIt->second.thr << " " << mapIt->second.mapping << " " << std::endl;
      *this->csvApplFile  << "," << mapIt->first << "," << mapIt->second.thr_ref << "," << mapIt->second.thr << "," << mapIt->second.mapping << "," << mapIt->second.unit_th_processing_time;  
    }
    *this->csvApplFile << std::endl;
  }

  // =========== SAC begin ===========
  float thr_ref = 0.0;
  if(appl_monitor->nAttached > 0) {
    pid_t critical_pid = appl_monitor->appls[0].pid;
    thr_ref = appl_info[critical_pid].thr_ref;
    this->next_state_vector = {this->cpu_freq_index, this->speed,
                              curr_mech_power/1000.0, curr_comp_power/1000.0, 
                              appl_info[critical_pid].thr, thr_ref};
  } else {
    // No applications running
    this->next_state_vector = {this->cpu_freq_index, this->speed,
                              curr_mech_power/1000.0, curr_comp_power/1000.0, 
                              0.0, 0.0};
  }

  std::cout << "1. Next state vector: " << this->next_state_vector << std::endl;               
  if (INFERENCE_ENABLE == false) {
    std::cout << "SAC training" << std::endl;
    agent.step(this->state_vector, this->action_vector, this->reward, this->next_state_vector, true);
    std::cout << "2. State vector: " << state_vector << std::endl;
    // int thr, float delta_soc, float delta_soh, int target_thr, float speed
    float curr_battery_delta_soc = this->battery->calBatteryDeltaSOC(curr_elec_W, 1.0);
    this->reward = get_reward(this->next_state_vector[4],   // current throughput
                              curr_battery_delta_soc,  
                              abs(curr_battery_soh - soh_prev), // delta soh
                              thr_ref,                      // required throughput
                              this->next_state_vector[1]);  // current speed
    soh_prev = curr_battery_soh; // Update previous SOH for next cycle
    std::cout << "3. Reward: " << reward << std::endl;
    // this->total_reward += this->reward;
    // this->max_reward = std::max(this->max_reward, this->reward);
    // this->min_reward = std::min(this->min_reward, this->reward);
    this->action_vector = this->agent.get_action(this->next_state_vector);
    this->target_speed = SPEED_MIN + ((this->action_vector[0] + 1.0f) / 2.0f) * (SPEED_MAX - SPEED_MIN);
    int freq_action = static_cast<int>(((this->action_vector[1] + 1.0f) / 2.0f) * (this->clk_action_list.size() - 1));
    this->cpu_freq_index = std::get<0>(clk_action_list[freq_action]);
    this->gpu_freq_index = std::get<1>(clk_action_list[freq_action]);
    std::cout << "4. Action: [" << this->action_vector 
              << "], Speed: " << this->speed 
              << ", CPU index: " << this->cpu_freq_index
              << ", GPU index: " << this->gpu_freq_index 
              << std::endl;

    *this->csvSACFile << cycle << "," << this->reward << "," << std::endl;
    this->state_vector = this->next_state_vector; // Update state vector for the next cycle
    if (cycle > 0 && cycle % 500 == 0) {
      // *this->csvSACFile << cycle << "," << this->total_reward/16.0 << ","
      //                    << this->min_reward << "," << this->max_reward << std::endl;
      // std::cout.flush();
      // this->csvSACFile->flush();
      // std::cout << "Checkpoint saved: " << cycle << ", reward: " << this->total_reward << std::endl;
      // this->total_reward = 0;
      // this->min_reward = 10;
      // this->max_reward = 0;
      agent.save_model(this->model_dir);
    }
  // } else if (cycle == 0 || new_appl || ended_appl) {
  } else if (cycle > 20) {
    auto t_end = Clock::now();
    this->action_vector = this->agent.get_action(this->next_state_vector);
    auto t_init = Clock::now();
    double inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_init).count();
    std::cout << "[SAC] Decision making time: " << inference_time << " ms" << std::endl;
    this->target_speed = SPEED_MIN + ((this->action_vector[0] + 1.0f) / 2.0f) * (SPEED_MAX - SPEED_MIN);
    int freq_action = static_cast<int>(((this->action_vector[1] + 1.0f) / 2.0f) * (this->clk_action_list.size() - 1));
    this->cpu_freq_index = std::get<0>(clk_action_list[freq_action]);
    this->gpu_freq_index = std::get<1>(clk_action_list[freq_action]);

    std::cout << "4. Action: [" << this->action_vector 
              << "], Speed: " << this->speed 
              << ", CPU index: " << this->cpu_freq_index
              << ", GPU index: " << this->gpu_freq_index 
              << std::endl;
  }
  // =========== SAC end ===========
  
  //intra-cluster mapping (allocation of cores for applications running on the same cluster)
  //since there are only 4 cores per cluster I have identified a set of possible situations 
  //and for each of them a fixed mapping solution is proposed   
  if(new_appl || ended_appl) {
    std::cout << "LITTLE " << st_appls_count[MAP_LITTLE_ID].size() << " " << mt_appls_count[MAP_LITTLE_ID].size() << std::endl;
    std::cout << "big    " << st_appls_count[MAP_BIG_ID].size() << " " << mt_appls_count[MAP_BIG_ID].size() << std::endl;
    //analyze LITTLE workload
    if(st_appls_count[MAP_LITTLE_ID].size()>0 && mt_appls_count[MAP_LITTLE_ID].size()==0) { //N sequential appls
      std::vector<int> core;
      core.push_back(0);
      core.push_back(3);      
      core.push_back(4);
      core.push_back(5);
      // std::cout << this->sensors->isCPUActive(3) << " " << this->sensors->isCPUActive(4) << " " << this->sensors->isCPUActive(5) << "\n";
      for(int i=0; i<st_appls_count[MAP_LITTLE_ID].size(); i++)    
        UpdateCpuSet(this->appl_monitor, st_appls_count[MAP_LITTLE_ID][i],core);
    } else if(st_appls_count[MAP_LITTLE_ID].size()==0 && mt_appls_count[MAP_LITTLE_ID].size()==1) { //1 multithreaded appl
      std::vector<int> core;
      core.push_back(0);
      core.push_back(3);
      core.push_back(4);
      core.push_back(5);
      UpdateCpuSet(this->appl_monitor, mt_appls_count[MAP_LITTLE_ID][0],core);    
    } else if(st_appls_count[MAP_LITTLE_ID].size()==1 && mt_appls_count[MAP_LITTLE_ID].size()==1) { //1 sequential appl + 1 multithreaded appl
      std::vector<int> core;
      core.push_back(0);
      UpdateCpuSet(this->appl_monitor, st_appls_count[MAP_LITTLE_ID][0],core);    
      core.clear();
      core.push_back(3);
      core.push_back(4);
      core.push_back(5);
      UpdateCpuSet(this->appl_monitor, mt_appls_count[MAP_LITTLE_ID][0],core);       
    } else if(st_appls_count[MAP_LITTLE_ID].size()==0 && mt_appls_count[MAP_LITTLE_ID].size()==2) { //2 multithreaded appls
      std::vector<int> core;
      core.push_back(0);
      core.push_back(3);
      UpdateCpuSet(this->appl_monitor, mt_appls_count[MAP_LITTLE_ID][0],core);    
      core.clear();
      core.push_back(4);
      core.push_back(5);
      UpdateCpuSet(this->appl_monitor, mt_appls_count[MAP_LITTLE_ID][1],core);       
    } else if(st_appls_count[MAP_LITTLE_ID].size()==2 && mt_appls_count[MAP_LITTLE_ID].size()==1) { //2 sequential appls + 1 multithreaded appl
      std::vector<int> core;
      core.push_back(0);
      core.push_back(3);
      UpdateCpuSet(this->appl_monitor, mt_appls_count[MAP_LITTLE_ID][0],core);    
      core.clear();
      core.push_back(4);
      core.push_back(5);
      for(int i=0; i<st_appls_count[MAP_LITTLE_ID].size(); i++)    
        UpdateCpuSet(this->appl_monitor, st_appls_count[MAP_LITTLE_ID][i],core);    
    } else if(st_appls_count[MAP_LITTLE_ID].size()>2 && mt_appls_count[MAP_LITTLE_ID].size()==1) { //N sequential appls + 1 multithreaded appl
      std::vector<int> core;
      core.push_back(0);
      core.push_back(3);
      core.push_back(4);
      core.push_back(5);
      UpdateCpuSet(this->appl_monitor, mt_appls_count[MAP_LITTLE_ID][0],core);    
      for(int i=0; i<st_appls_count[MAP_LITTLE_ID].size(); i++)    
        UpdateCpuSet(this->appl_monitor, st_appls_count[MAP_LITTLE_ID][i],core);
    } else if(st_appls_count[MAP_LITTLE_ID].size()!=0 || mt_appls_count[MAP_LITTLE_ID].size()!=0){
      std::cout << "LITTLE: not supported mapping config " << std::endl;
      // exit(0);
    }
    
    //analyze big workload:
    if(st_appls_count[MAP_BIG_ID].size()>0 && mt_appls_count[MAP_BIG_ID].size()==0) { //N sequential appls
      std::vector<int> core;
      core.push_back(1);
      core.push_back(2);
      for(int i=0; i<st_appls_count[MAP_BIG_ID].size(); i++)    
        UpdateCpuSet(this->appl_monitor, st_appls_count[MAP_BIG_ID][i],core);
    } else if(st_appls_count[MAP_BIG_ID].size()==0 && mt_appls_count[MAP_BIG_ID].size()==1) { //1 multithreaded appl
      std::vector<int> core;
      core.push_back(1);
      core.push_back(2);
      UpdateCpuSet(this->appl_monitor, mt_appls_count[MAP_BIG_ID][0],core);    
    } else if(st_appls_count[MAP_BIG_ID].size()!=0 || mt_appls_count[MAP_BIG_ID].size()!=0){
      std::cout << "big: not supported mapping config " << std::endl;
      // exit(0);
    }
  }

  this->setFreqLittle(cpu_fre_list[this->cpu_freq_index]);
  // this->setFreqGpu(gpu_fre_list[this->gpu_freq_index]);
  this->setSpeed(this->target_speed);

  this->exec_time = (this->clock_to_ns(Clock::now()) - this->clock_to_ns(curr_time))*1e-9;
}