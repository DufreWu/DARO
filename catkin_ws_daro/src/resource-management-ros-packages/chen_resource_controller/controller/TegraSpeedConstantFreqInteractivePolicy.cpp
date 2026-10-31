#include "TegraSpeedConstantFreqInteractivePolicy.h"
#include <iostream>
#include <chrono>
#include <set>
#include <vector>
#include <thread>
#include "CGroupUtils.h"

#include <ros/ros.h>
#include <std_msgs/String.h>

#define BLOCKWIFI
#define CONTROL_EVERY   2   // change freq if cycle % CONTROL_EVERY == 0

typedef std::chrono::high_resolution_clock Clock;

TegraSpeedConstantFreqInteractivePolicy::TegraSpeedConstantFreqInteractivePolicy(std::string log_dir, int freq_b, int freq_l, int freq_g, float speed) : TegraSpeedPolicy(log_dir, freq_b, freq_l, freq_g, speed) {

}

void TegraSpeedConstantFreqInteractivePolicy::run(int cycle){
  auto curr_time = Clock::now();
  auto curr_time_str = this->clock_to_s_str(curr_time);
  auto curr_time_sec = this->clock_to_s(curr_time);

  //update status
  std::vector<pid_t> newAppls = updateAttachedApplications(this->appl_monitor);
  std::vector<int> u_cpu = utilization->getCPUUtilization();
  int u_gpu = sensors->getGPUCurUtilization();
  this->sensors->updateSensing();

  int curr_freqs[3]; //DO NOTE 3 clusters 0=LITTLE, 1=big, 2=GPU.
  curr_freqs[MAP_LITTLE_ID] = sensors->getCPUCurScalFreq(TEGRA_LITTLE_CORE_ID);
  curr_freqs[MAP_BIG_ID] = sensors->getCPUCurScalFreq(TEGRA_BIG_CORE_ID);
  curr_freqs[MAP_GPU_ID] = sensors->getGPUCurFreq();
    
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
  float curr_elec_W = (curr_mech_power + curr_comp_power)/1000.0;
  
  float curr_battery_delta_soc = this->battery->calBatteryDeltaSOC(curr_elec_W, 1.0) * 100.0;
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
  std::map<pid_t, TegraSpeedConstantFreqInteractivePolicy::appl_state>::iterator erase_it = appl_info.begin(); 
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
      } else if(appl_info[curr_pid].mapping == MAP_LITTLE_ID || appl_info[curr_pid].mapping == MAP_BIG_ID){
        appl_info[curr_pid].mt = (appl_monitor->appls[i].maxThreads >1);
      }
      new_appl = true;
    }
    
    //update curr throughput and corresponding requirement for ALL applications
    appl_info[curr_pid].name = appl_monitor->appls[i].name;
    appl_info[curr_pid].thr = getCurrThroughput(&temp_data);
    appl_info[curr_pid].thr_ref = getReqThroughput(&temp_data);
    //count how many sequential/multithreaded appls on big and  on LITTLE
    if(appl_info[curr_pid].mt)
      mt_appls_count[appl_info[curr_pid].mapping].push_back(curr_pid);
    else if(appl_info[curr_pid].mapping != MAP_GPU_ID)
      st_appls_count[appl_info[curr_pid].mapping].push_back(curr_pid);  
  }
  // ===================== DVFS CONTROL =====================
  int target_freqs[3];
  target_freqs[MAP_LITTLE_ID] = curr_freqs[MAP_LITTLE_ID];
  target_freqs[MAP_BIG_ID] = curr_freqs[MAP_BIG_ID];
  target_freqs[MAP_GPU_ID] = curr_freqs[MAP_GPU_ID];    

  if(!new_appl && cycle > CONTROL_EVERY && cycle % CONTROL_EVERY == 0) {
    bool cluster_violate[3] = {false,false,false};
    bool cluster_all_safe[3] = {true,true,true};

    for (const auto& [pid, info] : appl_info) {
      if(info.thr < info.thr_ref) {
        cluster_violate[info.mapping] = true;
        cluster_all_safe[info.mapping] = false;
      } else if(info.thr <= info.thr_ref + 2) {
        cluster_all_safe[info.mapping] = false;
      }
    }

    for (int c=0; c<3; c++) {
      int step = (c == MAP_GPU_ID) ? TEGRA_GPU_FREQ_STEP : TEGRA_CPU_FREQ_STEP;

      if(cluster_violate[c]) {
        target_freqs[c] = curr_freqs[c] + step;
      } else if(cluster_all_safe[c]  && !appl_info.empty()) {
        target_freqs[c] = curr_freqs[c] - step;
      } else {
        target_freqs[c] = curr_freqs[c];
      }
    }

    // ===================== Clamp =====================
    if(target_freqs[MAP_LITTLE_ID] > TEGRA_LITTLE_MAX_FREQ)
        target_freqs[MAP_LITTLE_ID] = TEGRA_LITTLE_MAX_FREQ;
    if(target_freqs[MAP_LITTLE_ID] < TEGRA_LITTLE_MIN_FREQ)
        target_freqs[MAP_LITTLE_ID] = TEGRA_LITTLE_MIN_FREQ;

    if(target_freqs[MAP_BIG_ID] > TEGRA_BIG_MAX_FREQ)
        target_freqs[MAP_BIG_ID] = TEGRA_BIG_MAX_FREQ;
    if(target_freqs[MAP_BIG_ID] < TEGRA_BIG_MIN_FREQ)
        target_freqs[MAP_BIG_ID] = TEGRA_BIG_MIN_FREQ;

    if(target_freqs[MAP_GPU_ID] > TEGRA_GPU_MAX_FREQ)
        target_freqs[MAP_GPU_ID] = TEGRA_GPU_MAX_FREQ;
    if(target_freqs[MAP_GPU_ID] < TEGRA_GPU_MIN_FREQ)
        target_freqs[MAP_GPU_ID] = TEGRA_GPU_MIN_FREQ;
  } 
  
  // ===================== Write CSV =====================
  if(this->csvApplFile){
    *this->csvApplFile << curr_time_str << "," << curr_time_sec << ","<< cycle;
    std::vector<std::string> apps_by_index(NUM_OF_APPLICATIONS);

    for (const auto& [name, idx] : appls_index_map) {
        apps_by_index[idx] = name;
    }

    for (int idx = 0; idx < NUM_OF_APPLICATIONS; ++idx) {
        const std::string& app_name = apps_by_index[idx];

        bool found = false;
        for (const auto& [pid, info] : appl_info) {
            if (info.name == app_name) {
                *this->csvApplFile
                    << "," << app_name
                    << "," << info.thr_ref
                    << "," << info.thr
                    << "," << info.mapping;
                found = true;
                break;
            }
        }
        if (!found) {
            appls_to_cluster[idx] = -1;
            *this->csvApplFile
                << "," << app_name
                << "," << 0
                << "," << 0
                << "," << -1;
        }
    }

    *this->csvApplFile << std::endl;
  }
  
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

  if(target_freqs[MAP_LITTLE_ID] != curr_freqs[MAP_LITTLE_ID]){
    std::cout << "---> change freq on LITTLE: " << target_freqs[MAP_LITTLE_ID] << std::endl;
    this->setFreqLittle(target_freqs[MAP_LITTLE_ID]);
  }
  if(target_freqs[MAP_BIG_ID] != curr_freqs[MAP_BIG_ID]){
    std::cout << "---> change freq on big: " << target_freqs[MAP_BIG_ID] << std::endl;
    this->setFreqBig(target_freqs[MAP_BIG_ID]);
  }
  if(target_freqs[MAP_GPU_ID] != curr_freqs[MAP_GPU_ID]){
    std::cout << "---> change freq on gpu: " << target_freqs[MAP_GPU_ID] << std::endl;
    this->setFreqGpu(target_freqs[MAP_GPU_ID]);
  }

  this->exec_time = (this->clock_to_ns(Clock::now()) - this->clock_to_ns(curr_time))*1e-9;
}