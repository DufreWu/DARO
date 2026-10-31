#include "TegraSpeedConstantPolicy.h"
#include <iostream>
#include <chrono>
#include <set>
#include <vector>
#include <thread>
#include "CGroupUtils.h"

#include <ros/ros.h>
#include <std_msgs/String.h>

#define WAIT_FOR_NEIGHBOR_PERIOD_MS 200 //milliseconds
#define SPEED_MAX 5
#define SPEED_MIN 0.25
#define SPEED_STEP 0.25

#define BLOCKWIFI

typedef std::chrono::high_resolution_clock Clock;

TegraSpeedConstantPolicy::TegraSpeedConstantPolicy(std::string log_dir, int freq_b, int freq_l, int freq_g, float speed) : TegraSpeedPolicy(log_dir, freq_b, freq_l, freq_g, speed) {};

void TegraSpeedConstantPolicy::run(int cycle){
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
  
  float curr_battery_delta_soc = this->battery->calBatteryDeltaSOC(curr_elec_W, 1.0) * 100.0;
  float curr_epd = curr_elec_W/this->speed;
  float curr_battery_soc = this->battery->updateBatterySOC(curr_elec_W, 1.0);
  float curr_battery_soh = this->battery->updateBatterySOH(curr_battery_soc, 25.0);
  float curr_distance = this->calDistance(this->speed, 1.0);

  if(this->csvSysFile){
    float currpower = this->sensors->getBoardW();    

    *this->csvSysFile << curr_time_str << "," << curr_time_sec << "," << cycle << "," << currpower << "," 
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
  std::map<pid_t, TegraSpeedConstantPolicy::appl_state>::iterator erase_it = appl_info.begin(); 
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
    appl_info[curr_pid].name = appl_monitor->appls[i].name;
    appl_info[curr_pid].thr = getCurrThroughput(&temp_data);
    appl_info[curr_pid].thr_ref = getReqThroughput(&temp_data);
    appl_info[curr_pid].unit_th_processing_time = getUnitThProcessingTime(&temp_data);

    //count how many sequential/multithreaded appls on big and  on LITTLE
    if(appl_info[curr_pid].mt)
      mt_appls_count[appl_info[curr_pid].mapping].push_back(curr_pid);
    else
      st_appls_count[appl_info[curr_pid].mapping].push_back(curr_pid);        

    std::cout << "appl1: " << appl_info[curr_pid].thr_ref << " " << appl_info[curr_pid].thr << " " << curr_freqs[appl_info[curr_pid].mapping] << " " << std::endl;
    std::cout << "appl2: " << 1.0/appl_info[curr_pid].thr_ref << " " << 1.0/appl_info[curr_pid].thr << " " << temp_data.usleepTime << std::endl;
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
  this->exec_time = (this->clock_to_ns(Clock::now()) - this->clock_to_ns(curr_time))*1e-9;
}