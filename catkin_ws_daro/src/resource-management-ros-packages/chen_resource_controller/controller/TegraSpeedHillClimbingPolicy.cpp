#include "TegraSpeedHillClimbingPolicy.h"
#include <iostream>
#include <chrono>
#include <set>
#include <vector>
#include <thread>
#include "CGroupUtils.h"

#include <ros/ros.h>
#include <std_msgs/String.h>

typedef std::chrono::high_resolution_clock Clock;

#define BLOCKWIFI
#define CONTROL_EVERY 3 // change freq if cycle % CONTROL_EVERY == 0

TegraSpeedHillClimbingPolicy::TegraSpeedHillClimbingPolicy(std::string log_dir, int freq_b, int freq_l, int freq_g, float speed) : TegraSpeedPolicy(log_dir, freq_b, freq_l, freq_g, speed) {
  this->state = IDLE;
  this->init_speed = speed;
  this->init_freq = freq_l;

  this->csvHCFile = new std::ofstream(this->log_dir + "/hc.csv");
  if (! this->csvHCFile->is_open()) {
    std::cout << "Cannot open HC CSV file" << std::endl;
    exit(0);
  }
  *this->csvHCFile << "TIME,UTIME,ITERATION,STATE,FREQ,SPEED,COMP_EpM,MECH_EpM,EpM" << std::endl;
}

void TegraSpeedHillClimbingPolicy::run(int cycle){

  std::cout << "Cycle HC " << cycle << std::endl;

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

  float curr_comp_power = this->sensors->getVddInW() + this->sensors->getMuxW();  
  float curr_mech_power = this->getCurrentMechW();
  
  if(this->csvSysFile){
    *this->csvSysFile << curr_time_str << "," << curr_time_sec << "," << cycle << "," << curr_comp_power << "," 
    << curr_freqs[MAP_BIG_ID] << "," << curr_freqs[MAP_LITTLE_ID] << "," << curr_freqs[MAP_GPU_ID] << "," ;
    for(int i=0; i<TEGRA_NUM_OF_CPU_CORES; i++)
      *this->csvSysFile << u_cpu[i] << ",";
    *this->csvSysFile << u_gpu << ",";
    *this->csvSysFile << this->sensors->getVddInW() << "," << this->sensors->getMuxW() << "," << this->sensors->getGpuW() << "," << this->sensors->getSocW() << "," 
                      << this->sensors->getCpuW() << "," << this->sensors->getDdrW() << "," << this->sensors->getWifiW() << "," 
                      << this->speed << "," << curr_mech_power << "," << this->exec_time;

    *this->csvSysFile << std::endl;
  }
  //"TIME,UTIME,ITERATION,STATE,NEXT_SPEED,NEXT_FREQ,NUM_NEIGHS,INDEX_NEIGHS"
  *this->csvHCFile << curr_time_str << "," << curr_time_sec << "," << cycle << "," << this->state << "," 
                    << curr_freqs[MAP_LITTLE_ID] << "," << this->speed << "," 
                    << this->getCompEnergyPerMeter(curr_comp_power, this->speed) << "," 
                    << this->getMechEnergyPerMeter(curr_mech_power, this->speed) << "," 
                    << this->getEnergyPerMeter(curr_comp_power, curr_mech_power, this->speed)
                    << std::endl;


  //run resource management policy
  std::set<pid_t> alreadyAnalyzedPid;
  int target_freq;
  int target_freqs[3];
  target_freqs[MAP_LITTLE_ID] = TEGRA_LITTLE_MIN_FREQ;
  target_freqs[MAP_BIG_ID] = TEGRA_BIG_MIN_FREQ;
  target_freqs[MAP_GPU_ID] = TEGRA_GPU_MIN_FREQ;

  if(appl_monitor->nAttached > 1) {
    std::cout << "Found: " << appl_monitor->nAttached << "apps. Only considering the first one." << std::endl;
  }

  std::cout << "BEGIN: (state:" << this->state << ", speed:" << this->speed << ", freq:" << curr_freqs[MAP_LITTLE_ID] << ")" << std::endl;
  this->print_explore_state();
  
  bool new_appl = false;
  bool ended_appl = false;

  //get the list of running applications to compute ended once by subtraction of sets
  for(int i=0; i < appl_monitor->nAttached; i++){
    pid_t curr_pid = appl_monitor->appls[i].pid;
    alreadyAnalyzedPid.insert(curr_pid);
  }

  //the iterator is not invalidated by the deletion!
  std::map<pid_t, TegraSpeedPolicy::appl_state>::iterator erase_it = appl_info.begin(); 
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
  if(appl_monitor->nAttached > 0) {
    for(int i=0; i < appl_monitor->nAttached; i++){
      //get data from shared memory
      pid_t curr_pid = appl_monitor->appls[i].pid;
      data_t temp_data = monitorRead(appl_monitor->appls[i].segmentId);
      std::cout << "Updating appl_info for: " << curr_pid << "\n";
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

      if(cycle > 3*CONTROL_EVERY && cycle % CONTROL_EVERY == 0){
        if (appl_info[curr_pid].mapping == MAP_GPU_ID) {
          target_freqs[MAP_GPU_ID] = curr_freqs[MAP_GPU_ID];
          if (appl_info[curr_pid].thr < appl_info[curr_pid].thr_ref - 2) {
            target_freqs[MAP_GPU_ID] = curr_freqs[MAP_GPU_ID] + TEGRA_GPU_FREQ_STEP;
            if (target_freqs[MAP_GPU_ID] > TEGRA_GPU_MAX_FREQ) {
              target_freqs[MAP_GPU_ID] = TEGRA_GPU_MAX_FREQ;
            }
          }
          if(appl_info[curr_pid].thr > appl_info[curr_pid].thr_ref + 2) {
            target_freqs[MAP_GPU_ID] = curr_freqs[MAP_GPU_ID] - TEGRA_GPU_FREQ_STEP;
            if (target_freqs[MAP_GPU_ID] < TEGRA_GPU_MIN_FREQ) {
              target_freqs[MAP_GPU_ID] = TEGRA_GPU_MIN_FREQ;
            }
          }
          this->setFreqGpu(target_freqs[MAP_GPU_ID]);
        }
        std::cout << "Change GPU Freq. to " << target_freqs[MAP_GPU_ID] << std::endl;
      }
    }
    if(appl_monitor->nAttached <= 1) {
      this->setFreqGpu(TEGRA_GPU_MIN_FREQ);
    }
    
    //consider first app to identify the config (?!)
    pid_t curr_pid = appl_monitor->appls[0].pid;
    std::cout << "curr_pid: " << curr_pid << "\n";
    float my_target_speed = this->speed;
    int my_target_freq = curr_freqs[MAP_LITTLE_ID];
    if(this->state == IDLE) {
      // let the application run for one cycle with current config for porfiling 
      this->reset_explore_state(this->speed, curr_freqs[MAP_LITTLE_ID], -1, -1, -1);
      this->state = EXPLORE;
    } else if(this->state == MAX_FREQ) {
      my_target_speed = this->speed;
      my_target_freq = TEGRA_LITTLE_MAX_FREQ;
      this->state = IDLE;
    } else if(this->state == EXPLORE) {
      // save profiled run for the center
      if(this->explore_state.index == 0) {
        this->explore_state.center_obs.thr = appl_info[curr_pid].thr;
        this->explore_state.center_obs.comp_power = curr_comp_power;
        this->explore_state.center_obs.mech_power = curr_mech_power;
      } else {
        this->explore_state.neighbors_obs[this->explore_state.index-1].thr = appl_info[curr_pid].thr;
        this->explore_state.neighbors_obs[this->explore_state.index-1].comp_power = curr_comp_power;
        this->explore_state.neighbors_obs[this->explore_state.index-1].mech_power = curr_mech_power;
      }

      my_target_speed = this->explore_state.neighbors_obs[this->explore_state.index].speed;
      my_target_freq = this->explore_state.neighbors_obs[this->explore_state.index].freq;

      this->explore_state.index++;
      if(this->explore_state.index == this->explore_state.count_neighbors) {
        this->state = APPLY_BEST;
      }
    } else if(this->state == APPLY_BEST) {
      this->explore_state.neighbors_obs[this->explore_state.index-1].thr = appl_info[curr_pid].thr;
      this->explore_state.neighbors_obs[this->explore_state.index-1].comp_power = curr_comp_power;
      this->explore_state.neighbors_obs[this->explore_state.index-1].mech_power = curr_mech_power;

      // Find those neighbors that satisfy required throughput
      float nei_thr;
      std::vector<int> valid_neigh_ids;
      for(int i=0;i<this->explore_state.count_neighbors;i++) {
        nei_thr = this->explore_state.neighbors_obs[i].thr;  
        if(nei_thr + 1 >= appl_info[curr_pid].thr_ref) {
          valid_neigh_ids.push_back(i);
        }  
      }

      bool center_provides_ref_thr = (this->explore_state.center_obs.thr + 1 >= appl_info[curr_pid].thr_ref);
      bool neigh_provides_ref_thr = (valid_neigh_ids.size() > 0);

      if(!center_provides_ref_thr && !neigh_provides_ref_thr) {
        my_target_freq = TEGRA_LITTLE_MAX_FREQ;
        my_target_speed = this->explore_state.center_obs.speed;
        /*my_target_freq = this->explore_state.center_obs.freq;
        for(int i=0;i<this->explore_state.count_neighbors;i++) {
          if(this->explore_state.neighbors_obs[i].freq > my_target_freq) {
            my_target_freq = this->explore_state.neighbors_obs[i].freq;
          }
        }*/
        this->state = IDLE;
      } else if(center_provides_ref_thr && !neigh_provides_ref_thr) {
        my_target_freq = this->explore_state.center_obs.freq;
        my_target_speed = this->explore_state.center_obs.speed;
        this->state = TOP;        
      } else if (!center_provides_ref_thr && neigh_provides_ref_thr) {
        int i_best = -1;
        float energy_best = 1e12;
        float nei_energy;
        for(int k=0;k<valid_neigh_ids.size();k++) {
          int i = valid_neigh_ids[k];
          nei_energy = this->getEnergyPerMeter(this->explore_state.neighbors_obs[i].comp_power, this->explore_state.neighbors_obs[i].mech_power, this->explore_state.neighbors_obs[i].speed);
          if(nei_energy < energy_best) {
            energy_best = nei_energy;
            i_best = i;
          }  
        }
        my_target_speed = this->explore_state.neighbors_obs[i_best].speed;
        my_target_freq = this->explore_state.neighbors_obs[i_best].freq;
        this->state = IDLE; 
      } else if (center_provides_ref_thr && neigh_provides_ref_thr) {
        int i_best = -1;
        float energy_best = this->getEnergyPerMeter(this->explore_state.center_obs.comp_power, this->explore_state.center_obs.mech_power, this->explore_state.center_obs.speed);
        float nei_energy;
        for(int k=0;k<valid_neigh_ids.size();k++) {
          int i = valid_neigh_ids[k];
          nei_energy = this->getEnergyPerMeter(this->explore_state.neighbors_obs[i].comp_power, this->explore_state.neighbors_obs[i].mech_power, this->explore_state.neighbors_obs[i].speed);
          if(nei_energy < energy_best) {
            energy_best = nei_energy;
            i_best = i;
          }  
        }
        if(i_best == -1) {
          my_target_freq = this->explore_state.center_obs.freq;
          my_target_speed = this->explore_state.center_obs.speed;
          this->state = TOP; 
        } else {
          my_target_speed = this->explore_state.neighbors_obs[i_best].speed;
          my_target_freq = this->explore_state.neighbors_obs[i_best].freq;
          this->state = IDLE;
        } 
      }
    } else if(this->state == TOP) {
      float p_time_req_low = (1.0 / appl_info[curr_pid].thr_ref) * 0.3;
      float p_time_req_high = (1.0 / appl_info[curr_pid].thr_ref) * 1.2;
      float p_time_req_mid = (p_time_req_low+p_time_req_high)/2;
      bool time_violation = (appl_info[curr_pid].unit_th_processing_time > p_time_req_high);
      if(time_violation) {
        this->state = MAX_FREQ;
      }
    }

    if(my_target_freq < TEGRA_LITTLE_MIN_FREQ)
      my_target_freq = TEGRA_LITTLE_MIN_FREQ;
    if((my_target_freq-TEGRA_LITTLE_MIN_FREQ)%TEGRA_CPU_FREQ_STEP != 0) //round up frequency to next step
      my_target_freq = ((my_target_freq-TEGRA_LITTLE_MIN_FREQ) / TEGRA_CPU_FREQ_STEP) * TEGRA_CPU_FREQ_STEP + TEGRA_CPU_FREQ_STEP + TEGRA_LITTLE_MIN_FREQ;
    if(my_target_freq > TEGRA_LITTLE_MAX_FREQ)
      my_target_freq = TEGRA_LITTLE_MAX_FREQ;
    
    if(my_target_speed > SPEED_MAX)
      my_target_speed = SPEED_MAX;
    if(my_target_speed < SPEED_MIN)
      my_target_speed = SPEED_MIN;
    std::cout << "targets: " << my_target_freq << " " << my_target_speed << "\n";

    if(new_appl) {
      // we have only one single thread app for now!
      std::vector<int> core;
      if(appl_info[curr_pid].mapping == MAP_LITTLE_ID) {
        core.push_back(0);
      } else if (appl_info[curr_pid].mapping == MAP_BIG_ID) {
        core.push_back(1);
      }
      UpdateCpuSet(this->appl_monitor, curr_pid, core);
    }
    
    this->setFreqBig(my_target_freq);
    this->setFreqLittle(my_target_freq);
    this->setSpeed(my_target_speed);
}

  this->print_explore_state();
  std::cout << "END: (state:" << this->state << ", speed:" << this->speed << ", freq:" << curr_freqs[MAP_LITTLE_ID] << ")" << std::endl; 
  std::cout << std::endl;

  //In a second csv file we save the data about the applications. they have to be postprocessed to be attached to the other system file
  if(this->csvApplFile){
    *this->csvApplFile << curr_time_str << "," << curr_time_sec << "," << cycle;
    for(std::map<pid_t, TegraSpeedPolicy::appl_state>::iterator mapIt = appl_info.begin(); mapIt != appl_info.end(); mapIt++) {
      *this->csvApplFile << "," << mapIt->first << "," << mapIt->second.thr_ref << "," << mapIt->second.thr << "," << mapIt->second.mapping << "," << mapIt->second.unit_th_processing_time;  
    }
    *this->csvApplFile << std::endl;
  }

  std::cout.flush();
  this->exec_time = (this->clock_to_ns(Clock::now()) - this->clock_to_ns(curr_time))*1e-9;
}

void TegraSpeedHillClimbingPolicy::print_explore_state() {
  std::cout << "BEGIN EXP_STATE: " << std::endl; 

  std::cout << "Center: (sp:" << this->explore_state.center_obs.speed 
          << ", freq:" << this->explore_state.center_obs.freq
          << ", thr:" << this->explore_state.center_obs.thr
          << ", mech_power:" << this->explore_state.center_obs.mech_power
          << ", comp_power:" << this->explore_state.center_obs.comp_power
          << ")" << std::endl;
  std::cout << "Neighbor index: " << this->explore_state.index << std::endl;
  std::cout << "Neighbor count: " << this->explore_state.count_neighbors << std::endl;
  for(int i=0; i<this->explore_state.count_neighbors;i++) {
    std::cout << "Neighbor " << i << " : (sp:" << this->explore_state.neighbors_obs[i].speed 
        << ", freq:" << this->explore_state.neighbors_obs[i].freq
        << ", thr:" << this->explore_state.neighbors_obs[i].thr
        << ", mech_power:" << this->explore_state.neighbors_obs[i].mech_power
        << ", comp_power:" << this->explore_state.neighbors_obs[i].comp_power
        << ")" << std::endl;
  }
  std::cout << "END EXP_STATE " << std::endl; 
}

void TegraSpeedHillClimbingPolicy::reset_explore_state(float speed_center, int freq_center, float thr_center, float mech_power_center, float comp_power_center) {
  this->explore_state.center_obs.speed = speed_center;
  this->explore_state.center_obs.freq = freq_center;
  this->explore_state.center_obs.thr = thr_center;
  this->explore_state.center_obs.mech_power = mech_power_center;
  this->explore_state.center_obs.comp_power = comp_power_center;

  this->explore_state.index = 0;
  
  int offset[2][4] = {{0, 0, 1, -1}, {1,-1, 0, 0}};
  this->explore_state.count_neighbors = 0;
  
  int n_freq;
  float n_speed;
  for(int i=0; i<4;i++) {
    n_freq = freq_center + TEGRA_CPU_FREQ_STEP * offset[0][i];
    n_speed = speed_center + SPEED_STEP * offset[1][i];
    if(n_freq <= TEGRA_LITTLE_MAX_FREQ && n_freq >= TEGRA_LITTLE_MIN_FREQ 
        && n_speed <= SPEED_MAX && n_speed >= SPEED_MIN) {
        this->explore_state.neighbors_obs[this->explore_state.count_neighbors].speed = n_speed;
        this->explore_state.neighbors_obs[this->explore_state.count_neighbors].freq = n_freq;
        this->explore_state.neighbors_obs[this->explore_state.count_neighbors].thr = -1;
        this->explore_state.neighbors_obs[this->explore_state.count_neighbors].mech_power = -1;
        this->explore_state.neighbors_obs[this->explore_state.count_neighbors].comp_power = -1;
        this->explore_state.count_neighbors++;
    }
  }
}

std::vector<std::pair<float, int> > TegraSpeedHillClimbingPolicy::get_neighbors(float speed, int freq) {
  std::vector<std::pair<float, int> > sp_fr_cfg;
  int x[2][8] = {
    {0, 0, 1, 1, 1,-1,-1},
    {1,-1, 0,-1, 1, 0, 1}
  };
      
  int n_freq;
  float n_speed;
  for(int i=0;i<8;i++) {
    n_freq = freq + TEGRA_CPU_FREQ_STEP * x[0][i];
    n_speed = speed + SPEED_STEP * x[1][i];
    if(n_freq <= TEGRA_LITTLE_MAX_FREQ && n_freq >= TEGRA_LITTLE_MIN_FREQ 
        && n_speed <= SPEED_MAX && n_speed >= SPEED_MIN) {
      sp_fr_cfg.push_back(std::pair<float, int>(n_speed, n_freq));
    }
  }
  return sp_fr_cfg;
}

