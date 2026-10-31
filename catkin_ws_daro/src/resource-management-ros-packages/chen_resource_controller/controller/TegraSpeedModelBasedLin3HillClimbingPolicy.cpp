#include "TegraSpeedModelBasedLin3HillClimbingPolicy.h"
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

TegraSpeedModelBasedLin3HillClimbingPolicy::TegraSpeedModelBasedLin3HillClimbingPolicy(std::string log_dir, std::string model_dir, int freq_b, int freq_l, int freq_g, float speed) : TegraSpeedHillClimbingPolicy(log_dir, freq_b, freq_l, freq_g, speed){
  this->model_dir = model_dir;
  this->load_linear3_model();
}

void TegraSpeedModelBasedLin3HillClimbingPolicy::run(int cycle){

  std::cout << "Cycle MBLIN3HC " << cycle << std::endl;

  //update status
  std::vector<pid_t> newAppls = updateAttachedApplications(this->appl_monitor);
  std::vector<int> u = utilization->getCPUUtilization();  
  this->sensors->updateSensing();

  int curr_freqs[2]; //DO NOTE 2 clusters 0=LITTLE and 1=big. GPU is not used.
  curr_freqs[MAP_LITTLE_ID] = sensors->getCPUCurScalFreq(TEGRA_LITTLE_CORE_ID);
  curr_freqs[MAP_BIG_ID] = sensors->getCPUCurScalFreq(TEGRA_BIG_CORE_ID);
  
  auto curr_time = Clock::now();
  auto curr_time_str = this->clock_to_s_str(curr_time);
  auto curr_time_sec = this->clock_to_s(curr_time);

  float curr_comp_power = this->sensors->getBoardW();    
  float curr_mech_power = this->sensors->getVddInW() + this->sensors->getMuxW();
  
  if(this->csvSysFile){
    *this->csvSysFile << curr_time_str << "," << curr_time_sec << "," << cycle << "," << curr_comp_power << "," << curr_freqs[MAP_BIG_ID] << "," << curr_freqs[MAP_LITTLE_ID] << "," << this->speed << ",";
    for(int i=0; i<TEGRA_NUM_OF_CPU_CORES; i++)
      *this->csvSysFile << u[i] << ",";
    
    *this->csvSysFile << this->sensors->getVddInW() << "," << this->sensors->getMuxW() << "," << this->sensors->getGpuW() << "," << this->sensors->getSocW() << "," 
                      << this->sensors->getCpuW() << "," << this->sensors->getDdrW() << "," << this->sensors->getWifiW() << "," 
                      << this->speed << "," << curr_mech_power << "," << curr_comp_power;

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
  int target_freq;

  if(appl_monitor->nAttached > 1) {
    std::cout << "Only supporting one application for now! found: " << appl_monitor->nAttached << std::endl;
    return ;
  }

  std::cout << "BEGIN: (state:" << this->state << ", speed:" << this->speed << ", freq:" << curr_freqs[MAP_LITTLE_ID] << ")" << std::endl;
  this->print_explore_state();

  //analyze running applications to update the local data structure and start analyzing 
  //the current situation
  if(appl_monitor->nAttached == 1) {
    //get data from shared memory
    pid_t curr_pid = appl_monitor->appls[0].pid;
    data_t temp_data = monitorRead(appl_monitor->appls[0].segmentId);
    
    //initialization of newly arrived applications
    if(appl_info.find(curr_pid) == appl_info.end()) {   
      //copy settings from the shared memory 
      appl_info[curr_pid].mapping = appl_monitor->appls[0].mapping;
      appl_info[curr_pid].mt = (appl_monitor->appls[0].maxThreads > 1);
    }
    
    //update curr throughput and corresponding requirement for ALL applications
    appl_info[curr_pid].thr = getCurrThroughput(&temp_data);
    appl_info[curr_pid].thr_ref = getReqThroughput(&temp_data);
    
    float my_target_speed = this->speed;
    int my_target_freq = curr_freqs[MAP_LITTLE_ID];

    float estimated_comp_power = this->estimate_computation_power(curr_freqs[MAP_BIG_ID], curr_freqs[MAP_LITTLE_ID], u);
    std::cout << "Error comp power: " << (estimated_comp_power - curr_comp_power) << "("<<100*(estimated_comp_power-curr_comp_power)/curr_comp_power<<"%)" << " estimate: " << estimated_comp_power << " actual: " << curr_comp_power << "\n"; 
    if(this->state == IDLE) {
      // let the application run for one cycle with current config for porfiling 
      this->reset_explore_state(this->speed, curr_freqs[MAP_LITTLE_ID], -1, -1, -1);
      this->state = EXPLORE;
    } else if(this->state == EXPLORE) {
      
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
        my_target_freq = this->explore_state.center_obs.freq;
        for(int i=0;i<this->explore_state.count_neighbors;i++) {
          if(this->explore_state.neighbors_obs[i].freq > my_target_freq) {
            my_target_freq = this->explore_state.neighbors_obs[i].freq;
          }
        }
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
      if(appl_info[curr_pid].thr + 1 < appl_info[curr_pid].thr_ref || appl_info[curr_pid].thr - 1 > appl_info[curr_pid].thr_ref) {
        this->state = IDLE;
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
    

    // we have only one single thread app for now!
    std::vector<int> core;
    if(appl_info[curr_pid].mapping == MAP_LITTLE_ID) {
      core.push_back(0);
    } else if (appl_info[curr_pid].mapping == MAP_BIG_ID) {
      core.push_back(1);
    }
    UpdateCpuSet(this->appl_monitor, curr_pid, core);
    
    this->setFreqBig(my_target_freq);
    this->setFreqLittle(my_target_freq);
    this->setSpeed(my_target_speed);

    std::cout << "appl1: " << appl_info[curr_pid].thr_ref << " " << appl_info[curr_pid].thr << " " << curr_freqs[appl_info[curr_pid].mapping] << " " << my_target_freq << std::endl;
    std::cout << "appl2: " << 1.0/appl_info[curr_pid].thr_ref << " " << 1.0/appl_info[curr_pid].thr << " " << temp_data.usleepTime << std::endl;
  }

  this->print_explore_state();
  std::cout << "END: (state:" << this->state << ", speed:" << this->speed << ", freq:" << curr_freqs[MAP_LITTLE_ID] << ")" << std::endl; 
  std::cout << std::endl;

  //In a second csv file we save the data about the applications. they have to be postprocessed to be attached to the other system file
  if(this->csvApplFile){
    *this->csvApplFile << curr_time_str << "," << curr_time_sec << "," << cycle;
    for(std::map<pid_t, TegraSpeedPolicy::appl_state>::iterator mapIt = appl_info.begin(); mapIt != appl_info.end(); mapIt++) {
      *this->csvApplFile << "," << mapIt->first << "," << mapIt->second.thr_ref << "," << mapIt->second.thr << "," << mapIt->second.mapping;  
    }
    *this->csvApplFile << std::endl;
  }

  std::cout.flush();
}

void TegraSpeedModelBasedLin3HillClimbingPolicy::load_linear3_model() {
  // std::string file_path = this->model_dir + "/params_linear1.csv";
  // std::ifstream file(file_path);
  // std::cout << "Reading csv file " << file_path << "\n";
  // CSVRow row;
  // int idx = 0, i,j;
  // while(file >> row)
  // {
  //     if(idx > 0) {
  //       i = int((atoi(std::string(row[1]).c_str()) - 499200) / TEGRA_CPU_FREQ_STEP);
  //       j = int((atoi(std::string(row[2]).c_str()) - 499200) / TEGRA_CPU_FREQ_STEP);
  //       this->linear_model[i][j][0] = atof(std::string(row[3]).c_str());
  //       this->linear_model[i][j][1] = atof(std::string(row[4]).c_str());
  //     }
  //     idx++;
  // }

  // for(i=0;i<11;i++) {
  //   for(j=0;j<11;j++) {
  //     std::cout << i << "," << j << "," << this->linear_model[i][j][0] << "," << this->linear_model[i][j][1] << "\n";
  //   }
  // }

}

float TegraSpeedModelBasedLin3HillClimbingPolicy::estimate_computation_power(int freq_b, int freq_l, std::vector<int> u) {
  int i = int((freq_b - 499200) / TEGRA_CPU_FREQ_STEP);
  int j = int((freq_l - 499200) / TEGRA_CPU_FREQ_STEP);
  float a = this->linear_model[i][j][0];
  float b = this->linear_model[i][j][1];
  float u_avg = 0;


  for(auto x: u) u_avg += x/u.size();
  std::cout <<std::setprecision(2) << "estimate_computation_power! freqs: " << freq_b << "," << freq_l << " coefs: " << a << "," << b << " " << u_avg << "\n";
  return a * u_avg + b;
}


