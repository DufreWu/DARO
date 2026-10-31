#include "TegraSpeedPredictiveSearchPolicy.h"
#include <iostream>
#include <chrono>
#include <set>
#include <vector>
#include <thread>
#include "CGroupUtils.h"
#include "CSVRow.h"
#include <ros/ros.h>
#include <std_msgs/String.h>

#include <Eigen/Dense>

typedef std::chrono::high_resolution_clock Clock;

#define BLOCKWIFI
#define CONTROL_EVERY   3 // change freq if cycle % CONTROL_EVERY == 0

TegraSpeedPredictiveSearchPolicy::TegraSpeedPredictiveSearchPolicy(std::string log_dir, std::string model_dir, int freq_b, int freq_l, int freq_g, float speed) : TegraSpeedHillClimbingPolicy(log_dir, freq_b, freq_l, freq_g, speed) {
  std::cout << "TegraSpeedPredictiveSearchPolicy " << std::endl;
  this->model_dir = model_dir;
  this->load_linear3_model();

  this->csvModelFile = new std::ofstream((this->log_dir + "/model.csv").c_str());
  if (! this->csvModelFile->is_open()) {
    std::cout << "Cannot open Model CSV file" << std::endl;
    exit(0);
  }
  *this->csvModelFile << "TIME,UTIME,ITERATION,POWER,PRED_POWER,ERROR,ERROR_PERCENTAGE" << std::endl;

  std::string data_file_path = this->log_dir + "/cp_data_points.csv";
  this->data_file = new std::ofstream(data_file_path.c_str());
  if (! this->data_file->is_open()) {
    std::cout << "Cannot open data file at " << data_file_path.c_str() << std::endl;
    exit(0);
  }

  this->state = IDLE;

  // set denver and gpu governor to default
  this->sensors->setCPUGovernor(1, CPU_DEFAULT_GOVERNOR);
  this->sensors->setCPUGovernor(2, CPU_DEFAULT_GOVERNOR);
  this->sensors->setGPUGovernor(GPU_DEFAULT_GOVERNOR);
}

void TegraSpeedPredictiveSearchPolicy::run(int cycle){

  std::cout << "Cycle PEO " << cycle << std::endl;

  auto curr_time = Clock::now();
  auto curr_time_str = this->clock_to_s_str(curr_time);
  auto curr_time_sec = this->clock_to_s(curr_time);
  auto t_init_ns = this->clock_to_ns(Clock::now());

  //update status
  std::vector<pid_t> newAppls = updateAttachedApplications(this->appl_monitor);
  std::vector<int> u_cpu = utilization->getCPUUtilization(); 
  int u_gpu = sensors->getGPUCurUtilization();
  this->sensors->updateSensing();

  std::cout << "Cycle PEO 1" << cycle << std::endl;

  int curr_freqs[3]; //DO NOTE 3 clusters 0=LITTLE and 1=big. 2=GPU.
  curr_freqs[MAP_LITTLE_ID] = sensors->getCPUCurScalFreq(TEGRA_LITTLE_CORE_ID);
  curr_freqs[MAP_BIG_ID] = sensors->getCPUCurScalFreq(TEGRA_BIG_CORE_ID);
  curr_freqs[MAP_GPU_ID] = sensors->getGPUCurFreq();
  
  float curr_mech_power = this->getCurrentMechW();
  float curr_comp_power = this->sensors->getVddInW() + this->sensors->getMuxW();
  // float curr_elec_W = (curr_mech_power/0.8 + curr_comp_power)/1000.0;
  float curr_elec_W = this->battery->getBatteryPower((curr_mech_power/0.8 + curr_comp_power)/1000.0);

  float curr_battery_delta_soc = this->battery->calBatteryDeltaSOC(curr_elec_W, 1.0) * 100.0;
  float curr_epd = curr_elec_W/this->speed;
  float curr_battery_soc = this->battery->updateBatterySOC(curr_elec_W, 1.0);
  float curr_battery_soh = this->battery->updateBatterySOH(curr_battery_soc, 25.0);
  float curr_distance = this->calDistance(this->speed, 1.0);
  
  this->add_data_point(curr_freqs[MAP_BIG_ID], curr_freqs[MAP_LITTLE_ID], u_cpu, curr_comp_power);

  float estimated_comp_power = this->estimate_computation_power(curr_freqs[MAP_BIG_ID], curr_freqs[MAP_LITTLE_ID], u_cpu);
  float estimation_error = estimated_comp_power - curr_comp_power;
  float estimation_error_percentage = 100*abs(estimated_comp_power-curr_comp_power)/curr_comp_power;
  
  *this->csvModelFile  << curr_time_str << "," << curr_time_sec << "," << cycle << ","
                      << curr_freqs[MAP_BIG_ID] << "," << curr_freqs[MAP_LITTLE_ID] << "," << curr_comp_power << "," << estimated_comp_power << "," << estimation_error << "," << estimation_error_percentage 
                      << std::endl;

  if(this->csvSysFile){
    *this->csvSysFile << curr_time_str << "," << curr_time_sec << "," << cycle << "," << curr_comp_power << "," 
    << curr_freqs[MAP_BIG_ID] << "," << curr_freqs[MAP_LITTLE_ID] << "," << curr_freqs[MAP_GPU_ID] << "," ;
    for(int i=0; i<TEGRA_NUM_OF_CPU_CORES; i++)
      *this->csvSysFile << u_cpu[i] << ",";
    *this->csvSysFile << u_gpu << ",";
    *this->csvSysFile << this->sensors->getVddInW() << "," << this->sensors->getMuxW() << "," << this->sensors->getGpuW() << "," << this->sensors->getSocW() << "," 
                      << this->sensors->getCpuW() << "," << this->sensors->getDdrW() << "," << this->sensors->getWifiW() << "," 
                      << this->speed << "," << curr_mech_power << "," << this->exec_time << "," 
                      << curr_epd << "," << curr_distance << "," << curr_battery_soc << "," << curr_battery_soh;

    *this->csvSysFile << std::endl;
  }
  //"TIME,UTIME,ITERATION,STATE,NEXT_SPEED,NEXT_FREQ,NUM_NEIGHS,INDEX_NEIGHS"
  *this->csvHCFile << curr_time_str << "," << curr_time_sec << "," << cycle << "," << this->state << "," 
                    << curr_freqs[MAP_LITTLE_ID] << "," << this->speed << "," 
                    << this->getCompEnergyPerMeter(curr_comp_power, this->speed) << "," 
                    << this->getMechEnergyPerMeter(curr_mech_power, this->speed) << "," 
                    << this->getEnergyPerMeter(curr_comp_power, curr_mech_power, this->speed) << ","
                    << this->config_map_gamma 
                    << std::endl;

  t_init_ns = this->clock_to_ns(Clock::now());
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

  std::cout << "CYCLE-TIME (PREP): " << (this->clock_to_ns(Clock::now()) - t_init_ns) << " ns\n";
  t_init_ns = this->clock_to_ns(Clock::now());

  std::cout << "BEGIN: (state:" << this->state << ", speed:" << this->speed << ", freq:" << curr_freqs[MAP_LITTLE_ID] << ")" << std::endl;

  bool new_appl = false;
  bool ended_appl = false;

  //get the list of running applications to compute ended once by subtraction of sets
  for(int i=0; i < appl_monitor->nAttached; i++){
    pid_t curr_pid = appl_monitor->appls[i].pid;
    alreadyAnalyzedPid.insert(curr_pid);
  }

  //the iterator is not invalidated by the deletion!
  std::map<pid_t, TegraSpeedPredictiveSearchPolicy::appl_state>::iterator erase_it = appl_info.begin(); 
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
    //consider first app to identify the config (?!)
    pid_t critical_pid = appl_monitor->appls[0].pid;
    float th_error_max = -1;
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
      // appl_info[curr_pid].unit_th_processing_time = getUnitThProcessingTime(&temp_data);
      appl_info[curr_pid].unit_th_processing_time = 1.0/appl_info[curr_pid].thr;
      float th_error = (appl_info[curr_pid].unit_th_processing_time - (1/appl_info[curr_pid].thr_ref)) / (1/appl_info[curr_pid].thr_ref);
      if(th_error > 0 && th_error > th_error_max) 
        critical_pid = curr_pid;
    }
    
    // pid_t curr_pid = critical_pid;
    pid_t curr_pid = appl_monitor->appls[0].pid;

    float p_time_req_low = (1.0 / appl_info[curr_pid].thr_ref) * 0.6;
    float p_time_req_high = (1.0 / appl_info[curr_pid].thr_ref) * 1.0;
    float p_time_req_mid = (p_time_req_low+p_time_req_high)/2;
    float p_c, p_m, epd_c, epd_m, epd, p_time, s_factor, acc_en;
    this->config_map_gamma = 0.8*this->config_map_gamma + 0.01*0.2;
    float gamma = this->config_map_gamma;

    for(int f=TEGRA_LITTLE_MIN_FREQ; f<=TEGRA_LITTLE_MAX_FREQ; f+=TEGRA_CPU_FREQ_STEP) {
      p_c = this->estimate_computation_power(curr_freqs[MAP_BIG_ID], f, u_cpu);
      for(float s=SPEED_MIN; s<=SPEED_MAX;s+=SPEED_STEP) {
        int i = int((f - TEGRA_LITTLE_MIN_FREQ) / TEGRA_CPU_FREQ_STEP);
        int j = int((s - SPEED_MIN) / SPEED_STEP);
        p_m = this->getMechW(s);
        epd_c = p_c / s;
        epd_m = p_m / s;
        acc_en = abs(this->speed - s) * 3 * 1e3 / 10;
        epd = epd_m + epd_c + acc_en;
        s_factor = 1 - 0.1*(this->speed - s);
        
        p_time = appl_info[curr_pid].unit_th_processing_time * (1.0*curr_freqs[MAP_LITTLE_ID] / f) * s_factor;
        // std::cout << "CHECK (EPD): " << p_c << "," << p_m << "," << epd_c << "," << epd_m << "," << epd << "\n";
        this->config_map[i][j].freq = f;
        this->config_map[i][j].speed = s;
        this->config_map[i][j].t_process = p_time*gamma + this->config_map[i][j].t_process*(1-gamma);
        this->config_map[i][j].p_c = p_c;
        this->config_map[i][j].p_m = p_m;
        this->config_map[i][j].epd_c = epd_c;
        this->config_map[i][j].epd_m = epd_m;
        this->config_map[i][j].epd = epd;
      }
    }

    std::cout << "epd,thr_sat,tprocess,epd_c,epd_m,pc,pm\n";

    std::cout << "XXX";
    for(float s=SPEED_MIN; s<=SPEED_MAX;s+=SPEED_STEP) {
      std::cout << "|" << s;
    }
    std::cout << "\n";

    for(int f=TEGRA_LITTLE_MIN_FREQ; f<=TEGRA_LITTLE_MAX_FREQ; f+=TEGRA_CPU_FREQ_STEP) {
      std::cout << f;
      for(float s=SPEED_MIN; s<=SPEED_MAX;s+=SPEED_STEP) {
        int i = int((f - TEGRA_LITTLE_MIN_FREQ) / TEGRA_CPU_FREQ_STEP);
        int j = int((s - SPEED_MIN) / SPEED_STEP);
        std::string th_sat = "FALSE";
        if(this->config_map[i][j].t_process < p_time_req_high)
            th_sat = "TRUE";
        std::cout << "|" << int(this->config_map[i][j].epd) << "," << th_sat << "," 
                  << this->config_map[i][j].t_process << "," 
                  << int(this->config_map[i][j].epd_c) << "," << int(this->config_map[i][j].epd_m) << "," 
                  << int(this->config_map[i][j].p_c) << "," << int(this->config_map[i][j].p_m)
                  ;
      }
      std::cout << "\n";
    }

    std::cout << "CYCLE-TIME (SEARCH-FILL-MAP) :" << (this->clock_to_ns(Clock::now()) - t_init_ns) << " ns\n";

    float my_target_speed = this->speed;
    int my_target_freq = curr_freqs[MAP_LITTLE_ID];
    if(this->state == IDLE) {
      this->state = APPLY_BEST;

      /*my_target_freq = curr_freqs[MAP_LITTLE_ID] + TEGRA_CPU_FREQ_STEP;
      if(my_target_freq > TEGRA_LITTLE_MAX_FREQ) {
        my_target_freq = TEGRA_LITTLE_MIN_FREQ;
      }
      this->start_cycle = cycle;
      this->state = IDLE;*/

      if(estimation_error_percentage >= 20) {
        this->state = COLLECT_DATA;
        this->start_cycle = cycle;
      } else {
        float r = ((float) rand() / (RAND_MAX));
        if( r <= 0.99999) {
          this->state = APPLY_BEST;
        } else {
          this->state = RANDOM_CONFIG;
        }
      }
    } else if(this->state == COLLECT_DATA) {
      if(cycle - this->start_cycle < 20) {
        // JUST WAIT TO COLLECT ENOUGH NUMBER OF SAMPLES!
      } else {
        int i = int((curr_freqs[MAP_BIG_ID] - 499200) / TEGRA_CPU_FREQ_STEP);
        int j = int((curr_freqs[MAP_LITTLE_ID] - 499200) / TEGRA_CPU_FREQ_STEP);
        this->update_computation_model(i, j);
        this->state = IDLE;
      }
    } else if(this->state == APPLY_BEST) {
      std::cout << "p_time_req_low: " << p_time_req_low << "\n" << "p_time_req_high: " << p_time_req_high << "\n";

      float best_epd = 1e9;
      float best_speed = 0;
      float best_freq = 0;
      for(int f=TEGRA_LITTLE_MIN_FREQ; f<=TEGRA_LITTLE_MAX_FREQ; f+=TEGRA_CPU_FREQ_STEP) {
        for(float s=SPEED_MIN; s<=SPEED_MAX;s+=SPEED_STEP) {
          int i = int((f - 499200) / TEGRA_CPU_FREQ_STEP);
          int j = int((s - SPEED_MIN) / SPEED_STEP);
          if(this->config_map[i][j].t_process < p_time_req_high) {
            if(this->config_map[i][j].epd < best_epd) {
              best_epd = this->config_map[i][j].epd;
              best_speed = s;
              best_freq = f;
            }
          }
        }
      }

      for(int f=TEGRA_LITTLE_MIN_FREQ; f<=TEGRA_LITTLE_MAX_FREQ; f+=TEGRA_CPU_FREQ_STEP) {
        for(float s=SPEED_MIN; s<=SPEED_MAX;s+=SPEED_STEP) {
          int i = int((f - 499200) / TEGRA_CPU_FREQ_STEP);
          int j = int((s - SPEED_MIN) / SPEED_STEP);
          if(this->config_map[i][j].t_process < p_time_req_high) {
            if(this->config_map[i][j].epd - best_epd < 100 && f > best_freq && s > best_speed) {
              best_speed = s;
              best_freq = f;
            }
          }
        }
      }

      if(best_speed == 0) {
        my_target_freq = TEGRA_LITTLE_MAX_FREQ;
        my_target_speed = (SPEED_MIN + SPEED_MAX)/2;
      } else {
        for(int f=TEGRA_LITTLE_MIN_FREQ; f<=TEGRA_LITTLE_MAX_FREQ; f+=TEGRA_CPU_FREQ_STEP) {
          int i = int((f - 499200) / TEGRA_CPU_FREQ_STEP);
          int j = int((best_speed - SPEED_MIN) / SPEED_STEP);
          if(this->config_map[i][j].t_process < p_time_req_high) {  
            best_epd = this->config_map[i][j].epd;
            best_freq = f;
            break;
          }
        }
        if(best_speed == this->speed && best_freq == curr_freqs[MAP_LITTLE_ID]) {
          this->state = TOP;
        } else {
          my_target_freq = best_freq;
          my_target_speed = best_speed;
          this->state = IDLE;
        }
      }
    } else if(this->state == TOP) {
      float p_time_req_low_limit = (1.0 / appl_info[curr_pid].thr_ref) * 0.5;
      float p_time_req_high_limit = (1.0 / appl_info[curr_pid].thr_ref) * 1.0;
      bool time_violation = (appl_info[curr_pid].unit_th_processing_time > p_time_req_high_limit) || 
            (appl_info[curr_pid].unit_th_processing_time < p_time_req_low_limit);
      if(time_violation && this->config_map_gamma < 0.5) {
        this->config_map_gamma = 0.5;
      }

      if(time_violation || ((float) rand() / (RAND_MAX)) < 0.000001 || cycle < 20) {
        this->state = APPLY_BEST;
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
    
    // ===================== DNVER and GPU DVFS CONTROL ===================== 
    if(!new_appl && cycle > CONTROL_EVERY && cycle % CONTROL_EVERY == 0) {
      bool cluster_violate[3] = {false, false,false};
      bool cluster_all_safe[3] = {true, true,true};

      for (const auto& [pid, info] : appl_info) {
        if(info.thr < info.thr_ref) {
          cluster_violate[info.mapping] = true;
          cluster_all_safe[info.mapping] = false;
        } else if(info.thr <= info.thr_ref + 3) {
          cluster_all_safe[info.mapping] = false;
        }
      }

      for (int c=1; c<3; c++) {
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
      if(target_freqs[MAP_BIG_ID] > TEGRA_BIG_MAX_FREQ)
          target_freqs[MAP_BIG_ID] = TEGRA_BIG_MAX_FREQ;
      if(target_freqs[MAP_BIG_ID] < TEGRA_BIG_MIN_FREQ)
          target_freqs[MAP_BIG_ID] = TEGRA_BIG_MIN_FREQ;

      if(target_freqs[MAP_GPU_ID] > TEGRA_GPU_MAX_FREQ)
          target_freqs[MAP_GPU_ID] = TEGRA_GPU_MAX_FREQ;
      if(target_freqs[MAP_GPU_ID] < TEGRA_GPU_MIN_FREQ)
          target_freqs[MAP_GPU_ID] = TEGRA_GPU_MIN_FREQ;
    } 

    std::cout << "CYCLE-TIME (CTRL-ALGO): " << (this->clock_to_ns(Clock::now()) - t_init_ns) << " ns\n";
    t_init_ns = this->clock_to_ns(Clock::now());
    if(new_appl) {
      for(std::map<pid_t, TegraSpeedPredictiveSearchPolicy::appl_state>::iterator mapIt = appl_info.begin(); mapIt != appl_info.end(); mapIt++) {
        pid_t pid = mapIt->first;
        std::vector<int> core;
        if(mapIt->second.mapping == MAP_LITTLE_ID) {
          core.push_back(0);
          core.push_back(3);
          core.push_back(4);
          core.push_back(5);
          UpdateCpuSet(this->appl_monitor, pid, core); 
        } else if (mapIt->second.mapping == MAP_BIG_ID) {
          core.push_back(1);
          core.push_back(2);
          UpdateCpuSet(this->appl_monitor, pid, core); 
        }
        // *this->csvApplFile << "," << mapIt->first << "," << mapIt->second.thr_ref << "," << mapIt->second.thr << "," << mapIt->second.mapping << "," << mapIt->second.unit_th_processing_time;  
      }
    }
    std::cout << "CYCLE-TIME (APPLY-CONFIG1): " << (this->clock_to_ns(Clock::now()) - t_init_ns) << " ns\n";
    t_init_ns = this->clock_to_ns(Clock::now());
    
    this->setFreqBig(target_freqs[MAP_BIG_ID]);
    this->setFreqLittle(my_target_freq);
    this->setFreqGpu(target_freqs[MAP_GPU_ID]);

    std::cout << "CYCLE-TIME (APPLY-CONFIG-SETFREQ): " << (this->clock_to_ns(Clock::now()) - t_init_ns) << " ns\n";
    t_init_ns = this->clock_to_ns(Clock::now());

    this->setSpeed(my_target_speed);

    std::cout << "CYCLE-TIME (APPLY-CONFIG-SETSPEED): " << (this->clock_to_ns(Clock::now()) - t_init_ns) << " ns\n";
    t_init_ns = this->clock_to_ns(Clock::now());
  }

  std::cout << "END: (state:" << this->state << ", speed:" << this->speed << ", freq:" << curr_freqs[MAP_LITTLE_ID] << ")" << std::endl; 
  std::cout << std::endl;

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

  std::cout.flush();
  this->csvModelFile->flush();
  this->exec_time = (this->clock_to_ns(Clock::now()) - this->clock_to_ns(curr_time))*1e-9;

  std::cout << "CYCLE-TIME (FINISHING): " << (this->clock_to_ns(Clock::now()) - t_init_ns) << " ns\n";
  std::cout << "CYCLE-TIME (TOTAL): " << (this->clock_to_ns(Clock::now()) - this->clock_to_ns(curr_time)) << " ns\n";
}

void TegraSpeedPredictiveSearchPolicy::load_linear3_model() {
  std::string file_path = this->model_dir + "/params_linear3.csv";
  std::ifstream file(file_path);

  if (!file.is_open()) {
    std::cout << "Cannot open model file at " << file_path.c_str() << std::endl;
    exit(0);
  }

  std::cout << "Reading csv file " << file_path << "\n";
  CSVRow row;
  int idx = 0, i,j,k;
  while(file >> row)
  {
      if(idx > 0) {
        i = int((atoi(std::string(row[1]).c_str()) - 499200) / TEGRA_CPU_FREQ_STEP);
        j = int((atoi(std::string(row[2]).c_str()) - 499200) / TEGRA_CPU_FREQ_STEP);
        
        for(k=3;k<=8;k++) {
          this->models[i][j].norm_mu(0) = 0;
          this->models[i][j].norm_mu(k-2) = atof(std::string(row[k]).c_str());
        }
        for(k=9;k<=14;k++) {
          this->models[i][j].norm_scale(0) = 1.0;
          this->models[i][j].norm_scale(k-8) = atof(std::string(row[k]).c_str());
        }

        for(k=15;k<=20;k++) {
          this->models[i][j].coefs(k-14) = atof(std::string(row[k]).c_str());
        }
        this->models[i][j].coefs(0) = atof(std::string(row[k]).c_str());
      }
      idx++;
  }
  
  for(i=0;i<11;i++) {
     for(j=0;j<11;j++) {
      std::cout << i << "," << j << ":\n";
      for(int k=0;k<7;k++)
         std::cout << this->models[i][j].coefs(k) << ",";
      std::cout << "\n";

      for(int k=0;k<7;k++)
         std::cout << this->models[i][j].norm_mu(k) << ",";
      std::cout << "\n";
      for(int k=0;k<7;k++)
         std::cout << this->models[i][j].norm_scale(k) << ",";
      std::cout << "\n";
      std::cout << "\n";
    }
  }
}

float TegraSpeedPredictiveSearchPolicy::estimate_computation_power(int freq_b, int freq_l, std::vector<int> u) {
  auto t_init = Clock::now();
  auto t_init_ns = this->clock_to_ns(t_init);
  
  int i = int((freq_b - 499200) / TEGRA_CPU_FREQ_STEP);
  int j = int((freq_l - 499200) / TEGRA_CPU_FREQ_STEP);
  
  Eigen::Matrix<float, 7, 1> x;
  x(0) = 1.0;
  for(int k=1;k<7;k++) x(k) = u[k-1];
  x = (x - this->models[i][j].norm_mu);
  x = x.array() / this->models[i][j].norm_scale.array();
  
  auto y = x.dot(this->models[i][j].coefs);

  auto processing_time = this->clock_to_ns(Clock::now()) - t_init_ns;
  std::cout << "INFERENCE-TIME: " << processing_time << "\n";
  return y;
}

void TegraSpeedPredictiveSearchPolicy::add_data_point(int freq_b, int freq_l, std::vector<int> u, float comp_p) {
  int i = int((freq_b - 499200) / TEGRA_CPU_FREQ_STEP);
  int j = int((freq_l - 499200) / TEGRA_CPU_FREQ_STEP);
  TegraSpeedPredictiveSearchPolicy::data_point dp = {};
  dp.u[0] = 1.0;
  for(int k=1;k<7;k++) dp.u[k] = u[k-1];
  dp.power = comp_p;

  *this->data_file << freq_b << "," << freq_l << ",";
  for(int k=0;k<7;k++) *data_file << dp.u[k] << ",";
  *this->data_file << dp.power << "\n";

  this->data[i][j].push_back(dp);
  if (this->data[i][j].size() > 5) {
    this->update_computation_model(i, j);
    if(data[i][j].size() > 20)
      this->data[i][j].clear();
  }
}

bool TegraSpeedPredictiveSearchPolicy::update_computation_model(int mi, int mj) {
  // update model[mi][mj] based on data[mi][mj]
  auto t_init_ns_total = this->clock_to_ns(Clock::now());
  
  std::cout << "Updating the model...\n";

  // std::cout << "Training data:\n";
  // for(int k=0;k<this->data[mi][mj].size();k++) {
  //   for(int i=0;i<7;i++) std::cout << this->data[mi][mj][k].u[i] << ",";
  //   std::cout << this->data[mi][mj][k].power << "\n";
  // }

  auto t_init_ns = this->clock_to_ns(Clock::now());
  int N = data[mi][mj].size();

  std::cout << "TRAIN-SIZE: " << N << "\n";

  Eigen::MatrixXf X(N, 7);
  Eigen::VectorXf y(N);
  for(int k=0;k<this->data[mi][mj].size();k++) {
    y(k) = this->data[mi][mj][k].power;
    for(int i=0;i<7;i++) {
      X(k,i) = (this->data[mi][mj][k].u[i] - this->models[mi][mj].norm_mu(i)) / this->models[mi][mj].norm_scale(i);
    }
  }
  std::cout << "TRAIN-TIME (MATCONST+NORM): " << (this->clock_to_ns(Clock::now()) - t_init_ns) << " ns\n";

  t_init_ns = this->clock_to_ns(Clock::now());
  Eigen::Matrix<float, 7, 7> A;
  Eigen::Matrix<float, 7, 1> coefs;
  Eigen::Matrix<float, 7, 1> b;
  Eigen::Matrix<float, 7, 7> I = Eigen::MatrixXf::Identity(7,7);
  
  A = X.transpose()*X + 0.001*I;
  b = X.transpose() * y;
  std::cout << "TRAIN-TIME (MATMUL): " << (this->clock_to_ns(Clock::now()) - t_init_ns) << " ns\n";

  t_init_ns = this->clock_to_ns(Clock::now());
  coefs = A.	householderQr().solve(b);
  std::cout << "TRAIN-TIME (householderQr): " << (this->clock_to_ns(Clock::now()) - t_init_ns) << " ns\n";

  
  
  // coefs = A.inverse() * X.transpose() * y;
  // std::cout << "TIME (INV+MATMUL): " << (this->clock_to_ns(Clock::now()) - t_init_ns) << " ns\n";

  // std::cout << "Coefs curr:\n" << this->models[mi][mj].coefs << "\n";
  // std::cout << "Coefs found on this:\n" << coefs << "\n";
  // Eigen::VectorXf y_pred(N);
  // y_pred = X * coefs;
  // std::cout << "y:\n" << y << "\n";
  // std::cout << "y_pred:\n" << y_pred << "\n";
  t_init_ns = this->clock_to_ns(Clock::now());
  this->models[mi][mj].coefs = 0.5 * this->models[mi][mj].coefs + 0.5 * coefs;
  std::cout << "TRAIN-TIME (MODEL-UPDATE): " << (this->clock_to_ns(Clock::now()) - t_init_ns) << " ns\n";
  
  // std::cout << "Coefs new:\n" << coefs << "\n";
  // y_pred = X * coefs;
  // std::cout << "y_pred (new):\n" << y_pred << "\n";
  this->save_model();

  auto processing_time = this->clock_to_ns(Clock::now()) - t_init_ns_total;
  std::cout << "TRAIN-TIME (TOTAL-TRAINING): " << processing_time << "\n";

  return true;
}

void TegraSpeedPredictiveSearchPolicy::save_model() {
  std::string model_file_path = this->log_dir + "/latest_model.csv"; 
  std::ofstream* model_file = new std::ofstream(model_file_path.c_str());
  if (! model_file->is_open()) {
    std::cout << "Cannot open speed file at " << model_file_path.c_str() << std::endl;
    exit(0);
  }

  *model_file << "i,freq_b,freq_l,std_scaler_mean_util0,std_scaler_mean_util1,std_scaler_mean_util2,std_scaler_mean_util3,std_scaler_mean_util4,std_scaler_mean_util5,std_scaler_scale_util0,std_scaler_scale_util1,std_scaler_scale_util2,std_scaler_scale_util3,std_scaler_scale_util4,std_scaler_scale_util5,lin_reg_coef_util0,lin_reg_coef_util1,lin_reg_coef_util2,lin_reg_coef_util3,lin_reg_coef_util4,lin_reg_coef_util5,lin_reg_intercept\n";
  for(int fb=499200; fb<=TEGRA_LITTLE_MAX_FREQ; fb+=TEGRA_CPU_FREQ_STEP) {
    for(int fl=499200; fl<=TEGRA_LITTLE_MAX_FREQ; fl+=TEGRA_CPU_FREQ_STEP) {
      int i = int((fb - 499200) / TEGRA_CPU_FREQ_STEP);
      int j = int((fl - 499200) / TEGRA_CPU_FREQ_STEP);
      *model_file << (i+j) << "," << fb << "," << fl << ",";
      for(int k=1;k<7;k++) *model_file << this->models[i][j].norm_mu[k] << ",";
      for(int k=1;k<7;k++) *model_file << this->models[i][j].norm_scale[k] << ",";
      for(int k=1;k<7;k++) *model_file << this->models[i][j].coefs[k] << ","; 
      *model_file << this->models[i][j].coefs[0] << "\n";
    }
  }
  model_file->close();
}