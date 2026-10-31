#include "TegraSpeedPDQNPolicy.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <set>
#include <vector>
#include <thread>
#include <utility>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sstream> 
#include "CGroupUtils.h"

#include <ros/ros.h>
#include <std_msgs/String.h>

typedef std::chrono::high_resolution_clock Clock;

TegraSpeedPDQNPolicy::TegraSpeedPDQNPolicy(
  std::string log_dir, 
  std::string model_dir, 
  int freq_b, 
  int freq_l, 
  int freq_g, 
  float speed) 
  : nh_private("~"), TegraSpeedPolicy(log_dir, freq_b, freq_l, freq_g, speed), agent(STATE_SIZE, MAPPING_SIZE, 2)
{
  this->model_dir = model_dir;

  auto t1 = Clock::now();
  agent.load_model(this->model_dir);
  auto load_time_ms = (this->clock_to_ns(Clock::now()) - this->clock_to_ns(t1))*1e-6;
  std::cout << "[PDQN] Load model, using " << load_time_ms << "ms" << std::endl;

  this->csvPDQNFile = new std::ofstream((this->log_dir + "/pdqn_reward.csv").c_str());
  if (!this->csvPDQNFile->is_open()) {
    std::cout << "Cannot open PDQN CSV file" << std::endl;
    exit(EXIT_FAILURE);
  }

  *this->csvPDQNFile << "ITERATION,REWARD" << std::endl;
  this->target_speed = speed;

  // this->nh_private.param<bool>("inference_enable", this->inference_enable, true);
  this->inference_enable = true;
  std::cout << "[TegraSpeedPDQNPolicy()] Inference enable: " << this->inference_enable << std::endl;
}

// ==========================================================
// Reward function
// ==========================================================
float TegraSpeedPDQNPolicy::get_reward(
  float mech_power, 
  float comp_power, 
  float speed, 
  std::vector<float> thr, 
  std::vector<float> qos) 
{
  float epd = (mech_power + comp_power) / std::max(speed, 1e-3f);
  float beta = 3.0f;
  float sys_crit = 1.0f;
  int active_apps = 0;

  for (size_t i = 0; i < thr.size() && i < qos.size(); ++i) {
    if (qos[i] < 1e-3f) continue; // Skip if throughput is negligible
    float ratio = std::min(1.0f, thr[i] / std::max(qos[i], 1e-3f));
    sys_crit *= ratio;
    active_apps++;
  }

  if (active_apps > 0) {
    sys_crit = std::pow(sys_crit, 1.0f / active_apps);
  } else {
    sys_crit = 1.0f;
  }

  float reward = sys_crit + beta / std::max(epd, 1e-3f);
  reward = std::max(reward, 0.0f);

  std::cout << "Reward calculation: mech_power=" << mech_power 
            << ", comp_power=" << comp_power 
            << ", speed=" << speed 
            << ", epd=" << epd 
            << ", sys_crit=" << sys_crit
            << ", reward=" << reward
            << std::endl;

  return reward;
}

// ==========================================================
// Main PDQN cycle
// ==========================================================
void TegraSpeedPDQNPolicy::run(int cycle){
  auto curr_time = Clock::now();
  auto curr_time_str = this->clock_to_s_str(curr_time);
  auto curr_time_sec = this->clock_to_s(curr_time);

  // ===================== System sensing =====================
  std::cout << " | Inference Enabled: " << (this->inference_enable ? "Yes" : "No") << std::endl;
  std::vector<int> u_cpu = utilization->getCPUUtilization(); 
  int u_gpu = sensors->getGPUCurUtilization();
  this->sensors->updateSensing();

  int curr_freqs[3];
  curr_freqs[MAP_LITTLE_ID] = sensors->getCPUCurScalFreq(TEGRA_LITTLE_CORE_ID);
  curr_freqs[MAP_BIG_ID] = sensors->getCPUCurScalFreq(TEGRA_BIG_CORE_ID);
  curr_freqs[MAP_GPU_ID] = sensors->getGPUCurFreq();
    
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

  auto sys_info_time_ms = (this->clock_to_ns(Clock::now()) - this->clock_to_ns(curr_time))*1e-6;
  std::cout << "[PDQN] Sys info time: " << sys_info_time_ms << " ms" << std::endl;
  
  // ===================== Application analysis =====================
  auto time_appl_analysis_begin = Clock::now();
  std::vector<pid_t> newAppls = updateAttachedApplications(this->appl_monitor);
  bool new_appl = false;
  bool ended_appl = false;

  std::set<pid_t> running;
  for(int i=0; i < appl_monitor->nAttached; i++){
    pid_t curr_pid = appl_monitor->appls[i].pid;
    running.insert(curr_pid);
  }

  for (auto it = appl_info.begin(); it != appl_info.end();) {
    if (running.count(it->first) == 0) {
      it = appl_info.erase(it);
      ended_appl = true;
    } else {
      ++it;
    }
  }

  for(int i=0; i < appl_monitor->nAttached; i++){
    pid_t curr_pid = appl_monitor->appls[i].pid;
    data_t temp_data = monitorRead(appl_monitor->appls[i].segmentId);

    if (appl_info.count(curr_pid) == 0) {
      new_appl = true;
    }

    appl_info[curr_pid].name = appl_monitor->appls[i].name;
    appl_info[curr_pid].thr = getCurrThroughput(&temp_data);
    appl_info[curr_pid].thr_ref = getReqThroughput(&temp_data);
    appl_info[curr_pid].unit_th_processing_time = getUnitThProcessingTime(&temp_data);    
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

  auto appl_info_time_ms = (this->clock_to_ns(Clock::now()) - this->clock_to_ns(time_appl_analysis_begin))*1e-6;
  std::cout << "[PDQN] Appl info time: " << appl_info_time_ms << " ms" << std::endl;

  // ===================== PDQN State =====================
  std::vector<float> thr(NUM_OF_APPLICATIONS, 0.0f);
  std::vector<float> qos(NUM_OF_APPLICATIONS, 0.0f);

  for (int i = 0; i < appl_monitor->nAttached; i++) {
      pid_t pid = appl_monitor->appls[i].pid;
      const std::string& name = appl_monitor->appls[i].name;

      auto it = appls_index_map.find(name);
      if (it == appls_index_map.end()) continue;

      int idx = it->second;
      thr[idx] = appl_info[pid].thr;
      qos[idx] = appl_info[pid].thr_ref;
  }

  this->next_state_vector[0] = curr_mech_power / 1000.0f / 20.479f;
  this->next_state_vector[1] = curr_comp_power / 1000.0f / 8.0f;
  this->next_state_vector[2] = (this->speed - PDQN_SPEED_MIN) / (PDQN_SPEED_MAX - PDQN_SPEED_MIN);

  this->next_state_vector[3] = freq_index_list[0] / float(ARM_FREQ_RANGE);
  this->next_state_vector[4] = freq_index_list[1] / float(DENVER_FREQ_RANGE);
  this->next_state_vector[5] = freq_index_list[2] / float(GPU_FREQ_RANGE);

  // Ensure exactly 5 values in thr/qos
  for (int i = 0; i < NUM_OF_APPLICATIONS; i++) {
    if (qos[i] > 1e-3f && appls_to_cluster[i] >= 0) {
      next_state_vector[6 + i] = appls_to_cluster[i] / float(NUM_OF_CORES - 1);

      thr[i] = thr[i] / qos[i]; // Cap throughput to QoS requirement
      qos[i] = 1.0f;
      next_state_vector[11 + i] = thr[i];
      next_state_vector[16 + i] = qos[i];
    } else {
      next_state_vector[6 + i] = -1.0f;
      next_state_vector[11 + i] = 0.0f;
      next_state_vector[16 + i] = 0.0f;
    }
  }

  std::cout << "1. Next state vector: " << this->next_state_vector << std::endl;

  if (this->inference_enable == false) {
    agent.step(this->state_vector, 
              this->action, 
              this->action_params, 
              this->reward, 
              this->next_state_vector, 
              0, 
              this->inference_enable);
    std::cout << "2. State vector: " << state_vector << std::endl;                 

    this->reward = get_reward(curr_mech_power / 1000.0,   // mechanical power
                              curr_comp_power / 1000.0,   // computational power
                              this->speed,   // speed
                              thr,                          // current throughput
                              qos);                         // current speed
    std::cout << "3. Reward: " << reward << std::endl;
  }

  auto time_inference_begin = Clock::now();
  if (this->inference_enable) {
    if (new_appl || ended_appl || cycle == 0) {// Run in inference (no training)
      auto [act_idx, act_params] = agent.get_action(this->next_state_vector, false);
      this->action = act_idx;
      this->action_params = act_params;
      pending_action = true;
      pending_cycle = cycle;
    } else if (pending_action && cycle == pending_cycle + 1) {
      // auto [act_idx, act_params] = agent.get_action(this->next_state_vector, false);
      // this->action = act_idx;
      // this->action_params = act_params;

      pending_action = false;
    }
  } else {
    // Run in training mode
    auto [act_idx, act_params] = agent.get_action(this->next_state_vector, true);
    this->action = act_idx;
    this->action_params = act_params;
  }
  auto inference_time_ms = (this->clock_to_ns(Clock::now()) - this->clock_to_ns(time_inference_begin))*1e-6;
  std::cout << "[PDQN] Inference completed in " << inference_time_ms << " ms"
          << " | action=" << this->action
          << " | params=[" << this->action_params[0] << ", " << this->action_params[1] << "]"
          << std::endl;

  // ===================== Decode PDQN outputs =====================
  auto time_decide_begin = Clock::now();
  // ----- Decode per-application cluster mapping -----
  int mapping_action = this->action;

  if (mapping_action < 0 || mapping_action >= MAPPING_SIZE) {
    std::cerr << "[PDQN] Invalid mapping action: "
              << mapping_action << std::endl;
    return;
  }

  for (int app = 0; app < NUM_OF_APPLICATIONS; app++) {
    appls_to_cluster[app] = mapping_action % NUM_OF_CORES;
    mapping_action /= NUM_OF_CORES;
    if (qos[app] <= 1e-3f) {
      appls_to_cluster[app] = -1; // Default to -1 if QoS is negligible
    }
  }

  // ----- Decode frequency action -----
  float freqs_normalized = (this->action_params[0] + 1.0f) / 2.0f;
  int freqs_action = static_cast<int>(freqs_normalized * (TOTAL_COMBINATIONS - 1));

  int gpu_idx = freqs_action % GPU_FREQ_RANGE;
  int temp = freqs_action / GPU_FREQ_RANGE;
  int denver_idx = temp % DENVER_FREQ_RANGE;
  int arm_idx = temp / ARM_FREQ_RANGE;

  // ----- Decode speed -----
  this->target_speed = PDQN_SPEED_MIN + ((this->action_params[1] + 1.0f) / 2.0f) * (PDQN_SPEED_MAX - PDQN_SPEED_MIN);

  // ----- Safety check -----
  if (this->inference_enable) {
    if (std::find(appls_to_cluster.begin(), appls_to_cluster.end(), MAP_BIG_ID) == appls_to_cluster.end()) {
      denver_idx = 0; // Limit Denver freq if no big app
    } 
    if (std::find(appls_to_cluster.begin(), appls_to_cluster.end(), MAP_GPU_ID) == appls_to_cluster.end()) {
      gpu_idx = 0;    // Limit GPU freq if no GPU app
    } 
    if (std::find(appls_to_cluster.begin(), appls_to_cluster.end(), MAP_LITTLE_ID) == appls_to_cluster.end()) {
      if (std::find(appls_to_cluster.begin(), appls_to_cluster.end(), MAP_GPU_ID) == appls_to_cluster.end()) {
        if (std::find(appls_to_cluster.begin(), appls_to_cluster.end(), MAP_BIG_ID) == appls_to_cluster.end()) {
          arm_idx = 0;    // Limit ARM freq if no little app and no GPU app
        }
      }
    }
  }

  this->freq_index_list = {arm_idx, denver_idx, gpu_idx};

  // ----- Debug output -----
  std::cout << "[PDQN] Action Decoded -> "
            << "App→Cluster: [";

  for (int app = 0; app < NUM_OF_APPLICATIONS; ++app) {
    std::cout << appls_to_cluster[app] << ", ";
  }

  std::cout << "], Freq List: ["
            << freq_index_list[0] << ", "
            << freq_index_list[1] << ", "
            << freq_index_list[2] << "]"
            << ", Speed: " << this->target_speed
            << std::endl;

  *this->csvPDQNFile << cycle << "," << this->reward << std::endl;
  this->state_vector = this->next_state_vector; // Update state vector for the next cycle
  if (this->inference_enable==false && cycle > 0 && cycle % 50 == 0) {
    agent.save_model(this->model_dir);
  }
  // =========== pdqn end ===========
  auto decide_time_ms = (this->clock_to_ns(Clock::now()) - this->clock_to_ns(time_decide_begin))*1e-6;
  std::cout << "[PDQN] Decision making time: " << decide_time_ms << " ms" << std::endl;
  
  // ===================== APPLY MAPPING via monitorWrite =====================
  auto time_actuate_begin = Clock::now();
  if (new_appl || ended_appl) {
    for (int i = 0; i < appl_monitor->nAttached; i++) {
      pid_t pid = appl_monitor->appls[i].pid;
      // std::cout << "[PDQN] Applying mapping for PID "
      //         << pid << ", Name: " << appl_monitor->appls[i].name << std::endl;
      int cluster = appls_to_cluster[
          appls_index_map.at(appl_monitor->appls[i].name)
      ];
      appl_monitor->appls[i].mapping = cluster;

      data_t temp_data = monitorRead(appl_monitor->appls[i].segmentId);
      // std::cout << "[PDQN] Read shared memory for PID "
      //         << pid << std::endl;
      setUseGPU(&temp_data, cluster == MAP_GPU_ID);
      monitorWrite(appl_monitor->appls[i].segmentId, &temp_data);
      // std::cout << "[PDQN] Updated shared memory for PID "
      //         << pid << " to mapping "
      //         << (cluster == MAP_LITTLE_ID ? "LITTLE" :
      //             cluster == MAP_BIG_ID    ? "BIG"    : "GPU")
      //         << std::endl;

      if (cluster == MAP_LITTLE_ID) {
        UpdateCpuSet(appl_monitor, pid, {0, 3, 4, 5});
      } else if (cluster == MAP_BIG_ID) {
        UpdateCpuSet(appl_monitor, pid, {1, 2});
      }
      // std::cout << "[PDQN] Updated CPU affinity for PID "
      //         << pid << std::endl;

      appl_info[pid].mapping = cluster;

      // std::cout << "[PDQN] Applied mapping to PID "
      //         << pid << " ("
      //         << appl_monitor->appls[i].name << " → "
      //         << (cluster == MAP_LITTLE_ID ? "LITTLE" :
      //             cluster == MAP_BIG_ID    ? "BIG"    : "GPU")
      //         << ")"
      //         << std::endl;
    }
  }

  auto apply_time_ms = (this->clock_to_ns(Clock::now()) - this->clock_to_ns(time_actuate_begin))*1e-6;
  std::cout << "[PDQN] Apply decision time: " << apply_time_ms << " ms" << std::endl;

  this->setFreqLittle(cpu_fre_list[freq_index_list[0]]);
  this->setFreqBig(cpu_fre_list[freq_index_list[1]]);
  this->setFreqGpu(gpu_fre_list[freq_index_list[2]]);
  this->setSpeed(this->target_speed);

  auto cycle_time_ms = (this->clock_to_ns(Clock::now()) - this->clock_to_ns(curr_time))*1e-6;
  std::cout << "[PDQN] Cycle time: " << cycle_time_ms << " ms" << std::endl;

  this->exec_time = (this->clock_to_ns(Clock::now()) - this->clock_to_ns(curr_time))*1e-9;
}