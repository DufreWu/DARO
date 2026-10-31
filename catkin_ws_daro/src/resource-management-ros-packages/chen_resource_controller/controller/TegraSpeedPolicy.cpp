#include "TegraSpeedPolicy.h"
#include <iostream>
#include <chrono>
#include <set>
#include "CGroupUtils.h"

#include <ros/ros.h>
#include <std_msgs/String.h>


#define BLOCKWIFI

typedef std::chrono::high_resolution_clock Clock;

TegraSpeedPolicy::TegraSpeedPolicy() : nh(){
  this->appl_monitor = monitorInit();
  this->sensors = new GetNode_Tegra();
  this->utilization = new Utilization(TEGRA_NUM_OF_CPU_CORES);
  this->sensors->setAutoCPUFreqScaling(false);
  this->battery = new Battery2RCECM(5.0, 1.0, 25.0); // 5Ah, 100% SOC, 25C
  this->speed = 2.5;

  //block wifi
#ifdef BLOCKWIFI
  // system("sudo rfkill block wifi");
  // system("sudo rfkill block bluetooth");
#endif
  
  this->csvSysFilename = ""; 
  this->csvSysFile = NULL;
  this->csvApplFilename = ""; 
  this->csvApplFile = NULL;
}

TegraSpeedPolicy::TegraSpeedPolicy(std::string log_dir, int freq_b, int freq_l, int freq_g, float speed) : TegraSpeedPolicy() {
  this->log_dir = log_dir;
  this->csvSysFilename = this->log_dir + "/sys.csv";
  this->csvSysFile = new std::ofstream(this->csvSysFilename.c_str());
  if (!this->csvSysFile->is_open()) {
    std::cout << "Cannot open System CSV file at " << this->csvSysFilename.c_str() << std::endl;
    exit(0);
  }
  *this->csvSysFile << "TIME,UTIME,ITERATION,COMPW,FREQ_B,FREQ_L,FREQ_G,";
  for(int i=0; i<TEGRA_NUM_OF_CPU_CORES; i++)
    *this->csvSysFile << "UTIL" << i << ",";
  *this->csvSysFile << "GPU_UTIL" << ",";
  *this->csvSysFile << "VDDINW,MUXW,GPUW,SOCW,CPUW,DDRW,WIFIW,SPEED,MECHW,EXECT,EPD,DIST,SOC,SOH" << std::endl;

  this->csvApplFilename = this->log_dir + "/" + "appls.csv"; 
  this->csvApplFile = new std::ofstream(this->csvApplFilename.c_str());
  if (! this->csvApplFile->is_open()) {
    std::cout << "Cannot open Applications CSV file" << std::endl;
    exit(0);
  }
  *this->csvApplFile << "TIME,UTIME,ITERATION,PID1,REF1,THR1,MAP1,PID2,REF2,THR2,MAP2,PID3,REF3,THR3,MAP3,PID4,REF4,THR4,MAP4,PID5,REF5,THR5,MAP5" << std::endl;

  this->speedPublisher = this->nh.advertise<std_msgs::String>("speed", 1000);

  this->setFreqBig(freq_b);
  this->setFreqLittle(freq_l);
  this->setFreqGpu(freq_g);
  this->setSpeed(speed);
}

TegraSpeedPolicy::~TegraSpeedPolicy(){  
  killAttachedApplications(this->appl_monitor);
  monitorDestroy(this->appl_monitor);
  this->sensors->setAutoCPUFreqScaling(true);

  delete this->sensors;
  delete this->utilization;
  if(this->csvSysFile){
    this->csvSysFile->close();
    delete this->csvSysFile;
  }
  if(this->csvApplFile){
    this->csvApplFile->close();
    delete this->csvApplFile;
  }
}

void TegraSpeedPolicy::run(int cycle){
  auto curr_time = Clock::now();
  auto curr_time_str = this->clock_to_s_str(curr_time);
  auto curr_time_sec = this->clock_to_s(curr_time);

  ROS_ERROR("Cycle %d", cycle);
  if(ros::ok()) {
    std_msgs::String msg;
    msg.data = "5";
    ROS_INFO("Publishing speed %s", msg.data);
    this->speedPublisher.publish(msg);
  } else {
    ROS_ERROR("ROS not ok!");
  }

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
  
//  std::cout << "Power consumption: " << sensors->getBigW() << " (big) " << sensors->getLittleW() << " (LITTLE)" << std::endl;        
  
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

  if(this->csvSysFile){
    float currpower = this->sensors->getBoardW();    

    *this->csvSysFile << curr_time_str << "," << curr_time_sec << "," << cycle << "," << currpower << "," 
    << curr_freqs[MAP_BIG_ID] << "," << curr_freqs[MAP_LITTLE_ID] << "," << curr_freqs[MAP_GPU_ID] << "," ;
    for(int i=0; i<TEGRA_NUM_OF_CPU_CORES; i++)
      *this->csvSysFile << u_cpu[i] << ",";
    *this->csvSysFile << u_gpu << ",";
    *this->csvSysFile << this->sensors->getVddInW() << "," << this->sensors->getMuxW() << "," << this->sensors->getGpuW() << "," << this->sensors->getSocW() << "," 
                      << this->sensors->getCpuW() << "," << this->sensors->getDdrW() << "," << this->sensors->getWifiW() << "," 
                      << this->speed << "," << curr_mech_power << "," << this->exec_time;

    *this->csvSysFile << std::endl;
  }
  
  //run resource management policy
  std::set<pid_t> alreadyAnalyzedPid;
  int target_freqs[2];
  target_freqs[MAP_LITTLE_ID] = TEGRA_LITTLE_MIN_FREQ; 
  target_freqs[MAP_BIG_ID] = TEGRA_BIG_MIN_FREQ; 
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

  //analyze running applications to update the local data structure and start analyzing the current situation
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

    //identify target frequency for the cluster the application is mapped on based on the current throughput, quota and frequency
    int my_target_freq = curr_freqs[appl_info[curr_pid].mapping];
    if (appl_info[curr_pid].unit_th_processing_time > 0) {
      if(appl_info[curr_pid].thr == 0){
        // just wait for the application to get started
      } else {
          float desired_time_low = (1/appl_info[curr_pid].thr_ref) * 0.6;
          float desired_time_high = (1/appl_info[curr_pid].thr_ref) * 0.8;
          float desired_time_mid = (desired_time_low + desired_time_high) / 2;
          float curr_processing_time = appl_info[curr_pid].unit_th_processing_time;
          freq_pid_params.err = (curr_processing_time - desired_time_mid);
          freq_pid_params.integ_err += freq_pid_params.err;
          freq_pid_params.deriv_err = freq_pid_params.err - freq_pid_params.prev_err;
          freq_pid_params.prev_err = freq_pid_params.err;
          float error = freq_pid_params.err * freq_pid_params.kp + freq_pid_params.integ_err * freq_pid_params.ki + freq_pid_params.deriv_err * freq_pid_params.kd;

          float factor = 1;
          if(curr_processing_time < desired_time_low || curr_processing_time > desired_time_high) {
             factor = 1 + ( error / desired_time_mid);
          }
          my_target_freq = curr_freqs[appl_info[curr_pid].mapping] * factor;
          
          std::cout << "Hiiii, I'm here to change the freq by factor of " << factor << " because processing time is only " 
            << appl_info[curr_pid].unit_th_processing_time << " but it could be " << (1/appl_info[curr_pid].thr) 
            << ", so let's change the freq from " << curr_freqs[appl_info[curr_pid].mapping] << " to " << my_target_freq <<  "\n";
      }
    }

    if(appl_info[curr_pid].mapping == MAP_BIG_ID) {
      if(my_target_freq < TEGRA_BIG_MIN_FREQ)
        my_target_freq = TEGRA_BIG_MIN_FREQ;
      if((my_target_freq-TEGRA_BIG_MIN_FREQ)%TEGRA_CPU_FREQ_STEP != 0) //round up frequency to next step
        my_target_freq = ((my_target_freq-TEGRA_BIG_MIN_FREQ) / TEGRA_CPU_FREQ_STEP) * TEGRA_CPU_FREQ_STEP + TEGRA_CPU_FREQ_STEP + TEGRA_BIG_MIN_FREQ;
    } else if(appl_info[curr_pid].mapping == MAP_LITTLE_ID) {
      if(my_target_freq < TEGRA_LITTLE_MIN_FREQ)
        my_target_freq = TEGRA_LITTLE_MIN_FREQ;
      if((my_target_freq-TEGRA_LITTLE_MIN_FREQ)%TEGRA_CPU_FREQ_STEP != 0) //round up frequency to next step
        my_target_freq = ((my_target_freq-TEGRA_LITTLE_MIN_FREQ) / TEGRA_CPU_FREQ_STEP) * TEGRA_CPU_FREQ_STEP + TEGRA_CPU_FREQ_STEP + TEGRA_LITTLE_MIN_FREQ;
    } else if(appl_info[curr_pid].mapping == MAP_GPU_ID) {
      if(my_target_freq < TEGRA_GPU_MIN_FREQ)
        my_target_freq = TEGRA_GPU_MIN_FREQ;
      if((my_target_freq-TEGRA_GPU_MIN_FREQ)%TEGRA_GPU_FREQ_STEP != 0) //round up frequency to next step
        my_target_freq = ((my_target_freq-TEGRA_GPU_MIN_FREQ) / TEGRA_GPU_FREQ_STEP) * TEGRA_GPU_FREQ_STEP + TEGRA_GPU_FREQ_STEP + TEGRA_GPU_MIN_FREQ;
    } else { 
      std::cout << "GPU mapping not implemented yet" << std::endl;
      exit(0); 
    }    
    //update cluster frequency if the one required by the current application is higher
    if(my_target_freq > target_freqs[appl_info[curr_pid].mapping]){
      target_freqs[appl_info[curr_pid].mapping] = my_target_freq;
    }   

    std::cout << "appl1: " << appl_info[curr_pid].thr_ref << " " << appl_info[curr_pid].thr << " " << curr_freqs[appl_info[curr_pid].mapping] << " " << my_target_freq << std::endl;
    std::cout << "appl2: " << 1.0/appl_info[curr_pid].thr_ref << " " << 1.0/appl_info[curr_pid].thr << " " << temp_data.usleepTime << std::endl;
  } 
  
  //In a second csv file we save the data about the applications. they have to be postprocessed to be attached to the other system file
  if(this->csvApplFile){
    *this->csvApplFile << curr_time_str << "," << curr_time_sec << "," << cycle;
    for(std::map<pid_t, TegraSpeedPolicy::appl_state>::iterator mapIt = appl_info.begin(); mapIt != appl_info.end(); mapIt++) {
      *this->csvApplFile  << "," << mapIt->first << "," << mapIt->second.thr_ref << "," << mapIt->second.thr << "," << mapIt->second.mapping << "," << mapIt->second.unit_th_processing_time;  
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
    
  if(!new_appl) { //DO NOTE: if an application entered the system do not apply DVFS since we need 1 cycle to profile
    if(target_freqs[MAP_LITTLE_ID] > TEGRA_LITTLE_MAX_FREQ)
      target_freqs[MAP_LITTLE_ID] = TEGRA_LITTLE_MAX_FREQ;
    else if(target_freqs[MAP_LITTLE_ID] < TEGRA_LITTLE_MIN_FREQ)
      target_freqs[MAP_LITTLE_ID] = TEGRA_LITTLE_MIN_FREQ;

    if(target_freqs[MAP_BIG_ID] > TEGRA_BIG_MAX_FREQ)
      target_freqs[MAP_BIG_ID] = TEGRA_BIG_MAX_FREQ;  
    else if(target_freqs[MAP_BIG_ID] < TEGRA_BIG_MIN_FREQ)
      target_freqs[MAP_BIG_ID] = TEGRA_BIG_MIN_FREQ;

    if(target_freqs[MAP_LITTLE_ID] != curr_freqs[MAP_LITTLE_ID]){
      std::cout << "---> change freq on LITTLE: " << target_freqs[MAP_LITTLE_ID] << std::endl;
      sensors->setCPUFreq(TEGRA_LITTLE_CORE_ID, target_freqs[MAP_LITTLE_ID]);
      //sensors->setCPUFreq(TEGRA_LITTLE_CORE_ID, TEGRA_LITTLE_MIN_FREQ);
    }
    if(target_freqs[MAP_BIG_ID] != curr_freqs[MAP_BIG_ID]){
      std::cout << "---> change freq on big: " << target_freqs[MAP_BIG_ID] << std::endl;
      sensors->setCPUFreq(TEGRA_BIG_CORE_ID, target_freqs[MAP_BIG_ID]);
    }
  }
  this->exec_time = (this->clock_to_ns(Clock::now()) - this->clock_to_ns(curr_time))*1e-9;
  // if((cycle+1) % 60 == 0) {
  //   this->setSpeed(this->speed + 0.25);
  // }

}

float TegraSpeedPolicy::getCurrentMechW() {
  return this->getMechW(this->speed);
}


float TegraSpeedPolicy::getMechW(float sp) {
  // float E_per_meter = 0.00701982 * sp + 0.41035056*sp*sp;
  // float mechW = E_per_meter * sp;
  // float mechW = 0.95 *sp*sp + -0.75 * sp + 3.82;
  // return mechW * 1e3;
  return this->getMechW(sp, 0);
}

float TegraSpeedPolicy::getMechW(float v, float acc) {
  // float p = 3.8908665732713166*acc*v + 1.2995375283981614*v + 0.23225537743664665*v*v*v + 4.6499999999999995;
  // float p = 0.8764185547037674*acc*v + 0.2916812491360945*v + 0.05283711225146047*v*v*v + 1.1624999999999999;
  // float p = 2.3081405089670817*acc*v + 1.495789636454757*v + 0.12776126223382892*v*v*v + 1.332;
  float p = 1.14005*v + 0.108930*v*v*v + 1.1625;
  return p * 1e3;
}

void TegraSpeedPolicy::setSpeed(float sp){
  this->speed = sp;

  this->nh.setParam("/chen/speed", this->speed);

  std_msgs::String msg;
  msg.data = std::to_string(sp);
  this->speedPublisher.publish(msg);
  
  std::string speed_file_path = "/home/chen2/catkin_ws_chen/src/resource-management-ros-packages/chen_resource_controller_speed.txt"; 
  std::ofstream* speed_file = new std::ofstream(speed_file_path.c_str());
  if (! speed_file->is_open()) {
    std::cout << "Cannot open speed file at " << speed_file_path.c_str() << std::endl;
    exit(0);
  }
  *speed_file << sp << "\n";
  speed_file->close();

  std::cout << "Setting speed: " << sp << std::endl;
}

float TegraSpeedPolicy::calDistance(float sp, float dt) {
  this->distance += sp * dt;
  return this->distance;
}

void TegraSpeedPolicy::setFreqBig(int freq) {
  this->nh.setParam("/chen/freq_big", freq);
  if(freq > TEGRA_BIG_MAX_FREQ)
    freq = TEGRA_BIG_MAX_FREQ;  
  else if(freq < TEGRA_BIG_MIN_FREQ)
    freq = TEGRA_BIG_MIN_FREQ;
  if((freq-TEGRA_BIG_MIN_FREQ)%TEGRA_CPU_FREQ_STEP != 0) //round up frequency to next step
    freq = ((freq-TEGRA_BIG_MIN_FREQ) / TEGRA_CPU_FREQ_STEP) * TEGRA_CPU_FREQ_STEP + TEGRA_CPU_FREQ_STEP + TEGRA_BIG_MIN_FREQ;
  std::cout << "Setting freq big: " << freq << "\n";
  sensors->setCPUFreq(TEGRA_BIG_CORE_ID, freq);
}

void TegraSpeedPolicy::setFreqLittle(int freq) {
  this->nh.setParam("/chen/freq_little", freq);
  if(freq > TEGRA_LITTLE_MAX_FREQ)
    freq = TEGRA_LITTLE_MAX_FREQ;
  else if(freq < TEGRA_LITTLE_MIN_FREQ)
    freq = TEGRA_LITTLE_MIN_FREQ;
  if((freq-TEGRA_LITTLE_MIN_FREQ)%TEGRA_CPU_FREQ_STEP != 0) //round up frequency to next step
    freq = ((freq-TEGRA_LITTLE_MIN_FREQ) / TEGRA_CPU_FREQ_STEP) * TEGRA_CPU_FREQ_STEP + TEGRA_CPU_FREQ_STEP + TEGRA_LITTLE_MIN_FREQ;
  std::cout << "Setting freq little: " << freq << "\n";
  sensors->setCPUFreq(TEGRA_LITTLE_CORE_ID, freq);
}

// 216750000 318750000 420750000 522750000 624750000 726750000 854250000 930750000 1032750000 1122000000 1236750000 1300500000
void TegraSpeedPolicy::setFreqGpu(int freq) {
  this->nh.setParam("/chen/freq_gpu", freq);
  if(freq > TEGRA_GPU_MAX_FREQ)
    freq = TEGRA_GPU_MAX_FREQ;
  else if(freq < TEGRA_GPU_MIN_FREQ)
    freq = TEGRA_GPU_MIN_FREQ;

  std::vector<int> gpu_freq_list = sensors->getAvailableGPUFreq();
  if (find(gpu_freq_list.begin(), gpu_freq_list.end(), freq) == gpu_freq_list.end()) {
    // Find the closest neighbor if the frequency is not in the list
    auto it = std::lower_bound(gpu_freq_list.begin(), gpu_freq_list.end(), freq);
    if (it == gpu_freq_list.end() || (it != gpu_freq_list.begin() && abs(freq - *(it - 1)) < abs(freq - *it))) {
        --it; // Choose the previous neighbor if it's closer
    }

    freq = *it; // Set the frequency to the closest neighbor
    std::cout << "Setting freq gpu to closest neighbor: " << freq << "\n";
  }
  // if((freq-TEGRA_GPU_MIN_FREQ)%TEGRA_GPU_FREQ_STEP != 0) //round up frequency to next step
  //   freq = ((freq-TEGRA_GPU_MIN_FREQ) / TEGRA_GPU_FREQ_STEP) * TEGRA_GPU_FREQ_STEP + TEGRA_GPU_FREQ_STEP + TEGRA_GPU_MIN_FREQ;
  std::cout << "Setting freq gpu: " << freq << "\n";
  sensors->setGPUFreq(freq);
}

float TegraSpeedPolicy::getMechEnergyPerMeter(float mech_power, float speed) {
  return mech_power / speed;
}

float TegraSpeedPolicy::getCompEnergyPerMeter(float comp_power, float speed) {
  return comp_power / speed;
}

float TegraSpeedPolicy::getEnergyPerMeter(float comp_power, float mech_power, float speed) {
  return this->getMechEnergyPerMeter(mech_power, speed) + this->getCompEnergyPerMeter(comp_power, speed);
}

std::chrono::seconds::rep TegraSpeedPolicy::clock_to_s(std::chrono::time_point<std::chrono::system_clock> tp) {
    return std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch()).count();
}

std::chrono::nanoseconds::rep TegraSpeedPolicy::clock_to_ns(std::chrono::time_point<std::chrono::system_clock> tp) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
}

std::string TegraSpeedPolicy::clock_to_s_str(std::chrono::time_point<std::chrono::system_clock> tp) {
    std::time_t curr_time_c = std::chrono::system_clock::to_time_t(tp);
    std::string curr_time_s = std::string(std::ctime(&curr_time_c));
    curr_time_s = curr_time_s.substr(0, curr_time_s.size()-1);
    return curr_time_s;
}