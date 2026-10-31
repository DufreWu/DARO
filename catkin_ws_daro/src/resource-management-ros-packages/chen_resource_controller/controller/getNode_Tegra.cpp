#include "getNode_Tegra.h"
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstdio> // sprintf
#include <cstdlib> // atoi
#include <fcntl.h>
#include <cstring>  // strncmp
#include <string>
#include <iostream> // cerr
#include <fstream>  // cerr

GetNode_Tegra::GetNode_Tegra() {
  //paths to access CPU frequency/status infos
  for (int i = 0; i < TEGRA_NUM_OF_CPU_CORES; i++) {
    std::string temp;
    temp = CPU_PATH + std::to_string(i) + CPU_CURR_STATUS;
    cpu_online_list[i] = temp;
    temp = CPU_PATH + std::to_string(i) + CPU_CURR_FREQ;
    cpu_freq_list[i] = temp;
    temp = CPU_PATH + std::to_string(i) + CPU_CURR_SCAL_FREQ;
    cpu_scal_freq_list[i] = temp;
    temp = CPU_PATH + std::to_string(i) + CPU_AVAILABLE_FREQS;
    cpu_avail_freq_list[i] = temp;
    temp = CPU_PATH + std::to_string(i) + CPU_FREQ_SCALING_GOV;
    cpu_freq_gov_list[i] = temp;
    temp = CPU_PATH + std::to_string(i) + CPU_SET_FREQ_MAX;
    cpu_set_freq_max_list[i] = temp;
    temp = CPU_PATH + std::to_string(i) + CPU_SET_FREQ_MIN;
    cpu_set_freq_min_list[i] = temp;
  }
  
  //read sensors for the first time  
  this->update_read();

  //initialize attributes for power consumption
  this->gpuWavg[0] = this->gpuWs = this->getGpuW();
  this->socWavg[0] = this->socWs = this->getSocW();
  this->wifiWavg[0] = this->wifiWs = this->getWifiW();
  this->vddInWavg[0] = this->vddInWs = this->getVddInW();
  this->cpuWavg[0] = this->cpuWs = this->getCpuW();
  this->ddrWavg[0] = this->ddrWs = this->getDdrW();
  this->muxWavg[0] = this->muxWs = this->getMuxW();
  this->vdd5v0IoWavg[0] = this->vdd5v0IoWs = this->getVdd5v0IoW();
  this->vdd3V3SysWavg[0] = this->vdd3V3SysWs = this->getVdd3V3SysW();
  this->vdd3V3IoSlpWavg[0] = this->vdd3V3IoSlpWs = this->getVdd3V3IoSlpW();
  this->vdd1V8IoWavg[0] = this->vdd1V8IoWs = this->getVdd1V8IoW();
  this->vdd3V3SysM2Wavg[0] = this->vdd3V3SysM2Ws = this->getVdd3V3SysM2W();
  
  for(int i = 1; i < POW_WINDOW_SIZE; i++) {
    this->gpuWavg[i] = 0;
    this->socWavg[i] = 0;
    this->wifiWavg[i] = 0;
    this->vddInWavg[i] = 0;
    this->cpuWavg[i] = 0;
    this->ddrWavg[i] = 0;
    this->muxWavg[i] = 0;
    this->vdd5v0IoWavg[i] = 0;
    this->vdd3V3SysWavg[i] = 0;
    this->vdd3V3IoSlpWavg[i] = 0;
    this->vdd1V8IoWavg[i] = 0;
    this->vdd3V3SysM2Wavg[i] = 0;

  }
  this->slidingIndex = 1;
}

GetNode_Tegra::~GetNode_Tegra() {
}

bool GetNode_Tegra::isAutoCPUFreqScaling(){
  FILE* fp;
  char value;
  fp = fopen(AUTO_CPU_FREQ_SCALING, "r");
  if(fp) {
    fscanf(fp, "%c", &value);
    fclose(fp);
  } else
    value = 'N'; //error opening the file
  return (value=='Y'?true:false);
}

bool GetNode_Tegra::isCPUActive(int cpuNum) {
  if(cpuNum==0) //the first core is always on
    return true;
  else if(cpuNum<0 || cpuNum > TEGRA_NUM_OF_CPU_CORES)
    return false; //do note. it does not exist

  FILE* fp;
  int value;
  fp = fopen(cpu_online_list[cpuNum].c_str(), "r");
  if(fp) {
    fscanf(fp, "%d", &value);
    fclose(fp);
  } else
    value = -1; //error opening the file
  return (value==1)?true:false;
}

int GetNode_Tegra::getCPUCurFreq(int cpuNum) { //DO NOTE: values are returned in KHz
  if(cpuNum<0 || cpuNum > TEGRA_NUM_OF_CPU_CORES)
    return -1; //do note. it does not exist

  FILE* fp;
  int value;
  fp = fopen(cpu_freq_list[cpuNum].c_str(), "r");
  if(fp) {
    fscanf(fp, "%d", &value);
    fclose(fp);
  } else
    value = -1; //error opening the file
  return value; //DO NOTE: values are returned in KHz
}

int GetNode_Tegra::getCPUCurScalFreq(int cpuNum) { //DO NOTE: values are returned in KHz
  if(cpuNum<0 || cpuNum > TEGRA_NUM_OF_CPU_CORES)
    return -1; //do note. it does not exist

  FILE* fp;
  int value;
  fp = fopen(cpu_scal_freq_list[cpuNum].c_str(), "r");
  if(fp) {
    fscanf(fp, "%d", &value);
    fclose(fp);
  } else
    value = -1; //error opening the file
  return value; //DO NOTE: values are returned in KHz
}


int GetNode_Tegra::getGPUCurFreq() { //DO NOTE: values are returned in KHz
  FILE* fp;
  int value;
  fp = fopen(GPU_CURR_FREQ, "r");
  if(fp) {
    fscanf(fp, "%d", &value);
    value /= 1000; //DO NOTE: values are returned in KHz
    fclose(fp);
  } else
    value = -1; //error opening the file
  return value;
}

int GetNode_Tegra::getGPUCurUtilization() { 
  FILE* fp;
  int value;
  fp = fopen(GPU_UTILIZATION, "r");
  if(fp) {
    fscanf(fp, "%d", &value);
    fclose(fp);
  } else
    value = 0; //error opening the file
  return value;
}

int GetNode_Tegra::getMemCurFreq() { //DO NOTE: values are returned in KHz
  FILE* fp;
  int value;
  fp = fopen(MEM_CURR_FREQ, "r");
  if(fp) {
    fscanf(fp, "%d", &value);
    value /= 1000; //DO NOTE: values are returned in KHz
    fclose(fp);
  } else
    value = -1; //error opening the file
  return value;
}

std::vector<int> GetNode_Tegra::getAvailableCPUFreq(int cpuNum) {
  std::vector<int> freqs;
  if(cpuNum>=0 && cpuNum < TEGRA_NUM_OF_CPU_CORES){
    FILE* fp;
    int value;
    fp = fopen(cpu_avail_freq_list[cpuNum].c_str(), "r");
    if(fp) {
      fscanf(fp, "%d", &value);
      while(!feof(fp)) {
        freqs.push_back(value);       
        fscanf(fp, "%d", &value);
      }
      fclose(fp);
    }
  }
  return freqs; //DO NOTE: values are returned in KHz  
}

std::vector<int> GetNode_Tegra::getAvailableGPUFreq() {
  std::vector<int> freqs;
  FILE* fp;
  int value;
  fp = fopen(GPU_AVAILABLE_FREQS, "r");
  if(fp) {
    fscanf(fp, "%d", &value);
    while(!feof(fp)) {
      freqs.push_back(value/1000);    
      fscanf(fp, "%d", &value);
    }
    fclose(fp);
  }
  return freqs; //DO NOTE: values are returned in KHz  
}

std::vector<int> GetNode_Tegra::getAvailableMemFreq() {
  std::vector<int> freqs;
  //TODO 
  std::cout << "NOT IMPLEMENTED!" << std::endl;
  exit(0);
  return freqs; //DO NOTE: values are returned in KHz  
}

void GetNode_Tegra::setAutoCPUFreqScaling(bool status) {
  //it is necessary to modify three files to disable/enable automatic dvfs governor.

  //we need to call the nvpmodel to reset the overall configuration of the various governors (e.g. max/min freqs for governor, etc etc.)
  system(NVPMODEL_M0);

  FILE* fp;
  fp = fopen(AUTO_CPU_FREQ_SCALING, "w");
  if(fp) {
    fprintf(fp, "%c", (status==true)?'Y':'N');
    fclose(fp);
  }
  fp = fopen(AUTO_B_CLUSTER_FREQ_SCALING, "w");
  if(fp) {
    fprintf(fp, "%d", (status==true)?1:0);
    fclose(fp);
  }
  fp = fopen(AUTO_M_CLUSTER_FREQ_SCALING, "w");
  if(fp) {
    fprintf(fp, "%d", (status==true)?1:0);
    fclose(fp);
  }
  //then change the governor
  fp = fopen(cpu_freq_gov_list[0].c_str(), "w");
  if(fp) {
    fprintf(fp, "%s", (status==true)?CPU_SCHEDUTIL_GOVERNOR:CPU_PERFORMANCE_GOVERNOR);
    fclose(fp);
  }
  fp = fopen(cpu_freq_gov_list[1].c_str(), "w");
  if(fp) {
    fprintf(fp, "%s", (status==true)?CPU_SCHEDUTIL_GOVERNOR:CPU_PERFORMANCE_GOVERNOR);
    fclose(fp);
  }
  if(!status)
    for (int i = 0; i < TEGRA_NUM_OF_CPU_CORES; i++)
      this->setCPUActive(i, true);
}

void GetNode_Tegra::setCPUActive(int cpuNum, bool status) {
  //the first core is always on. IDs not in [1;TEGRA_NUM_OF_CPU_CORES] are not valid
  if(cpuNum==0 || cpuNum<0 || cpuNum > TEGRA_NUM_OF_CPU_CORES)
    return;

  FILE* fp;
  fp = fopen(cpu_online_list[cpuNum].c_str(), "w");
  if(fp) {
    fprintf(fp, "%d", (status==true)?1:0);
    fclose(fp);
  }
}

void GetNode_Tegra::setCPUFreq(int cpuNum, int freq) {
  //the first core is always on. IDs not in [1;TEGRA_NUM_OF_CPU_CORES] are not valid
  if(cpuNum==0 || cpuNum<0 || cpuNum > TEGRA_NUM_OF_CPU_CORES)
    return;

  FILE* fp;

  //actually on the ARM cluster the userspace governor does not work. so the trick is to 
  //use the performance governor and set the same frequency for the max and min bound.
  //the bothering part is that it is relevant the order of writing of the min and max 
  //value since min cannot be > of max
  //TODO REFACTOR THE CODE

  //read min scaling freq
  int minvalue;
  fp = fopen(cpu_set_freq_min_list[cpuNum].c_str(), "r");
  if(fp) {
    fscanf(fp, "%d", &minvalue);
    fclose(fp);
  } else
    minvalue = -1; //error opening the file

  //read max scaling freq
  int maxvalue;
  fp = fopen(cpu_set_freq_max_list[cpuNum].c_str(), "r");
  if(fp) {
    fscanf(fp, "%d", &maxvalue);
    fclose(fp);
  } else
    maxvalue = -1; //error opening the file

  if(freq < minvalue) {
    fp = fopen(cpu_set_freq_min_list[cpuNum].c_str(), "w");
    if(fp) {
      fprintf(fp, "%d", freq);
      fclose(fp);
    }
    fp = fopen(cpu_set_freq_max_list[cpuNum].c_str(), "w");
    if(fp) {
      fprintf(fp, "%d", freq);
      fclose(fp);
    }
  } else {
    fp = fopen(cpu_set_freq_max_list[cpuNum].c_str(), "w");
    if(fp) {
      fprintf(fp, "%d", freq);
      fclose(fp);
    }
    fp = fopen(cpu_set_freq_min_list[cpuNum].c_str(), "w");
    if(fp) {
      fprintf(fp, "%d", freq);
      fclose(fp);
    }    
  }

}

void GetNode_Tegra::setGPUFreq(int freq) {
  FILE* fp;
  //read min gpu freq
  int minvalue;
  fp = fopen(GPU_SET_FREQ_MIN, "r");
  if(fp) {
    fscanf(fp, "%d", &minvalue);
    fclose(fp);
  } else
    minvalue = -1; //error opening the file

  //read max scaling freq
  int maxvalue;
  fp = fopen(GPU_SET_FREQ_MAX, "r");
  if(fp) {
    fscanf(fp, "%d", &maxvalue);
    fclose(fp);
  } else
    maxvalue = -1; //error opening the file

//  if (freq < minvalue)
//    freq = minvalue;
//  if (freq > maxvalue)
//    freq = maxvalue;

  freq *= 1000; //DO NOTE: values are returned in KHz
  if(freq < minvalue) {
    fp = fopen(GPU_SET_FREQ_MIN, "w");
    if(fp) {
      fprintf(fp, "%d", freq);
      fclose(fp);
    }
    fp = fopen(GPU_SET_FREQ_MAX, "w");
    if(fp) {
      fprintf(fp, "%d", freq);
      fclose(fp);
    }
  } else {
    fp = fopen(GPU_SET_FREQ_MAX, "w");
    if(fp) {
      fprintf(fp, "%d", freq);
      fclose(fp);
    }
    fp = fopen(GPU_SET_FREQ_MIN, "w");
    if(fp) {
      fprintf(fp, "%d", freq);
      fclose(fp);
    }    
  }
}

void GetNode_Tegra::setMemFreq(int freq) {
  //TODO
  std::cout << "NOT IMPLEMENTED!" << std::endl;
  exit(0);
}

void GetNode_Tegra::setCPUGovernor(int cpuNum, std::string gov) {
  FILE* fp;
  fp = fopen(cpu_freq_gov_list[cpuNum].c_str(), "w");
  if (fp) {
    fprintf(fp, "%s", gov.c_str());
    fclose(fp);
  }
}
void GetNode_Tegra::setGPUGovernor(std::string gov) {
  FILE* fp;
  fp = fopen(GPU_FREQ_SCALING_GOV, "w");
  if (fp) {
    fprintf(fp, "%s", gov.c_str());
    fclose(fp);
  }
}

int GetNode_Tegra::getFanSpeed() {
  FILE* fp;
  int value;
  fp = fopen(FAN_CURR_SPEED, "r");
  if(fp) {
    fscanf(fp, "%d", &value);
    fclose(fp);
  } else
    value = -1; //error opening the file
  return value;
}

void GetNode_Tegra::setFanSpeed(int pwm) {
  FILE* fp;
  fp = fopen(FAN_SET_SPEED, "w");
  if(fp) {
    fprintf(fp, "%d", pwm);
    fclose(fp);
  }
}

//DO NOTE: I need such functions because the default step time is quite long 
//and not compatible with the power monitor
int GetNode_Tegra::getFanStepTime() {
  FILE* fp;
  int value;
  fp = fopen(FAN_STEP_TIME, "r");
  if(fp) {
    fscanf(fp, "%d", &value);
    fclose(fp);
  } else
    value = -1; //error opening the file
  return value;
}

void GetNode_Tegra::resetFanStepTime() {
  this->setFanStepTime(DEFAULT_FAN_STEP_TIME);
}

void GetNode_Tegra::setFanStepTime(int time) {
  FILE* fp;
  fp = fopen(FAN_STEP_TIME, "w");
  if(fp) {
    fprintf(fp, "%d", time);
    fclose(fp);
  }
}

float GetNode_Tegra::read_sensor(const char *node) {
  float value;
  FILE* fp;
  fp = fopen(node, "r");
  if(fp) {
    fscanf(fp, "%f", &value);
    fclose(fp);
  } else
    value = -1;
  return value;
}

float GetNode_Tegra::getGpuV() {
  return this->gpuV;
}

float GetNode_Tegra::getGpuA() {
  return this->gpuA;
}

float GetNode_Tegra::getGpuW() {
  return this->gpuW;
}

float GetNode_Tegra::getSocV() {
  return this->socV;
}

float GetNode_Tegra::getSocA() {
  return this->socA;
}

float GetNode_Tegra::getSocW() {
  return this->socW;
}

float GetNode_Tegra::getWifiV() {
  return this->wifiV;
}

float GetNode_Tegra::getWifiA() {
  return this->wifiA;
}

float GetNode_Tegra::getWifiW() {
  return this->wifiW;
}

float GetNode_Tegra::getVddInV() {
  return this->vddInV;
}

float GetNode_Tegra::getVddInA() {
  return this->vddInA;
}

float GetNode_Tegra::getVddInW() {
  return this->vddInW;
}

float GetNode_Tegra::getCpuV() {
  return this->cpuV;
}

float GetNode_Tegra::getCpuA() {
  return this->cpuA;
}

float GetNode_Tegra::getCpuW() {
  return this->cpuW;
}

float GetNode_Tegra::getDdrV() {
  return this->ddrV;
}

float GetNode_Tegra::getDdrA() {
  return this->ddrA;
}

float GetNode_Tegra::getDdrW() {
  return this->ddrW;
}

float GetNode_Tegra::getMuxV() {
  return this->muxV;
}

float GetNode_Tegra::getMuxA() {
  return this->muxA;
}

float GetNode_Tegra::getMuxW() {
  return this->muxW;
}

float GetNode_Tegra::getVdd5v0IoV() {
  return this->vdd5v0IoV;
}

float GetNode_Tegra::getVdd5v0IoA() {
  return this->vdd5v0IoA;
}

float GetNode_Tegra::getVdd5v0IoW() {
  return this->vdd5v0IoW;
}

float GetNode_Tegra::getVdd3V3SysV() {
  return this->vdd3V3SysV;
}

float GetNode_Tegra::getVdd3V3SysA() {
  return this->vdd3V3SysA;
}

float GetNode_Tegra::getVdd3V3SysW() {
  return this->vdd3V3SysW;
}

float GetNode_Tegra::getVdd3V3IoSlpV() {
  return this->vdd3V3IoSlpV;
}

float GetNode_Tegra::getVdd3V3IoSlpA() {
  return this->vdd3V3IoSlpA;
}

float GetNode_Tegra::getVdd3V3IoSlpW() {
  return this->vdd3V3IoSlpW;
}

float GetNode_Tegra::getVdd1V8IoV() {
  return this->vdd1V8IoV;
}

float GetNode_Tegra::getVdd1V8IoA() {
  return this->vdd1V8IoA;
}

float GetNode_Tegra::getVdd1V8IoW() {
  return this->vdd1V8IoW;
}

float GetNode_Tegra::getVdd3V3SysM2V() {
  return this->vdd3V3SysM2V;
}

float GetNode_Tegra::getVdd3V3SysM2A() {
  return this->vdd3V3SysM2A;
}

float GetNode_Tegra::getVdd3V3SysM2W() {
  return this->vdd3V3SysM2W;
}


void GetNode_Tegra::update_read(){
  //collect instantaneous values
  this->gpuV = this->read_sensor(SENSOR_GPU_V);
  this->gpuA = this->read_sensor(SENSOR_GPU_A);
  this->gpuW = this->read_sensor(SENSOR_GPU_W);

  this->socV = this->read_sensor(SENSOR_SOC_V);
  this->socA = this->read_sensor(SENSOR_SOC_A);
  this->socW = this->read_sensor(SENSOR_SOC_W);

  this->wifiV = this->read_sensor(SENSOR_WIFI_V);
  this->wifiA = this->read_sensor(SENSOR_WIFI_A);
  this->wifiW = this->read_sensor(SENSOR_WIFI_W);

  this->vddInV = this->read_sensor(SENSOR_VDD_IN_V);
  this->vddInA = this->read_sensor(SENSOR_VDD_IN_A);
  this->vddInW = this->read_sensor(SENSOR_VDD_IN_W);

  this->cpuV = this->read_sensor(SENSOR_CPU_V);
  this->cpuA = this->read_sensor(SENSOR_CPU_A);
  this->cpuW = this->read_sensor(SENSOR_CPU_W);

  this->ddrV = this->read_sensor(SENSOR_DDR_V);
  this->ddrA = this->read_sensor(SENSOR_DDR_A);
  this->ddrW = this->read_sensor(SENSOR_DDR_W);

  this->muxV = this->read_sensor(SENSOR_MUX_V);
  this->muxA = this->read_sensor(SENSOR_MUX_A);
  this->muxW = this->read_sensor(SENSOR_MUX_W);

  this->vdd5v0IoV = this->read_sensor(SENSOR_5V0_IO_SYS_V);
  this->vdd5v0IoA = this->read_sensor(SENSOR_5V0_IO_SYS_A);
  this->vdd5v0IoW = this->read_sensor(SENSOR_5V0_IO_SYS_W);

  this->vdd3V3SysV = this->read_sensor(SENSOR_3V3_SYS_V);
  this->vdd3V3SysA = this->read_sensor(SENSOR_3V3_SYS_A);
  this->vdd3V3SysW = this->read_sensor(SENSOR_3V3_SYS_W);

  this->vdd3V3IoSlpV = this->read_sensor(SENSOR_3V3_IO_SLP_V);
  this->vdd3V3IoSlpA = this->read_sensor(SENSOR_3V3_IO_SLP_A);
  this->vdd3V3IoSlpW = this->read_sensor(SENSOR_3V3_IO_SLP_W);

  this->vdd1V8IoV = this->read_sensor(SENSOR_3V3_IO_SLP_V);
  this->vdd1V8IoA = this->read_sensor(SENSOR_3V3_IO_SLP_A);
  this->vdd1V8IoW = this->read_sensor(SENSOR_3V3_IO_SLP_W);

  this->vdd1V8IoV = this->read_sensor(SENSOR_1V8_IO_V);
  this->vdd1V8IoA = this->read_sensor(SENSOR_1V8_IO_A);
  this->vdd1V8IoW = this->read_sensor(SENSOR_1V8_IO_W);

  this->vdd3V3SysM2V = this->read_sensor(SENSOR_3V3_SYS_M2_W);
  this->vdd3V3SysM2A = this->read_sensor(SENSOR_3V3_SYS_M2_A);
  this->vdd3V3SysM2W = this->read_sensor(SENSOR_3V3_SYS_M2_V);
}

void GetNode_Tegra::updateSensing() {
  this->update_read();

  //update average values
  this->gpuWs = this->gpuWs - this->gpuWavg[this->slidingIndex] + this->gpuW;
  this->socWs = this->socWs - this->socWavg[this->slidingIndex] + this->socW;
  this->wifiWs = this->wifiWs - this->wifiWavg[this->slidingIndex] + this->wifiW;
  this->vddInWs = this->vddInWs - this->vddInWavg[this->slidingIndex] + this->vddInW;
  this->cpuWs = this->cpuWs - this->cpuWavg[this->slidingIndex] + this->cpuW;
  this->ddrWs = this->ddrWs - this->ddrWavg[this->slidingIndex] + this->ddrW;
  this->muxWs = this->muxWs - this->muxWavg[this->slidingIndex] + this->muxW;
  this->vdd5v0IoWs = this->vdd5v0IoWs - this->vdd5v0IoWavg[this->slidingIndex] + this->vdd5v0IoW;
  this->vdd3V3SysWs = this->vdd3V3SysWs - this->vdd3V3SysWavg[this->slidingIndex] + this->vdd3V3SysW;
  this->vdd3V3IoSlpWs = this->vdd3V3IoSlpWs - this->vdd3V3IoSlpWavg[this->slidingIndex] + this->vdd3V3IoSlpW;
  this->vdd1V8IoWs = this->vdd1V8IoWs - this->vdd1V8IoWavg[this->slidingIndex] + this->vdd1V8IoW;
  this->vdd3V3SysM2Ws = this->vdd3V3SysM2Ws - this->vdd3V3SysM2Wavg[this->slidingIndex] + this->vdd3V3SysM2W;

  this->gpuWavg[this->slidingIndex] = this->gpuW;
  this->socWavg[this->slidingIndex] = this->socW;
  this->wifiWavg[this->slidingIndex] = this->wifiW;
  this->vddInWavg[this->slidingIndex] = this->vddInW;
  this->cpuWavg[this->slidingIndex] = this->cpuW;
  this->ddrWavg[this->slidingIndex] = this->ddrW;
  this->muxWavg[this->slidingIndex] = this->muxW;
  this->vdd5v0IoWavg[this->slidingIndex] = this->vdd5v0IoW;
  this->vdd3V3SysWavg[this->slidingIndex] = this->vdd3V3SysW;
  this->vdd3V3IoSlpWavg[this->slidingIndex] = this->vdd3V3IoSlpW;
  this->vdd1V8IoWavg[this->slidingIndex] = this->vdd1V8IoW;
  this->vdd3V3SysM2Wavg[this->slidingIndex] = this->vdd3V3SysM2W;

  this->slidingIndex = (this->slidingIndex+1)%POW_WINDOW_SIZE;
}

float GetNode_Tegra::getGpuWavg() {
  return this->gpuWs / POW_WINDOW_SIZE;
}

float GetNode_Tegra::getSocWavg() {
  return this->socWs / POW_WINDOW_SIZE;
}

float GetNode_Tegra::getWifiWavg() {
  return this->wifiWs / POW_WINDOW_SIZE;
}

float GetNode_Tegra::getVddInWavg() {
  return this->vddInWs / POW_WINDOW_SIZE;
}

float GetNode_Tegra::getCpuWavg() {
  return this->cpuWs / POW_WINDOW_SIZE;
}

float GetNode_Tegra::getDdrWavg() {
  return this->ddrWs / POW_WINDOW_SIZE;
}

float GetNode_Tegra::getMuxWavg() {
  return this->muxWs / POW_WINDOW_SIZE;
}

float GetNode_Tegra::getVdd5v0IoWavg() {
  return this->vdd5v0IoWs / POW_WINDOW_SIZE;
}

float GetNode_Tegra::getVdd3V3SysWavg() {
  return this->vdd3V3SysWs / POW_WINDOW_SIZE;
}

float GetNode_Tegra::getVdd3V3IoSlpWavg() {
  return this->vdd3V3IoSlpWs / POW_WINDOW_SIZE;
}

float GetNode_Tegra::getVdd1V8IoWavg() {
  return this->vdd1V8IoWs / POW_WINDOW_SIZE;
}

float GetNode_Tegra::getVdd3V3SysM2Wavg() {
  return this->vdd3V3SysM2Ws / POW_WINDOW_SIZE;
}

float GetNode_Tegra::getChipW() {
  return this->vddInW;
}

float GetNode_Tegra::getChipWavg() {
  return this->getVddInWavg();
}

float GetNode_Tegra::getBoardW() {
  return this->vddInW + this->muxW;
}

float GetNode_Tegra::getBoardWavg() {
  return this->getVddInWavg() + this->getMuxWavg();
}
