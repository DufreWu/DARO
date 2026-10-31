#ifndef GETNODETEGRA_H
#define GETNODETEGRA_H

#include <sys/ioctl.h>
#include <string>
#include <vector>

/*
 * This class represents the interface to the virtual file system to get information from the Odroid sensors (current, voltage power. temperature and utilization).
 * This code has been taken (and slightly readapted) from the Odroid EnergyMonitor
 */

#define SENSOR_GPU_W  "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input"
#define SENSOR_GPU_A  "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_current0_input"
#define SENSOR_GPU_V  "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_voltage0_input"

#define SENSOR_SOC_W  "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power1_input"
#define SENSOR_SOC_A  "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_current1_input"
#define SENSOR_SOC_V  "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_voltage1_input"

#define SENSOR_WIFI_W  "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power2_input"
#define SENSOR_WIFI_A  "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_current2_input"
#define SENSOR_WIFI_V  "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_voltage2_input"

#define SENSOR_VDD_IN_W "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power0_input"
#define SENSOR_VDD_IN_A "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_current0_input"
#define SENSOR_VDD_IN_V "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_voltage0_input"

#define SENSOR_CPU_W "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power1_input"
#define SENSOR_CPU_A "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_current1_input"
#define SENSOR_CPU_V "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_voltage1_input"

#define SENSOR_DDR_W "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power2_input"
#define SENSOR_DDR_A "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_current2_input"
#define SENSOR_DDR_V "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_voltage2_input"

#define SENSOR_MUX_W "/sys/bus/i2c/drivers/ina3221x/0-0042/iio:device2/in_power0_input"
#define SENSOR_MUX_A "/sys/bus/i2c/drivers/ina3221x/0-0042/iio:device2/in_current0_input"
#define SENSOR_MUX_V "/sys/bus/i2c/drivers/ina3221x/0-0042/iio:device2/in_voltage0_input"

#define SENSOR_5V0_IO_SYS_W "/sys/bus/i2c/drivers/ina3221x/0-0042/iio:device2/in_power1_input"
#define SENSOR_5V0_IO_SYS_A "/sys/bus/i2c/drivers/ina3221x/0-0042/iio:device2/in_current1_input"
#define SENSOR_5V0_IO_SYS_V "/sys/bus/i2c/drivers/ina3221x/0-0042/iio:device2/in_voltage1_input"

#define SENSOR_3V3_SYS_W "/sys/bus/i2c/drivers/ina3221x/0-0042/iio:device2/in_power2_input"
#define SENSOR_3V3_SYS_A "/sys/bus/i2c/drivers/ina3221x/0-0042/iio:device2/in_current2_input"
#define SENSOR_3V3_SYS_V "/sys/bus/i2c/drivers/ina3221x/0-0042/iio:device2/in_voltage2_input"

#define SENSOR_3V3_IO_SLP_W "/sys/bus/i2c/drivers/ina3221x/0-0043/iio:device3/in_power0_input"
#define SENSOR_3V3_IO_SLP_A "/sys/bus/i2c/drivers/ina3221x/0-0043/iio:device3/in_current0_input"
#define SENSOR_3V3_IO_SLP_V "/sys/bus/i2c/drivers/ina3221x/0-0043/iio:device3/in_voltage0_input"

#define SENSOR_1V8_IO_W "/sys/bus/i2c/drivers/ina3221x/0-0043/iio:device3/in_power1_input"
#define SENSOR_1V8_IO_A "/sys/bus/i2c/drivers/ina3221x/0-0043/iio:device3/in_current1_input"
#define SENSOR_1V8_IO_V "/sys/bus/i2c/drivers/ina3221x/0-0043/iio:device3/in_voltage1_input"

#define SENSOR_3V3_SYS_M2_W "/sys/bus/i2c/drivers/ina3221x/0-0043/iio:device3/in_power2_input"
#define SENSOR_3V3_SYS_M2_A "/sys/bus/i2c/drivers/ina3221x/0-0043/iio:device3/in_current2_input"
#define SENSOR_3V3_SYS_M2_V "/sys/bus/i2c/drivers/ina3221x/0-0043/iio:device3/in_voltage2_input"

#define TEGRA_NUM_OF_CPU_CORES 6

// 345600 499200 652800 806400 960000 1113600 1267200 1420800 1574400 1728000 1881600 2035200 
#define TEGRA_LITTLE_CORE_ID 3
#define TEGRA_LITTLE_MAX_FREQ 2035200
#define TEGRA_LITTLE_MIN_FREQ 499200 

#define TEGRA_BIG_CORE_ID 1
#define TEGRA_BIG_MAX_FREQ 2035200
#define TEGRA_BIG_MIN_FREQ 499200

#define TEGRA_CPU_FREQ_STEP 153600

// 114750 216750 318750 420750 522750 624750 726750 854250 930750 1032750 1122000 1236750 1300500
#define TEGRA_GPU_MAX_FREQ 1300500
#define TEGRA_GPU_MIN_FREQ 216750
#define TEGRA_GPU_FREQ_STEP 102000

#define AUTO_CPU_FREQ_SCALING "/sys/module/qos/parameters/enable"
#define AUTO_M_CLUSTER_FREQ_SCALING "/sys/kernel/debug/tegra_cpufreq/M_CLUSTER/cc3/enable"
#define AUTO_B_CLUSTER_FREQ_SCALING "/sys/kernel/debug/tegra_cpufreq/B_CLUSTER/cc3/enable"

#define CPU_PATH "/sys/devices/system/cpu/cpu"
#define CPU_CURR_STATUS "/online"
#define CPU_CURR_FREQ "/cpufreq/cpuinfo_cur_freq"
#define CPU_CURR_SCAL_FREQ "/cpufreq/scaling_cur_freq"
#define CPU_SET_FREQ_MAX "/cpufreq/scaling_max_freq"
#define CPU_SET_FREQ_MIN "/cpufreq/scaling_min_freq"
#define CPU_FREQ_SCALING_GOV "/cpufreq/scaling_governor"
#define CPU_AVAILABLE_FREQS "/cpufreq/scaling_available_frequencies"
#define CPU_PERFORMANCE_GOVERNOR "performance"
#define CPU_DEFAULT_GOVERNOR "interactive"
#define CPU_SCHEDUTIL_GOVERNOR "schedutil"
#define NVPMODEL_M0 "sudo nvpmodel -m 0"

#define MEM_CURR_FREQ "/sys/kernel/debug/clk/emc/clk_rate"

#define GPU_MIN_FREQ "/sys/devices/gpu.0/devfreq/17000000.gp10b/min_freq"
#define GPU_MAX_FREQ "/sys/devices/gpu.0/devfreq/17000000.gp10b/max_freq"
#define GPU_UTILIZATION "/sys/devices/17000000.gp10b/load"
#define GPU_CURR_FREQ "/sys/devices/gpu.0/devfreq/17000000.gp10b/cur_freq"
#define GPU_SET_FREQ_MIN "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/min_freq"
#define GPU_SET_FREQ_MAX "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/max_freq"
#define GPU_FREQ_SCALING_GOV "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/governor"
#define GPU_DEFAULT_GOVERNOR "nvhost_podgov"
#define GPU_AVAILABLE_FREQS "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/available_frequencies"

#define FAN_CURR_SPEED "/sys/kernel/debug/tegra_fan/cur_pwm"
#define FAN_SET_SPEED "/sys/kernel/debug/tegra_fan/target_pwm"
#define FAN_STEP_TIME "/sys/kernel/debug/tegra_fan/step_time"
#define DEFAULT_FAN_STEP_TIME 100

#define POW_WINDOW_SIZE 6


class GetNode_Tegra {
public:
  //Constructor and Destructor
  GetNode_Tegra();
  ~GetNode_Tegra();

  //getters for status, frequency and temperature
  bool isAutoCPUFreqScaling();
  bool isCPUActive(int cpuNum);
  //DO NOTE: there are two different getters for the CPU freq since the actual one (getCPUCurFreq) use to oscillate. 
  //so with the second getter (getCPUCurScalFreq) it is possible to check which is the value that was set
  int getCPUCurFreq(int cpuNum);
  int getCPUCurScalFreq(int cpuNum);
  int getGPUCurFreq();
  int getGPUCurUtilization();
  int getMemCurFreq();
  std::vector<int> getAvailableCPUFreq(int cpuNum);
  std::vector<int> getAvailableGPUFreq();
  std::vector<int> getAvailableMemFreq();
//  int getCPUTemp(int cpuNum);

  //setters for status and frequency
  void setAutoCPUFreqScaling(bool status);
  void setCPUActive(int cpuNum, bool status);
  void setCPUFreq(int cpuNum, int freq);  
  void setGPUFreq(int freq);
  void setMemFreq(int freq);

  // set governor for frequency scaling (performance, schedutil)
  void setCPUGovernor(int cpuNum, std::string gov);
  void setGPUGovernor(std::string gov);
  
  //get fan speed
  int getFanSpeed();
  void setFanSpeed(int pwm);
  int getFanStepTime();
  void resetFanStepTime();
  void setFanStepTime(int time);

  //getter for current, voltage and power consumption
  float getGpuV();
  float getGpuA();
  float getGpuW();
  float getSocV();
  float getSocA();
  float getSocW();
  float getWifiV();
  float getWifiA();
  float getWifiW();
  float getVddInV();
  float getVddInA();
  float getVddInW();
  float getCpuV();
  float getCpuA();
  float getCpuW();
  float getDdrV();
  float getDdrA();
  float getDdrW();
  float getMuxV();
  float getMuxA();
  float getMuxW();
  float getVdd5v0IoV();
  float getVdd5v0IoA();
  float getVdd5v0IoW();
  float getVdd3V3SysV();
  float getVdd3V3SysA();
  float getVdd3V3SysW();
  float getVdd3V3IoSlpV();
  float getVdd3V3IoSlpA();
  float getVdd3V3IoSlpW();
  float getVdd1V8IoV();
  float getVdd1V8IoA();
  float getVdd1V8IoW();
  float getVdd3V3SysM2V();
  float getVdd3V3SysM2A();
  float getVdd3V3SysM2W();
  
  //getters for average power consumption on a sliding window of POW_WINDOW_SIZE samples
  float getGpuWavg();
  float getSocWavg();
  float getWifiWavg();
  float getVddInWavg();
  float getCpuWavg();
  float getDdrWavg();
  float getMuxWavg();
  float getVdd5v0IoWavg();
  float getVdd3V3SysWavg();
  float getVdd3V3IoSlpWavg();
  float getVdd1V8IoWavg();
  float getVdd3V3SysM2Wavg();
  
  //getters of the (average) power consumption of the overall chip/board //TODO board
  float getChipW();
  float getChipWavg();
  float getBoardW();
  float getBoardWavg();
  
  void updateSensing(); //PAY ATTENTION: this function has to be called to update sensed data. therefore BEFORE using above getters

//  float estimate_power(std::vector<int> utils, int bigFreq, int littleFreq);

private:
  //helper attribute
  std::string cpu_online_list[TEGRA_NUM_OF_CPU_CORES];
  std::string cpu_freq_list[TEGRA_NUM_OF_CPU_CORES];
  std::string cpu_scal_freq_list[TEGRA_NUM_OF_CPU_CORES];
  std::string cpu_avail_freq_list[TEGRA_NUM_OF_CPU_CORES];
  std::string cpu_freq_gov_list[TEGRA_NUM_OF_CPU_CORES];
  std::string cpu_set_freq_max_list[TEGRA_NUM_OF_CPU_CORES];
  std::string cpu_set_freq_min_list[TEGRA_NUM_OF_CPU_CORES];

  //variables where last read is stored
  float gpuV, gpuA, gpuW;
  float socV, socA, socW;
  float wifiV, wifiA, wifiW;
  float vddInV, vddInA, vddInW;
  float cpuV, cpuA, cpuW;
  float ddrV, ddrA, ddrW;
  float muxV, muxA, muxW;
  float vdd5v0IoV, vdd5v0IoA, vdd5v0IoW;
  float vdd3V3SysV, vdd3V3SysA, vdd3V3SysW;
  float vdd3V3IoSlpV, vdd3V3IoSlpA, vdd3V3IoSlpW;
  float vdd1V8IoV, vdd1V8IoA, vdd1V8IoW;
  float vdd3V3SysM2V, vdd3V3SysM2A, vdd3V3SysM2W;
  
  //sum of last POW_WINDOW_SIZE samples
  float gpuWs, socWs, wifiWs, vddInWs, cpuWs, ddrWs, muxWs, vdd5v0IoWs, vdd3V3SysWs, vdd3V3IoSlpWs, vdd1V8IoWs, vdd3V3SysM2Ws;

  //Attributes used to compute the mean power consumption of the last POW_WINDOW_SIZE periodic invocations of updateWavg() function
  //array of last POW_WINDOW_SIZE samples
  float gpuWavg[POW_WINDOW_SIZE];
  float socWavg[POW_WINDOW_SIZE];
  float wifiWavg[POW_WINDOW_SIZE];
  float vddInWavg[POW_WINDOW_SIZE];
  float cpuWavg[POW_WINDOW_SIZE];
  float ddrWavg[POW_WINDOW_SIZE];
  float muxWavg[POW_WINDOW_SIZE];
  float vdd5v0IoWavg[POW_WINDOW_SIZE];
  float vdd3V3SysWavg[POW_WINDOW_SIZE];
  float vdd3V3IoSlpWavg[POW_WINDOW_SIZE];
  float vdd1V8IoWavg[POW_WINDOW_SIZE];
  float vdd3V3SysM2Wavg[POW_WINDOW_SIZE];

  //index of the next position of the above arrays to be written
  int slidingIndex;

  //manage sensor drivers
  float read_sensor(const char *node);
  void update_read();
};

#endif // GETNODE_H
