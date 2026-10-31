#include "utilization.h"
#include <unistd.h>
#include <cstdio>	// sprintf
#include <cstdlib> // atoi
#include <cstring>	// strncmp
#include <string>
#include <iostream>	// cerr

Utilization::Utilization(int numOfCores) {
  this->numOfCores=numOfCores;
  for(int i=0; i<numOfCores; i++){
    mOldUserCPU.push_back(0);
    mOldSystemCPU.push_back(0);
    mOldIdleCPU.push_back(0);
  }
  this->getCPUUtilization();
}

Utilization::~Utilization() {
}

int Utilization::calUtilization(int cpu_idx, long int user, long int nice, long int system, long int idle, long int iowait, long int irq, long int softirq, long int steal) {
  long int total = 0;
  int usage = 0;
  long int diff_user, diff_system, diff_idle;
  long int curr_user, curr_system, curr_idle;

  curr_user = (user+nice);
  curr_system = (system+irq+softirq+steal);
  curr_idle = (idle+iowait);

  diff_user = mOldUserCPU[cpu_idx] - curr_user;
  diff_system = mOldSystemCPU[cpu_idx] - curr_system;
  diff_idle = mOldIdleCPU[cpu_idx] - curr_idle;

  total = diff_user + diff_system + diff_idle;
  if (total != 0)
    usage = (diff_user + diff_system) * 100 / total;

  mOldUserCPU[cpu_idx] = curr_user;
  mOldSystemCPU[cpu_idx] = curr_system;
  mOldIdleCPU[cpu_idx] = curr_idle;

  return usage;
}

std::vector<int> Utilization::getCPUUtilization() {
  char buf[80] = {'\0',};
  char cpuid[8] = {'\0',};
  long int user, system, nice, idle, iowait, irq, softirq, steal;
  FILE *fp;
  int cpu_index;
  std::vector<int> currUtilizations;
  int error;

  for(int i = 0; i < this->numOfCores; i++)
    currUtilizations.push_back(0);

  fp = fopen("/proc/stat", "r");
  if (fp == NULL)
    return currUtilizations;

  error = 0;
  if(fgets(buf, 80, fp)==NULL) //discard first line since it contains aggregated values for the CPU
    error= 1; 

  cpu_index = 0;
    
  char *r = fgets(buf, 80, fp);
  while (r && cpu_index < numOfCores) {
      char temp[5] = "cpu";
      temp[3] = '0' + cpu_index;
      temp[4] = '\0';
      
      if(!strncmp(buf, temp, 4)){
          sscanf(buf, "%s %ld %ld %ld %ld %ld %ld %ld %ld", cpuid, &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal);
          currUtilizations[cpu_index] = calUtilization(cpu_index, user, nice, system, idle, iowait, irq, softirq, steal);
          r = fgets(buf, 80, fp);
      }
      cpu_index++;
  }
  fclose(fp);

  return currUtilizations;
}


