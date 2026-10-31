#ifndef UTILIZATION_H
#define UTILIZATION_H

#include <string>
#include <vector>

class Utilization {
public:
  //Constructor and Destructor    
  Utilization(int numOfCores);
  ~Utilization();

  //getters utilization of utilization of all the cores
  std::vector<int> getCPUUtilization();
        
private:
  //helper method to compute utilization
  int calUtilization(int cpu_idx, long int user, long int nice, long int system, long int idle, long int iowait, long int irq, long int softirq, long int steal);

  //since core usage is computed between two invocations of the getCPUUitlization() function. 
  //it is necessary to save values collected at the previous invocation to compute the difference with the currently sampled values
  std::vector<long int> mOldUserCPU;
  std::vector<long int> mOldSystemCPU;
  std::vector<long int> mOldIdleCPU;
  int numOfCores;
};

#endif 
