#include "ApplicationMonitor.h"
#include <fstream>
#include <iostream>
#include <queue>
#include <signal.h>
#include "CGroupUtils.h"

monitor_t* monitorInit () {
  //Init CGroups!
  int error = CGUtils::Setup();
  if (error==-1){
    std::cout << "Error configuring Cgroups" << std::endl;
    exit(EXIT_FAILURE);
  }

  //initialize shared memory and the companion semaphore for the controller
  //(semaphore is used for mutually exclusive access between applications and controller)
  int semid = semget (getpid(), 1, IPC_CREAT | IPC_EXCL | 0666);
  union semun argument;
  unsigned short values[1];
  values[0] = 1;
  argument.array = values;
  int semval = semctl (semid, 0, SETALL, argument);
  int shmid = shmget(getpid(), sizeof(monitor_t), IPC_CREAT | IPC_EXCL | 0666);
  monitor_t* monitor = (monitor_t*) shmat(shmid, 0, 0);
  if (semid == -1 || semval == -1 || shmid == -1 || monitor==(void*)-1){
    std::cout << "Something wrong happened during shared memory initialization" << std::endl;
    exit(EXIT_FAILURE);
  }

  //initialize counters of applications
  monitor->nAttached = 0;
  monitor->nDetached = 0;

  //map the controller on the first cpu cluster
  pid_t pid = getpid();
  error = CGUtils::Initialize(pid);
  if (error==-1){
    std::cout << "monitorInit (pid) error initializing Cgroups" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::vector<int> cpu0;
  cpu0.push_back(0);
  error = CGUtils::UpdateCpuSet(pid, cpu0);
  if (error==-1){
    std::cout << "Error setting Cgroups" << std::endl;
    exit(EXIT_FAILURE);
  }

  return monitor;
}

void monitorDestroy (monitor_t* monitor) {
  //destroy shared memories containing data_t data structures for all just-killed applications
  for(int j=0; j < monitor->nAttached; j++){
    //delete shared memory for the application data_t structure
    int segmentId = monitor->appls[j].segmentId;
    int retval1 = shmctl(segmentId, IPC_RMID, 0);
    if(retval1 == -1) {
      std::cout << "Something wrong happened during shared memory deletion!" << std::endl;
      exit (EXIT_FAILURE);
    }
  }
  //destroy shared memory containing the monitor data structure and related semaphore. if some crash happens (I don't think so) it may be
  //because taskset data structures are stored in the shared memory and sched_setaffinity tries to access a destroied memory location.
  //in that case try to find a solution...
  shmdt(monitor);
  int shmid = shmget (getpid(), sizeof(monitor_t), 0);
  int semid = semget (getpid(), 1, 0);
  int semval1 = shmctl (shmid, IPC_RMID, 0);
  union semun ignored_argument;
  int semval2 = semctl (semid, 1, IPC_RMID, ignored_argument);
  if (shmid == -1 || semid == -1 || semval1 == -1 || semval2 == -1){
    std::cout << "Something wrong happened during shared memory initialization" << std::endl;
    exit(EXIT_FAILURE);
  }
  pid_t pid = getpid();
  int error = CGUtils::Remove(pid);
  if (error==-1){
    std::cout << "Error removing Cgroups" << std::endl;
    exit(EXIT_FAILURE);
  }
  
  //Init CGroups!
  error = CGUtils::Destroy();
  if (error==-1){
    std::cout << "Error destroying Cgroups" << std::endl;
    exit(EXIT_FAILURE);
  }
}

data_t monitorRead (int segmentId) {
  //return data_t structure
  data_t* shm = (data_t*) shmat(segmentId, 0, 0); //attach shared memory
  if(shm == (void*) -1){
    std::cout << "Something wrong happened during shared memory access!" << std::endl;
  }
  data_t retval;
  memcpy(&retval, shm, sizeof (data_t));
  int ret1 = shmdt(shm); //detach shared memory
  if(ret1 == -1){
    std::cout << "Something wrong happened during shared memory detach!" << std::endl;
  }
  return retval;
}

data_t* monitorPtrRead (int segmentId) {
  //return data_t structure
  data_t* shm = (data_t*) shmat(segmentId, 0, 0); //attach shared memory
  if(shm == (void*) -1){
    std::cout << "Something wrong happened during shared memory access!" << std::endl;
  }
  return shm;
}

void monitorWrite(int segmentId, data_t* data) {
    // Attach to shared memory
    data_t* shm = static_cast<data_t*>(shmat(segmentId, nullptr, 0));
    if (shm == (void*)-1) {
        std::cerr << "Error: Could not attach to shared memory for writing!" << std::endl;
        return;
    }

    // Write data into shared memory
    memcpy(shm, data, sizeof(data_t));

    // Detach from shared memory
    if (shmdt(shm) == -1) {
        std::cerr << "Error: Could not detach from shared memory after writing!" << std::endl;
    }
}

double getGlobalThroughput (data_t *data) {
  if(data->curr == 0 && data->ticks[data->curr].time == 0)
    return 0;
  //get current time and date
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  long double currTime = (tv.tv_sec + 0.000001*tv.tv_usec) - data->startTime;

  return data->ticks[data->curr].value / currTime;
}

double getCurrThroughput (data_t *data, int TPwindow) {
  //no data in the structure!
  if(data->curr == 0 && data->ticks[data->curr].time == 0)
    return 0;

  //get current time and date
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  long double currTime = (tv.tv_sec + 0.000001*tv.tv_usec) - data->startTime;

  //no tick in the previous period
  if(currTime - data->ticks[data->curr].time > TPwindow)
    return 0;

  //we assume to have at least 2 values in the structure. so if data->curr==0 and we passed the check at the beginning of the function,
  //the buffer is completelly full and we can move back to the last element!
  int i = (data->curr==0)? MAX_TICKS_SIZE-1 : data->curr-1;
  while(currTime - data->ticks[i].time < TPwindow && !(i == 0 && data->ticks[i].time == 0) && i!=data->curr){
    if(i>0) //i = (i+MAX_TICKS_SIZE-1)%MAX_TICKS_SIZE; //it is the same
      i--;
    else
      i = MAX_TICKS_SIZE-1;
  }
  long double actualWindow = TPwindow;
  if(i == 0 && data->ticks[i].time == 0) //do note: at the beginning the time window is smaller than the prefedined one
    actualWindow = data->ticks[data->curr].time;
  else if(i == data->curr){ //this is the case in which the application is too fast and it fills more than once the whole window between two iterations of the controller!
    //so we have to increment i to count the whole list and
    i = (i+1) % MAX_TICKS_SIZE;
    actualWindow = data->ticks[data->curr].time - data->ticks[i].time;
  }
  return (data->ticks[data->curr].value - data->ticks[i].value) / actualWindow;
}

double getReqThroughput(data_t* data){
  return data->reqThr;
}

float getUnitThProcessingTime(data_t* data){
  //no data in the structure!
  if(data->curr_idx_unit_th_processing_times == 0 && data->unit_th_processing_time_ticks[data->curr_idx_unit_th_processing_times].time == 0)
    return 0;

  //get current time and date
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  long double currTime = (tv.tv_sec + 0.000001*tv.tv_usec) - data->startTime;

  
  int curr_idx = data->curr_idx_unit_th_processing_times;
  int end_idx = (curr_idx + 1) % MAX_TICKS_SIZE;
  std::vector<float> tps;
  float tp;
  while (data->unit_th_processing_time_ticks[curr_idx].time > currTime - 1.0 && curr_idx != end_idx) {
    tp = (float (data->unit_th_processing_time_ticks[curr_idx].value))*1e-9;
    tps.push_back(tp);
    // std::cout << curr_idx << "," << data->unit_th_processing_time_ticks[curr_idx].time << "," << data->unit_th_processing_time_ticks[curr_idx].value << "\n";
    curr_idx = (curr_idx - 1 + MAX_TICKS_SIZE) % MAX_TICKS_SIZE;
  }
  std::sort(tps.begin(), tps.end());
  int q85 = int(tps.size() * 0.85);
  int q50 = int(tps.size() * 0.5);
  return tps[q85];
  // return data->unit_th_processing_time;
}


void printAttachedApplications(monitor_t* monitor){
  //display application table
  std::cout << "---------------------------" << std::endl;
  std::cout << "attached appls: " << monitor->nAttached << std::endl;
  for(int j=0; j < monitor->nAttached; j++){
    data_t data = monitorRead(monitor->appls[j].segmentId);
    std::cout << monitor->appls[j].name << " " << monitor->appls[j].pid << " global tp: " << getGlobalThroughput(&data) << " current tp: " << getCurrThroughput(&data) << std::endl;
  }
  std::cout << "---------------------------" << std::endl;
}

void setUseGPU(data_t *data, bool value) {
  data->useGPU = value;
}

void setPrecisionLevel(data_t *data, int value) {
  data->precisionLevel = value;
}

void setUsleepTime(data_t *data, int value) {
  data->usleepTime = value;
}

void setNumThreads(data_t *data, int value) {
  data->numThreads = value;
}

std::vector<pid_t> updateAttachedApplications(monitor_t* monitor, bool check_died){
  std::vector<pid_t> newAppls;
  std::vector<pid_t> diedAppls;
  //look for new applications
  for(int j=0; j < monitor->nAttached; j++){
    if(!monitor->appls[j].alreadyInit){ //if the application is new
      //init cgroups and keep track about the application
      //DO NOTE: for OpenX applications it is necessary to register each single tid otherwise CGroups does not work properly
      if(monitor->appls[j].isOpenX){
        std::vector<pid_t> pids = getAppPids(monitor->appls[j].pid);
        std::vector<int>::iterator p;
        for (p = pids.begin(); p != pids.end(); p++) {
            int error=CGUtils::Initialize(*p);
            if (error==-1){
              std::cout << "updateAttachedApplications 1 error initializing Cgroups." << std::endl;
              continue;
              // exit(EXIT_FAILURE);
            }      
        }
      } else{
        int error = CGUtils::Initialize(monitor->appls[j].pid);
        if (error==-1){
          std::cout << "updateAttachedApplications 2 error initializing Cgroups." << std::endl;
          exit(EXIT_FAILURE);
        }      
      }
      
      //save pid in the data structure
      monitor->appls[j].alreadyInit = true;
      newAppls.push_back(monitor->appls[j].pid);
    }
    if(check_died && !isRunning(monitor->appls[j].pid)){
      //we add to the died list and we process them later
      diedAppls.push_back(monitor->appls[j].pid);
    }
  }
  if(check_died){
    //wait semaphore since I'm going to modify the monitor_t structure
    pid_t mypid = getpid();
    int semid = semget (mypid, 1, 0);
    binarySemaphoreWait(semid);
    for(int j=0; j<diedAppls.size(); j++){
      //search the entry of the died application in the monitor_t data structure
      int i = 0;
      for (; monitor->appls[i].pid != diedAppls[j] && i < MAX_NUM_OF_APPLS; i++);
      if(i == MAX_NUM_OF_APPLS){
          std::cout << "Pid " << diedAppls[j] << "  not found!" << std::endl; //may it happen?
          exit (EXIT_FAILURE);
      }

      //delete shared memory for the application data_t structure
      int segmentId = monitor->appls[i].segmentId;
      int retval1 = shmctl(segmentId, IPC_RMID, 0);
      if(retval1 == -1) {
        std::cout << "Something wrong happened during shared memory deletion!" << std::endl;
        exit (EXIT_FAILURE);
      }

      //clean entry of the current application in the monitor_t structure.
      //we copy subsequent records a position behind and we decrease n_attached value
      for (; i < monitor->nAttached-1; i++) {
        monitor->appls[i] = monitor->appls[i + 1];
      }
      monitor->nAttached--;

      //add the pid of the completed application to the deattached list.
      if(monitor->nDetached == MAX_NUM_OF_APPLS-1) {
        std::cout << "Maximum number of manageable deattached applications reached!" << std::endl;
        exit (EXIT_FAILURE);
      }
      monitor->detached[monitor->nDetached] = diedAppls[j];
      monitor->nDetached++;
    }
    //post semaphore
    binarySemaphorePost(semid);
  }
  //scan list of detatched applications to reset cgroups and deregister them
  for(int j=0; j < monitor->nDetached; j++){
    std::cout << "removing " << monitor->detached[j] << std::endl;
    int error = CGUtils::Remove(monitor->detached[j]);
    if (error==-1){
      std::cout << "Error removing Cgroups" << std::endl;
      exit(EXIT_FAILURE);
    }
    //clean all settings for OpenX applications, if any    
    std::vector<pid_t> pids = getAppPids(monitor->detached[j]);
    std::vector<int>::iterator p;
    for (p = pids.begin(); p != pids.end(); p++) {
        int error=CGUtils::Remove(*p);
        if (error==-1){
          std::cout << "Error removing Cgroups" << std::endl;
          exit(EXIT_FAILURE);
        }      
    }
  }
  monitor->nDetached = 0; //deleting list...
  return newAppls; //return pids of new appls
}

bool isRunning(pid_t pid) {
  std::string cmdname = std::string("/proc/") + std::to_string(pid) + std::string("/status");
  FILE *fp;
  fp = fopen(cmdname.c_str(), "r");
  if (fp){
    fclose(fp);
    return true;
  }
  return false;
}

void killAttachedApplications(monitor_t* monitor){
  //remove all cgroup settings for running applications.
  for(int j=0; j < monitor->nAttached; j++){
    int error = CGUtils::Remove(monitor->appls[j].pid);
    if (error==-1){
      std::cout << "Error removing Cgroups" << std::endl;
      exit(EXIT_FAILURE);
    }
    std::vector<pid_t> pids = getAppPids(monitor->appls[j].pid);
    std::vector<int>::iterator p;
    for (p = pids.begin(); p != pids.end(); p++) {
        int error=CGUtils::Remove(*p);
        if (error==-1){
          std::cout << "Error removing Cgroups" << std::endl;
          exit(EXIT_FAILURE);
        }      
    }
    //kill process!
    kill(monitor->appls[j].pid, SIGKILL);
  }
}

std::vector<pid_t> getAppPids(pid_t appPid) {
  std::vector<pid_t> pids;
  std::string pipe = std::string("ps -e -T | grep ")+ std::to_string(appPid) + std::string(" | awk '{print $2}'");
  FILE *fp;
  fp = popen(pipe.c_str(), "r");
  if (fp == NULL){
    std::cout << "can't read application PIDs" << std::endl;
    exit (EXIT_FAILURE);
  }
  pid_t pid;
  fscanf(fp, "%d", &pid);
  while (!feof(fp)) {
    pids.push_back(pid);
    fscanf(fp, "%d", &pid);
  }
  /*if (feof(fp)){
    std::cout << "can't read application PIDs" << std::endl;
    exit (EXIT_FAILURE);
  }*/
  pclose(fp);
  return pids;
}

bool isOpenX(monitor_t* monitor, pid_t pid){
  int i;
  for(i=0; i < monitor->nAttached; i++)
    if(monitor->appls[i].pid == pid)
      return monitor->appls[i].isOpenX;
  std::cout << "can't find specified PID" << std::endl;
  exit (EXIT_FAILURE);  
}

void UpdateCpuSet(monitor_t* monitor, pid_t pid, std::vector<int> cores) {
  if(isOpenX(monitor, pid)){
    //DO NOTE: sometimes this call fails generating a list of failure messages on screen... in any case the pinning seems to work...
    std::vector<pid_t> pids = getAppPids(pid);
    std::vector<int>::iterator p;
    for (p = pids.begin(); p != pids.end(); p++) {
      int error=CGUtils::UpdateCpuSet(*p, cores);
      if (error==-1){
        std::cout << "Error updating Cgroups" << std::endl;
        exit(EXIT_FAILURE);
      }   
    }  
  } else {
    int error=CGUtils::UpdateCpuSet(pid, cores);
    if (error==-1){
      std::cout << "Error updating Cgroups" << std::endl;
      exit(EXIT_FAILURE);
    }   
  }
}

//DO NOTE: quota is set per entire application for non-OpenX applications. It does not work with OpenX applications
void UpdateCpuQuota(monitor_t* monitor, pid_t pid, float quota) {
   if(isOpenX(monitor, pid)){
/*    std::vector<pid_t> pids = getAppPids(pid);
    std::vector<int>::iterator p; 
    for (p = pids.begin(); p != pids.end(); p++) {
      int error=CGUtils::UpdateCpuQuota(*p, quota);
      if (error==-1){
        std::cout << "Error updating Cgroups" << std::endl;
        exit(EXIT_FAILURE);
      }   
    }  */
    std::cout << "Update quota does not work with OpenX applications" << std::endl;
    exit(EXIT_FAILURE);
  } else {
    int error = CGUtils::UpdateCpuQuota(pid, quota);  
    if (error==-1){
      std::cout << "Error updating Cgroups" << std::endl;
      exit(EXIT_FAILURE);
    }   
  }
}
