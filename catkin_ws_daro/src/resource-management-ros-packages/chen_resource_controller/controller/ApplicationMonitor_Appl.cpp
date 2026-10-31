#include "ApplicationMonitor.h"
#include "MappingConsts.h"
#include <unistd.h>
#include <iostream>

std::vector<int> appls_to_cluster = {-1, -1, -1, -1, -1};

data_t* monitorAttach (const char* applName, long double req_thr, int mapping, int maxThreads, bool isOpenX) {
  //get controller pid
  pid_t pid = getMonitorPid(DEFAULT_CONTROLLER_NAME);

  /*--- begin data initialization ------*/
  //create shared memory to store application data_t structure
  int segmentId = shmget(getpid(), sizeof(data_t), IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR);
  data_t* data = (data_t*) shmat(segmentId, 0, 0);
  if (segmentId == -1 || data == (void*)-1){
    std::cout << "Something wrong happened during shared memory initialization" << std::endl;
    exit(EXIT_FAILURE); 
  }

  //get current time and date
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);

  //initialize data_t structure
  data->segmentId = segmentId;
  data->useGPU = (mapping == MAP_GPU_ID);
  data->usleepTime = 0;
  data->precisionLevel = 0;
  data->numThreads = 1;
  data->reqThr = req_thr;

  //first entry contains 0 ticks at time zero (from application start!)
  data->ticks[0].value = 0;
  data->ticks[0].time  = 0;
  data->curr = 0; 
  data->startTime = tv.tv_sec + 0.000001*tv.tv_usec;
  data->lastTimeSample = data->startTime;
  data->unit_th_processing_time_ticks[0].value = 0;
  data->unit_th_processing_time_ticks[0].time = 0;
  data->curr_idx_unit_th_processing_times = 0;
  /*--- end data initialization --------*/

  /*--- begin monitor initialization ---*/
  //wait semaphore
  int semid = semget (pid, 1, 0);
  binarySemaphoreWait(semid);
  //attach to monitor_t structure in shared memory
  int shmid_monitor = shmget(pid, sizeof(monitor_t), 0);
  monitor_t* monitor = (monitor_t*) shmat (shmid_monitor, 0, 0);
  if (shmid_monitor == -1 || semid == -1 || monitor == (void*)-1){
    std::cout << "Something wrong happened during controller shared memory attaching" << std::endl;
    exit(EXIT_FAILURE);
  }
  
  //save application data in the monitor
  if(monitor->nAttached == MAX_NUM_OF_APPLS - 1){
    std::cout << "Maximum number of manageable applications reached!" << std::endl;
    exit (EXIT_FAILURE);
  }
  monitor->appls[monitor->nAttached].alreadyInit = false;
  monitor->appls[monitor->nAttached].segmentId   = segmentId;
  monitor->appls[monitor->nAttached].pid         = getpid();
  monitor->appls[monitor->nAttached].isOpenX     = isOpenX;
  monitor->appls[monitor->nAttached].mapping     = mapping;
  monitor->appls[monitor->nAttached].maxThreads  = maxThreads;
  strcpy(monitor->appls[monitor->nAttached].name, applName);

  monitor->nAttached++;

  //post semaphore
  binarySemaphorePost(semid);
  /*--- end monitor initialization -----*/

  return data;
}

int monitorDetach(data_t *data) { 
  //get controller pid
  pid_t pid = getMonitorPid(DEFAULT_CONTROLLER_NAME);

  //wait semaphore  
  int semid = semget (pid, 1, 0);
  binarySemaphoreWait(semid);
  //attach shared memory of the controller
  int shmid_monitor = shmget(pid, sizeof(monitor_t), 0);
  monitor_t* monitor = (monitor_t*) shmat (shmid_monitor, 0, 0);
  if (shmid_monitor == -1 || semid == -1 || monitor == (void*)-1) {
    std::cout << "Something wrong happened during controller shared memory attaching" << std::endl;
    exit(EXIT_FAILURE);
  }

  //clean the entry of the current application in the monitor_t structure. we copy subsequent records a position behind and we decrease n_attached value
  int i = 0;
  for (; monitor->appls[i].segmentId != data->segmentId && i < MAX_NUM_OF_APPLS; i++);

  if(i == MAX_NUM_OF_APPLS){
      std::cout << "Segment_id " << data->segmentId << "  not found!" << std::endl; //may it happen?
      exit (EXIT_FAILURE);
  }

  for (; i < monitor->nAttached-1; i++) {
    //DO NOTE: direct assignment of data structures
    monitor->appls[i] = monitor->appls[i + 1];
  }
  monitor->nAttached--;

  //add the pid of the completed application to the deattached list.
  if(monitor->nDetached == MAX_NUM_OF_APPLS-1) {
    std::cout << "Maximum number of manageable deattached applications reached!" << std::endl;
    exit (EXIT_FAILURE);
  }
  monitor->detached[monitor->nDetached] = getpid();
  monitor->nDetached++;

  //post semaphore
  binarySemaphorePost(semid);

  //delete shared memory for the application data_t structure
  int segmentId = data->segmentId;
  int retval1 = shmdt(data); //we are removing the memory, therefore we need to previously save the id...
  int retval2 = shmctl(segmentId, IPC_RMID, 0);
  if(retval1 == -1 || retval2 == -1) {
    std::cout << "Something wrong happened during shared memory deletion!" << std::endl;
    exit (EXIT_FAILURE);
  }

  return 0;
}

void monitorTick (data_t *data) {
  monitorTick_v(data, 1);
}

void monitorTick_v (data_t *data, int value) {
  //TODO do we need semaphors for ticks?
  //data tick increment
  int next = (data->curr + 1) % MAX_TICKS_SIZE; 

  //get current time and date
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);

  //save data in the next position
  data->ticks[next].value = data->ticks[data->curr].value + value;
  data->ticks[next].time = (tv.tv_sec + 0.000001*tv.tv_usec) - data->startTime;
  
  //update reference to the current position
  data->curr = next;
}

pid_t getMonitorPid(const char* monitorName) {
  pid_t pid;
  // get controller pid given its name
  std::string pipe = std::string("pgrep ") + monitorName;
  FILE *fp;
  fp = popen(pipe.c_str(), "r");
  if (fp == NULL){
    std::cout << "can't read controller PID" << std::endl;
    exit (EXIT_FAILURE); 
  }
  int ret = fscanf(fp, "%d", &pid); //get controller PID
  if (feof(fp)){
    std::cout << "can't read controller PID" << std::endl;
    exit (EXIT_FAILURE); 
  }
  pclose(fp);
  return pid;
}

void setReqThroughput(data_t* data, long double reqThr){
  data->reqThr = reqThr;
}

void setUnitThProcessingTime(data_t* data, float unit_th_processing_time){
  //std::cout << "setUnitThProcessingTime\n";
  int next = (data->curr_idx_unit_th_processing_times + 1) % MAX_TICKS_SIZE; 

  //get current time and date
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);

  //save data in the next position
  data->unit_th_processing_time_ticks[next].value = (long long) (unit_th_processing_time*1e9);
  data->unit_th_processing_time_ticks[next].time = (tv.tv_sec + 0.000001*tv.tv_usec) - data->startTime;
  
  // std::cout << next << "," << data->unit_th_processing_time_ticks[next].time << "," << data->unit_th_processing_time_ticks[next].value << "," << unit_th_processing_time << "\n";

  //update reference to the current position
  data->curr_idx_unit_th_processing_times = next;
  
  // data->unit_th_processing_time = 0.01*unit_th_processing_time + 0.99*data->unit_th_processing_time;
}


void autosleep(data_t* data, long double ref_thr) {
  //get timestamp
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  long double curr_time = tv.tv_sec + 0.000001*tv.tv_usec; //in seconds!
  data->usleepTime = (1.0/ref_thr - (curr_time - data->lastTimeSample))*1000000;
  //printf("%d %f %f\n", data->usleepTime, data->lastTimeSample, 1.0/ref_thr);
  if(data->usleepTime>0)
    usleep(data->usleepTime);
  else
    data->usleepTime = 0;
  gettimeofday(&tv, &tz);
  data->lastTimeSample = tv.tv_sec + 0.000001*tv.tv_usec; //in seconds!  
}

bool useGPU(data_t *data) {
  return data->useGPU;
}

int useUsleepTime(data_t* data) {
  return data->usleepTime;
}

int usePrecisionLevel(data_t *data) {
  return data->precisionLevel;
}

int useNumThreads(data_t *data) {
  return data->numThreads;
}

