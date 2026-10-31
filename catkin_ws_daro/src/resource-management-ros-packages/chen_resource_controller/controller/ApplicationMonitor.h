#ifndef MONITOR_H_
#define MONITOR_H_

#include <sys/time.h>
#include <sched.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>

//used only in C++ code
#ifdef __cplusplus
#include <vector>
#endif

//constants with maximum sizes for the various sets (implemented by using static arrays for simplicity since stored in a shared memory)
#define MAX_TICKS_SIZE 128
#define MAX_NUM_OF_APPLS 128
#define MAX_NUM_OF_CPU_SETS 32
#define MAX_NAME_SIZE 100
#define DEFAULT_TIME_WINDOW 1
#define DEFAULT_CONTROLLER_NAME "controller"
#define ZERO_APPROX 1e-5

//in C code we need to define Boolean symbols since bool type is not available
#ifndef __cplusplus
typedef enum bool {false=0, true=1} bool;
#endif

/*
 * tick_t type is used to collect heartbeat
 */
typedef struct {
  long long   value; //number of accumulated ticks
  long double time;  //since the start time of the application
} tick_t;

/*
 * data_t type is used to specify the data (saved in the shared memory) exchanged between the controller and the application during the execution (performance data and actuation values).
 * A segment in the shared memory is created for each application
 */
typedef struct {
  int         segmentId;             // id of the memory segment where this structure is stored
  long double startTime;             // logged start time of the application
  //monitoring
  tick_t      ticks[MAX_TICKS_SIZE]; // vector of ticks. Actually this vector is used as a sliding window for efficiency reasons. therefore we have following pointers
  int         curr;                  // "pointer" to the current position where the last tick has been saved
  //requirement
  long double reqThr;                //  throughput to be guaranteed (if equal to 0, it is not set)
  //actuation
  long double  lastTimeSample;       // last timestamp that has been sampled. it is necessary to implement the autosleep function
  bool         useGPU;               // GPU/notCPU
  int          usleepTime;           //in microseconds
  int          precisionLevel;       //0:exact computation; >0 approximate computation. it represents the percentage of approximation
  int          numThreads;           //number of threads to set- ak
  float        unit_th_processing_time = 0 ;           // time to process one unit of throughput
  tick_t       unit_th_processing_time_ticks[MAX_TICKS_SIZE];
  int          curr_idx_unit_th_processing_times = 0;

} data_t;

/*
 * appl_t type is used as a descriptor of an application.
 */
typedef struct {
  pid_t       pid;                   // application pid
  char        name[MAX_NAME_SIZE+1]; // name of the application
  int         segmentId;             // id of the memory segment containing the corresponding data_t structure
  bool        isOpenX;               // specify if the application is an OpenCL/OpenMP/OpenCV one. in such a case threads have to be forced separaterly with Cgroups.
  bool        gpuImpl;               // has a GPU implementation or not
  int         maxThreads;            // 1= serial implementation; >1 parallel implementation
  int         mapping;               // 0=LITTLE, 1=big, 2=GPU
  bool        alreadyInit;    // states if CGroups have been already initialized by the controller for the application or not
  //this last field is used to understand if cgroups have been already configured for each registered application.
  //indeed we don't have a way for an application to signal the controller on the registration.
  //therefore the controller use to check periodically the application table (i.e. the monitor structure in the shared memory)
  //to detect new applications and configure them
  //DO NOTE: according to this asynchronous scheme it is necessary to check for new applications with a quite high frequency (1 sec);
  //then we should run the resource management policy with a lower frequency
} appl_t;

/*
 * monitor_t type is used to contain the status of the system (regarding to the running applications).
 * a single data structure is allocated for a controller and it is stored in a shared memory (just because in this way the transmission of the appl_t descriptor is easier).
 * TODO this may be refactored in some more efficient way
 */
typedef struct {
  //set of monitored applications
  appl_t  appls[MAX_NUM_OF_APPLS];
  int     nAttached; //current number of monitored applications

  //set of applications that have been detached recently. necessary to update above appls set
  pid_t   detached[MAX_NUM_OF_APPLS];
  int     nDetached; //current number of detached applications
} monitor_t;

//semaphore management, data structure and wait/post functions.
//The semaphore is used to access the monotor_t data structure in the shared memory since both the applications and the controller may write it
union semun {
  int                val;
  struct semid_ds    *buf;
  unsigned short int *array;
  struct seminfo     *__buf;
};

void binarySemaphoreWait(int);
void binarySemaphorePost(int);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//PAY ATTENTION: first set of functions can be used only in the application code, the second set only in the controller code
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//functions used in the application code. The application may be implemented either in C or C++. So we need to specify that the following
//piece of code is C++ and may be used in C
#ifdef __cplusplus
extern "C"
{
#endif
//attach the current application on the monitor
data_t* monitorAttach(char const*, long double, int, int, bool);
//Detach the application from the monitor
int monitorDetach(data_t*);
//Send a tick to the monitor which the application is connected
void monitorTick(data_t*); // DO NOTE: I cannot use default value since not compiant with C code...
void monitorTick_v(data_t*, int );
//Set new throughput
void setReqThroughput(data_t*, long double);
//Set new processing time for unit throughput
void setUnitThProcessingTime(data_t*, float);
//Get the current device to be used true=GPU/false=CPU
bool useGPU(data_t*);
//get the usleep time
int useUsleepTime(data_t*);
//Get the current precision level to be used (0 maximum - 100 minimum)
int usePrecisionLevel(data_t*);
//Get the number of threads of an application (preferably multiples of 2)
int useNumThreads(data_t*);
//get the controller pid given its name (usually "main")
pid_t getMonitorPid(const char*);
//sleep for a given amount of time to satisfy the required throughput
void autosleep(data_t*, long double);
#ifdef __cplusplus
}
#endif

//since this piece of code is used in the controller code, it will be C++ for sure.
//DO NOTE: not so elegant trick but it works
#ifdef __cplusplus
//functions used in the controller code
//initialize and destroy monitor_t data structure
monitor_t* monitorInit();
void monitorDestroy(monitor_t*);
//read (copy!) data_t values from the shared memory given a segment id
data_t monitorRead(int);
//get a reference of the shared memory given a segment id
data_t* monitorPtrRead (int);
//write data_t values to the shared memory given a segment id
void monitorWrite(int, data_t*);

//get current and global throughput
double getGlobalThroughput(data_t*);
double getCurrThroughput(data_t*, int = DEFAULT_TIME_WINDOW);
//get current required throughput
double getReqThroughput(data_t*);
//get current processing time of one throughput
float getUnitThProcessingTime(data_t* data);
//setters of various flags
void setUseGPU(data_t*, bool);
void setPrecisionLevel(data_t*, int);
void setUsleepTime(data_t*, int);
void setNumThreads(data_t*, int);

//print the status of the attached applications
void printAttachedApplications(monitor_t*);
//update the status (we have to call it at the beginning of the control loop). the last flag says if we have to look for died applications
std::vector<pid_t> updateAttachedApplications(monitor_t*, bool = true);
//check if a give application is running or not
bool isRunning(pid_t);
//kill all attached applications
void killAttachedApplications(monitor_t*);
//Get thread ids of the running application, in order to apply cgroups policy
std::vector<pid_t> getAppPids(pid_t);
//Return true if the application is an OpenCL/OpenMP/OpenCV one; otherwise false.
bool isOpenX(monitor_t* , pid_t);
//Set CPU cores where to run the application. WRAPPER to CGroups
void UpdateCpuSet(monitor_t*, pid_t, std::vector<int>);
//Set CPU quota where to run the application. WRAPPER to CGroups
void UpdateCpuQuota(monitor_t*, pid_t, float);

#endif

#endif /* MONITOR_H_ */
