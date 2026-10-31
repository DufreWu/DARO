#include "CGroupUtils.h"
#include "MappingConsts.h"
#include <iostream>
#include <libcgroup.h>
#include <vector>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

//CODE DOCUMENTATION:
//documentation of the libcgroup library: http://libcg.sourceforge.net/html/index.html
//documentation to the terminal commands (cgroup-tools): https://wiki.archlinux.org/index.php/cgroups
//cgcreate -a root -t root -g memory,cpu,cpuset:mycontroller
//echo 0-3 > /sys/fs/cgroup/cpuset/mycontroller/cpuset.cpus
//echo 0 > /sys/fs/cgroup/cpuset/mycontroller/cpuset.mems
//cgclassify -g cpu,cpuset:mycontroller/#PID #PID
// ls -R /sys/fs/cgroup/cpuset/mycontroller   #check exist
//cgdelete -r cpu,cpuset:mycontroller       #delete

#define MAINGROUP "/mycontroller" //"/user.slice/"

// using namespace std;

namespace CGUtils {

// NOTE: this should match the CGControllers enum
const char *controller[] = {
  "cpuset",
  "cpu",
  "freezer"
};

// Needed controllers
std::vector<int> controller_ids = {
  CPUSET,
  CPU,
  FREEZER
};

char *mounts[CGCONTROLLERS_COUNT] = { nullptr, };
const unsigned int cfs_period = 100000;

// To be called ONLY once when the orchestrator starts. It setups libcgroup
// internal info and searches for mount points.
int Setup() {
  int result = 0;
  std::string cg_mount_point = "";
  struct cgroup *p_cg;
  struct cgroup_controller *p_cg_contr_cpuset;
  struct cgroup_controller *p_cg_contr_cpu;
  struct cgroup_controller *p_cg_contr_freezer;

  // Creating a sample CGroup with the app PID
  std::string app_mount_point = MAINGROUP;
   
  // Initialization
  result = cgroup_init();
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  // Looking for controllers
  for (int id : controller_ids) {
    result = cgroup_get_subsys_mount_point(controller[id], &mounts[id]);
    if (result) {
      std::cerr << "Error: " << cgroup_strerror(result) << " -- " << controller[id] << std::endl;
      return -1;
    } else {
      std::cout << "CGroup controller [" << controller[id] << "] available at [" << mounts[id]<< "]" << std::endl;
    }
  }
  
  p_cg = cgroup_new_cgroup(app_mount_point.c_str());
  if (!p_cg) {
    std::cerr << "Error: cannot create " << app_mount_point << std::endl;
    return -1;
  }

  // Set CPUSET configuration (if available)
  if (!mounts[CPUSET]) {
    std::cerr << app_mount_point << ": CPUSET controller not mounted" << std::endl;
    return -1;
  }

  p_cg_contr_cpuset = cgroup_add_controller(p_cg, "cpuset");
  if (!p_cg_contr_cpuset) {
    std::cerr << app_mount_point << ": set CPUSET controller FAILED" << std::endl;
    return -1;
  }

  //DO NOTE: we need to specify the entire set of cores in the system, otherwise not included cores
  // cannot be used in any mapping commands. This piece of info may be read from the cgroup fs (from the root folder)
  // but I'm not able to do it right now
  std::string cpus = std::string("0-") + std::to_string(NUM_OF_CORES - 1);
  result = cgroup_set_value_string(p_cg_contr_cpuset, "cpuset.cpus", cpus.c_str());
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  result = cgroup_set_value_string(p_cg_contr_cpuset, "cpuset.mems", "0");
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  // Set CPU configuration
  if (!mounts[CPU]) {
    std::cerr << app_mount_point << ": CPU controller not configured" << std::endl;
    return -1;
  }

  p_cg_contr_cpu = cgroup_add_controller(p_cg, "cpu");
  if (!p_cg_contr_cpu) {
    std::cerr << app_mount_point << ": set CPU controller FAILED" << std::endl;
    return -1;
  }

  result = cgroup_set_value_string(p_cg_contr_cpu, "cpu.cfs_period_us", std::to_string(cfs_period).c_str());
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  result = cgroup_set_value_string(p_cg_contr_cpu, "cpu.cfs_quota_us", "-1");
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  //Set freezer configuration
  if (!mounts[FREEZER]) {
    std::cerr << app_mount_point << ": FREEZER controller not mounted" << std::endl;
      return -1;
  }

  p_cg_contr_freezer = cgroup_add_controller(p_cg, "freezer");
  if (!p_cg_contr_freezer) {
    std::cerr << app_mount_point << ": set FREEZER controller FAILED" << std::endl;
    return -1;
  }

  result = cgroup_set_value_string(p_cg_contr_freezer, "freezer.state", "THAWED");
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  // Create the kernel-space CGroup
  result = cgroup_create_cgroup(p_cg, 1);
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    cgroup_free(&p_cg);
    return -1;
  }
  
  cgroup_free(&p_cg);
  
  return 0;
}


int Initialize(pid_t appID) {
  int result = 0;
  std::string cg_mount_point = "";
  struct cgroup *p_cg;
  struct cgroup_controller *p_cg_contr_cpuset;
  struct cgroup_controller *p_cg_contr_cpu;
  struct cgroup_controller *p_cg_contr_freezer;

  // Creating a sample CGroup with the app PID
  std::string cg_name = std::to_string(appID);
  std::string app_mount_point = MAINGROUP + cg_name;

  p_cg = cgroup_new_cgroup(app_mount_point.c_str());
  if (!p_cg) {
    std::cerr << "Error: cannot create " << app_mount_point << std::endl;
    return -1;
  }

  // Set CPUSET configuration (if available)
  if (!mounts[CPUSET]) {
    std::cerr << app_mount_point << ": CPUSET controller not mounted" << std::endl;
    return -1;
  }

  p_cg_contr_cpuset = cgroup_add_controller(p_cg, "cpuset");
  if (!p_cg_contr_cpuset) {
    std::cerr << app_mount_point << ": set CPUSET controller FAILED" << std::endl;
    return -1;
  }

  std::string cpus = std::string("0-") + std::to_string(NUM_OF_CORES - 1);
  result = cgroup_set_value_string(p_cg_contr_cpuset, "cpuset.cpus", cpus.c_str());
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  result = cgroup_set_value_string(p_cg_contr_cpuset, "cpuset.mems", "0");
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  // Set CPU configuration
  if (!mounts[CPU]) {
    std::cerr << app_mount_point << ": CPU controller not configured" << std::endl;
    return -1;
  }

  p_cg_contr_cpu = cgroup_add_controller(p_cg, "cpu");
  if (!p_cg_contr_cpu) {
    std::cerr << app_mount_point << ": set CPU controller FAILED" << std::endl;
    return -1;
  }

  result = cgroup_set_value_string(p_cg_contr_cpu, "cpu.cfs_period_us", std::to_string(cfs_period).c_str());
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  result = cgroup_set_value_string(p_cg_contr_cpu, "cpu.cfs_quota_us", "-1");
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  //Set freezer configuration
  if (!mounts[FREEZER]) {
    std::cerr << app_mount_point << ": FREEZER controller not mounted" << std::endl;
      return -1;
  }

  p_cg_contr_freezer = cgroup_add_controller(p_cg, "freezer");
  if (!p_cg_contr_freezer) {
    std::cerr << app_mount_point << ": set FREEZER controller FAILED" << std::endl;
    return -1;
  }

  result = cgroup_set_value_string(p_cg_contr_freezer, "freezer.state", "THAWED");
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  // Create the kernel-space CGroup
  result = cgroup_create_cgroup(p_cg, 1);
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }
  
  // Setting app into the CGroup
  result = cgroup_attach_task_pid (p_cg, appID);
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  result = cgroup_modify_cgroup(p_cg);
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  cgroup_free(&p_cg);
  return appID;
}

int Remove(pid_t appID) {
  int result = 0;
  struct cgroup *p_cg;
  
  // Creating a sample CGroup with the app PID
  std::string cg_name = std::to_string(appID);
  std::string app_mount_point = MAINGROUP + cg_name;

  p_cg = cgroup_new_cgroup(app_mount_point.c_str());
  if (!p_cg) {
    std::cerr << "Error: cannot retrieve " << app_mount_point << std::endl;
    return -1;
  }

  //it is necessary to add controllers to specify to delete them!
  for (int id : controller_ids) {
    struct cgroup_controller *p_cg_contr;
    p_cg_contr = cgroup_add_controller(p_cg, controller[id]);
    if (!p_cg_contr) {
      std::cerr << app_mount_point << ": set " << controller[id] << " controller FAILED" << std::endl;
      return -1;
    }
  }
 
  result = cgroup_delete_cgroup_ext(p_cg, CGFLAG_DELETE_RECURSIVE); //actually delete the cgroup with all controllers
  cgroup_free(&p_cg);

  return appID;
}

int Destroy() {
  int result = 0;
  struct cgroup *p_cg;
  
  // Creating a sample CGroup with the app PID
  std::string app_mount_point = MAINGROUP;

  p_cg = cgroup_new_cgroup(app_mount_point.c_str());
  if (!p_cg) {
    std::cerr << "Error: cannot retrieve " << app_mount_point << std::endl;
    return -1;
  }

  //it is necessary to add controllers to specify to delete them!
  for (int id : controller_ids) {
    struct cgroup_controller *p_cg_contr;
    p_cg_contr = cgroup_add_controller(p_cg, controller[id]);
    if (!p_cg_contr) {
      std::cerr << app_mount_point << ": set " << controller[id] << " controller FAILED" << std::endl;
      return -1;
    }
  }
 
  result = cgroup_delete_cgroup_ext(p_cg, CGFLAG_DELETE_RECURSIVE); //actually delete the cgroup with all controllers
  cgroup_free(&p_cg);

  return 0;
}

int UpdateCpuSet(pid_t appID, std::vector<int> cores) {
  int result = 0;
  struct cgroup *p_cg;
  struct cgroup_controller *p_cg_contr_cpuset;

  // Creating a sample CGroup with the app PID
  std::string cg_name = std::to_string(appID);
  std::string app_mount_point = MAINGROUP + cg_name;

  p_cg = cgroup_new_cgroup(app_mount_point.c_str());
  if (!p_cg) {
    std::cerr << "Error: cannot retrieve " << app_mount_point << std::endl;
    return -1;
  }

  p_cg_contr_cpuset = cgroup_add_controller(p_cg, "cpuset");
  if (!p_cg_contr_cpuset) {
    std::cerr << app_mount_point << ": retrieve CPUSET controller FAILED" << std::endl;
    return -1;
  }

  if (cores.size() == 0) {
    std::cout << "Error: empty cpuset from policy" << std::endl;
    return -1;
  }

  std::string cpuset = std::to_string(cores[0]);

  for (int id = 1; id < cores.size(); id++) {
    cpuset += "," + std::to_string(cores[id]);
  }

  result = cgroup_set_value_string(p_cg_contr_cpuset, "cpuset.cpus", cpuset.c_str());
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  result = cgroup_modify_cgroup(p_cg);
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
  }

  cgroup_free(&p_cg);

  return appID;
}

// @param quota: X CPUS => X,0 (e.g 1 CPU and a half => 1,5)
int UpdateCpuQuota(pid_t appID, float quota) {

  int result = 0;
  struct cgroup *p_cg;
  struct cgroup_controller *p_cg_contr_cpu;

  // Creating a sample CGroup with the app PID
  std::string cg_name = std::to_string(appID);
  std::string app_mount_point = MAINGROUP + cg_name;

  p_cg = cgroup_new_cgroup(app_mount_point.c_str());
  if (!p_cg) {
    std::cerr << "Error: cannot retrieve " << app_mount_point << std::endl;
    return -1;
  }

  p_cg_contr_cpu = cgroup_add_controller(p_cg, "cpu");
  if (!p_cg_contr_cpu) {
    std::cerr << app_mount_point << ": set CPU controller FAILED" << std::endl;
    return -1;
  }

  result = cgroup_set_value_string(p_cg_contr_cpu, "cpu.cfs_period_us", std::to_string(cfs_period).c_str());
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  result = cgroup_set_value_string(p_cg_contr_cpu, "cpu.cfs_quota_us",
    std::to_string((int)(quota * cfs_period)).c_str());
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  result = cgroup_modify_cgroup(p_cg);
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
  }

  cgroup_free(&p_cg);
  return appID;
}

int UpdateCpuFreezer(pid_t appID, bool freeze) {
  int result = 0;
  struct cgroup *p_cg;
  struct cgroup_controller *p_cg_contr_freezer;

  // Creating a sample CGroup with the app PID
  std::string cg_name = std::to_string(appID);
  std::string app_mount_point = MAINGROUP + cg_name;

  p_cg = cgroup_new_cgroup(app_mount_point.c_str());
  if (!p_cg) {
    std::cerr << "Error: cannot retrieve " << app_mount_point << std::endl;
    return -1;
  }

  p_cg_contr_freezer = cgroup_add_controller(p_cg, "freezer");
  if (!p_cg_contr_freezer) {
    std::cerr << app_mount_point << ": set CPU controller FAILED" << std::endl;
    return -1;
  }

  if(freeze)
    result = cgroup_set_value_string(p_cg_contr_freezer, "freezer.state", "FROZEN");
  else
    result = cgroup_set_value_string(p_cg_contr_freezer, "freezer.state", "THAWED");
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
    return -1;
  }

  result = cgroup_modify_cgroup(p_cg);
  if (result) {
    std::cerr << "Error: " << cgroup_strerror(result) << std::endl;
  }

  cgroup_free(&p_cg);
  return appID;
}

}; // end CGUtils namespace