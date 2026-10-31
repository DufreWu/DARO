#ifndef UTILS_CGROUPS_H_
#define UTILS_CGROUPS_H_

#include <algorithm>
#include <vector>

namespace CGUtils {

enum CGControllers {
	CPUSET = 0,
	CPU,
	FREEZER,
 	CGCONTROLLERS_COUNT
};

/// Checks if the root control group is mounted on the file system.
/// Initialize libCGroup internal structures.
/// Enables control groups support in Orchestrator on success.
/// It takes in input the number of cores (I don't know how to read it directly from the main cgroup configuration - otherwise it is not needed)
/// ALERT: To be called only once.
/// @return 0 on success; -1 on failure
int Setup();

/// Create a CGroup and attach the application appID to it
/// @return appID on success; -1 on failure.
int Initialize(pid_t appID);

/// Removes the CGroup associated to the application appID.
/// @return appID on success; -1 on failure.
int Remove(pid_t appID);

/// Removes the main CGroup.
/// @return 0 on success; -1 on failure.
int Destroy();

/// Writes the CPUset represented by the vector 'cores' into the CGroup
/// associated to the application appID.
/// @return appID on success; -1 on failure.
int UpdateCpuSet(pid_t appID, std::vector<int> cores);

/// Writes the CPU quota represented by `quota` into the CGroup associated to
/// the application appID.
/// @param quota: X CPUS => X,0 (e.g 1 CPU and a half => 1,5)
/// @return appID on success; -1 on failure.
int UpdateCpuQuota(pid_t appID, float quota);

int UpdateCpuFreezer(pid_t appID, bool freeze);
} // end namespace

#endif // UTILS_CGROUPS_H_
