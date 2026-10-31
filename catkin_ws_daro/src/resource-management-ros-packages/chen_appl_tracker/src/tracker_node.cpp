#include <ros/ros.h>
#include <chrono>
#include <functional>
#include <mutex>

#include <glog/logging.h>

#include "tracker_cpu.h"
#include "tracker_gpu.h"

#include "../../chen_resource_controller/controller/ApplicationMonitor.h"
#include "../../chen_resource_controller/controller/MappingConsts.h"


// feature detection
DEFINE_int32(max_corners, 100, \
           "Maximum features allowed to be tracked.");
DEFINE_int32(min_corners, 60, \
           "Minimum features allowed to be tracked.");
DEFINE_int32(min_distance, 30, \
           "Minimum distance between detected features. Parameter passed to goodFeatureToTrack.");
DEFINE_int32(block_size, 30, \
           "Block size to compute Harris score. passed to harrisCorner and goodFeaturesToTrack.");

DEFINE_double(k, 0.04, \
              "Magic number for Harris score.");
DEFINE_double(quality_level, 0.3, \
              "Determines range of harris score allowed between the maximum and minimum. Passed to goodFeaturesToTrack.");
DEFINE_double(log_eps, 1e-2, \
              "Small constant to compute log image. To avoid numerical issues normally we compute log(img /255 + log_eps).");
DEFINE_double(first_image_t, -1, \
              "If specified discards all images until this time.");
DEFINE_string(tracks_file_txt, "", \
              "If specified, writes feature tracks to file with format id t x y.");

// tracker
DEFINE_int32(lk_window_size, 15, \
           "Parameter for KLT. Used for bootstrapping feature.");
DEFINE_int32(num_pyramidal_layers, 2, \
           "Parameter for KLT. Used for bootstrapping feature.");
DEFINE_int32(batch_size, 200, \
           "Determines the size of the event buffer for each patch. If a new event falls into a patch and the buffer is full, the older event in popped.");
DEFINE_int32(patch_size, 25, \
           "Determines size of patch around corner. All events that fall in this patch are placed into the features buffer.");
DEFINE_int32(max_num_iterations, 10, \
           "Maximum number of itrations allowed by the ceres solver to update optical flow direction and warp.");

DEFINE_double(displacement_px, 0.6, \
              "Controls scaling parameter for batch size calculation: from formula optimal batchsize == 1/Cth * sum |nabla I * v/|v||. displacement_px corresponds to factor 1/Cth");
DEFINE_double(tracking_quality, 0.4, \
              "minimum tracking quality allowed for a feature to be tracked. Can be a number between 0 (bad track) and 1 (good track). Is a rescaled number computed from ECC cost via tracking_quality == 1 - ecc_cost / 4. Note that ceres returns ecc_cost / 2.");
DEFINE_string(bootstrap, "",
              "Method for bootstrapping optical flow direction. Options are 'klt', 'events'");

typedef std::chrono::high_resolution_clock Clock;
using TrackerFn =
    std::function<void(const dvs_msgs::EventArray::ConstPtr&)>;
static TrackerFn tracker_function;   // MUST always be initialized

data_t *data = nullptr;
double req_thr = 25;
int mapping = MAP_LITTLE_ID;

// Runtime state
std::mutex switch_mutex;
bool gpu_initialized = false;
bool last_use_gpu = false;

/**
 * Switch implementation safely
 */
inline void updateImplementation(const dvs_msgs::EventArray::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(switch_mutex);

  if (!data) return;

  if (data->useGPU == last_use_gpu) {
    if (data->useGPU && !gpu_initialized) {
      init_gpu_memory(msg);
      gpu_initialized = true;
    }
    return;
  }

  if (data->useGPU) {
    ROS_INFO("Switching to GPU version at runtime");
    init_gpu_memory(msg);
    tracker_function = tracker_gpu_events;
    gpu_initialized = true;
  }
  else {
    ROS_INFO("Switching to CPU version at runtime");
    if (gpu_initialized) {
      free_gpu_memory();
      gpu_initialized = false;
    }
    tracker_function = tracker_cpu_events;
  }

  last_use_gpu = data->useGPU;
}

void eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg)
{
    auto t_init = Clock::now();

    updateImplementation(msg);
    // Defensive guard (should never trigger now)
    if (!tracker_function) {
        ROS_FATAL("tracker_function not initialized");
        return;
    }
    tracker_function(msg);

    auto t_end = Clock::now();
    double processing_time =
      std::chrono::duration<double>(t_end - t_init).count();

    monitorTick(data);
    setUnitThProcessingTime(data, processing_time);
}

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  ros::init(argc, argv, "tracker");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  nh_private.param<double>("req_thr", req_thr, 25);
  nh_private.param<int>("mapping", mapping, 0);
  if(req_thr <= 0 || mapping < 0) {
    ROS_ERROR("No req_thr/mapping given, exiting!");
    exit(EXIT_FAILURE);
  }
  std::cout << "tracker, throughput " << req_thr << ", mapping " << mapping << std::endl;
  data = monitorAttach("tracker", req_thr, mapping, 1, true);

  if (data->useGPU) {
    tracker_function = tracker_gpu_events;
    last_use_gpu = true;
  }
  else {
    tracker_function = tracker_cpu_events;
    last_use_gpu = false;
  }

  ros::Subscriber event_sub_ = nh.subscribe("/dvs/events", 1, eventsCallback);

  ros::spin();

  if (gpu_initialized)
    free_gpu_memory();

  return 0;
}
