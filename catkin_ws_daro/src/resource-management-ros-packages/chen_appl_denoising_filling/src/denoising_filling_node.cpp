/**
 * denoising_filling_node.cpp
 */

#include "denoising_filling/defines.hpp"
#include "denoising_filling/includes.hpp"
#include "denoising_filling/denoising_filling_cpu.hpp"
#include "denoising_filling/denoising_filling_gpu.hpp"

#include "../../chen_resource_controller/controller/ApplicationMonitor.h"
#include "../../chen_resource_controller/controller/MappingConsts.h"

typedef std::chrono::high_resolution_clock Clock;

#if FPS_MEASUREMENT
#include <fstream>
ofstream myfile;
bool gpu_mem_initialized = false;
#endif

/**
 * Global variables definition
 */
function<Mat(Mat, int, int)> denoising_filling_function;
int denoising_min_neighbours, filling_min_neighbours;
ros::Publisher df_edges_pub;

data_t *data = nullptr;
double req_thr = 25;
int mapping = MAP_LITTLE_ID;

// Runtime state
std::mutex switch_mutex;
bool gpu_initialized = false;
bool last_use_gpu = false;

// Throughput and performance tracking
double current_fps = 0.0;

/**
 * Switch CPU / GPU safely (only on change)
 */
inline void updateImplementation(const cv::Mat& edges)
{
  std::lock_guard<std::mutex> lock(switch_mutex);

  if (!data) return;

  // No change → do nothing
  if (data->useGPU == last_use_gpu) {
    if (data->useGPU) {
      init_gpu_memory(edges);
    }
    return;
  }

  if (data->useGPU) {
    ROS_INFO("Switching to GPU version at runtime");

    init_gpu_memory(edges);
    gpu_initialized = true;

    denoising_filling_function = denoising_filling_gpu;
  }
  else {
    ROS_INFO("Switching to CPU version at runtime");

    if (gpu_initialized) {
      free_gpu_memory();
      gpu_initialized = false;
    }

    denoising_filling_function = denoising_filling_cpu;
  }

  last_use_gpu = data->useGPU;
}

/**
 * ROS callback
 */
void callback_edges(const sensor_msgs::Image::ConstPtr& img)
{
  auto t_init = Clock::now();

  // 1. Image conversion
  cv::Mat edges;
  try {
    edges = cv_bridge::toCvShare(img)->image;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
    return;
  }

  // 2. Runtime switch (monitor-driven)
  updateImplementation(edges);

  // 3. Processing
  cv::Mat df_edges =
      denoising_filling_function(edges,
                                 denoising_min_neighbours,
                                 filling_min_neighbours);

  if (cv::sum(df_edges)[0] == 0) return;

  // 4. Publishing
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = img->header.stamp;
  out_msg.encoding = "mono8";
  out_msg.image = df_edges;
  df_edges_pub.publish(out_msg.toImageMsg());

  // 5. Monitoring
  auto t_end = Clock::now();
  double processing_time =
      std::chrono::duration<double>(t_end - t_init).count();

  monitorTick(data);
  setUnitThProcessingTime(data, processing_time);

  if (processing_time > 0.0) {
    current_fps = 1.0 / processing_time;
  }
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "denoising_filling_node");
  ros::NodeHandle n("~");

  n.param<double>("req_thr", req_thr, REQ_THR);
  n.param<int>("mapping", mapping, MAPPING);
  
  if(req_thr <= 0 || mapping < 0) {
    ROS_ERROR("No req_thr/mapping given, exiting!");
    exit(EXIT_FAILURE);
  }

  data = monitorAttach("denoising_filling", req_thr, mapping, 1, true);

  // Initial implementation
  if (data->useGPU) {
    denoising_filling_function = denoising_filling_gpu;
  } else {
    denoising_filling_function = denoising_filling_cpu;
  }
  last_use_gpu = data->useGPU;

  n.param<int>("denoising_min_neighbours", denoising_min_neighbours, DENOISING_MIN_NEIGHBOURS_DEFAULT);
  n.param<int>("filling_min_neighbours", filling_min_neighbours, FILLING_MIN_NEIGHBOURS_DEFAULT);

  int sub_pub_queues_size;
  n.param<int>("queues_size", sub_pub_queues_size, QUEUES_SIZE_DEFAULT);

  ros::Subscriber sub_edges = n.subscribe("/edge_image", sub_pub_queues_size, callback_edges);
  df_edges_pub = n.advertise<sensor_msgs::Image>("/denoised_filled_edge_image", sub_pub_queues_size);

  ros::spin();

  // Cleanup on shutdown
  if (gpu_initialized) {
    free_gpu_memory();
  }

  return 0;
}