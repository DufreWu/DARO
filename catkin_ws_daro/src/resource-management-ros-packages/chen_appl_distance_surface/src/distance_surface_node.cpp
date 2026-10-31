/**
 * distance_surface_node.cpp
 * This code corresponds to the ROS node which goal is to compute the distance surface image from
 * the edge images.
 * This is the third node of the pipeline architecture.
 */

#include "distance_surface/defines.hpp"
#include "distance_surface/includes.hpp"

#include "distance_surface/distance_surface_cpu.hpp"
#include "distance_surface/distance_surface_gpu.hpp"

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

// The function that will be used for computing the distance surfaces (CPU/GPU)
function<Mat(Mat, string)> dist_surface_function;

// String, containing the distance surface formulation that should be used
string dist_surface_formulation;

// ROS publisher, used to send the distance surface image once computed
ros::Publisher dist_surface_pub;

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
inline void updateImplementation(const cv::Mat& edges)
{
  std::lock_guard<std::mutex> lock(switch_mutex);

  if (!data) return;

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
    dist_surface_function = distance_surface_gpu;
  }
  else {
    ROS_INFO("Switching to CPU version at runtime");

    if (gpu_initialized) {
      free_gpu_memory();
      gpu_initialized = false;
    }
    dist_surface_function = distance_surface_cpu;
  }

  last_use_gpu = data->useGPU;
}

/**
 * \brief ROS callback, called when a new edge image message arrives.
 * From this image, its distance transform is computed, using the wanted formulation, and then sent
 * for further operations with it.
 */
void callback_edges_image(const sensor_msgs::Image::ConstPtr& img)
{
  ROS_INFO("Callback dist with %lu events", img->data.size());
  auto t_init = Clock::now();
#if FPS_MEASUREMENT
  ros::Time begin = ros::Time::now();
#endif

  // Converting the message to an usable OpenCV Mat
  Mat edges;
  try {
    edges = cv_bridge::toCvShare(img)->image;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
    return;
  }

  updateImplementation(edges);

  // Computing the distance surface, using the correct CPU or GPU function
  const Mat dist_surface = dist_surface_function(edges, dist_surface_formulation);

  // Before sending, if the edge image contained no edge pixel, the returned Mat is empty. So, if
  // we detect this is the case here, we just discard it.
  if(dist_surface.empty()) {
    return;
  }

  // Converting the OpenCV Mat to a ROS compatible msg and publishing it
  cv_bridge::CvImage dist_surface_msg;
  dist_surface_msg.encoding = "mono8";
  dist_surface_msg.image = dist_surface;
  dist_surface_pub.publish(dist_surface_msg.toImageMsg());

#if FPS_MEASUREMENT
  if(gpu_mem_initialized) {
    ros::Duration duration = ros::Time::now() - begin;
    myfile << duration.toSec()*1000 << endl;
  } else {
    gpu_mem_initialized = true;
  }
#endif
  // 5. Monitoring
  auto t_end = Clock::now();
  double processing_time =
      std::chrono::duration<double>(t_end - t_init).count();

  monitorTick(data);
  setUnitThProcessingTime(data, processing_time);
}


/**
 * \brief Main function.
 */
int main(int argc, char* argv[])
{
  // ROS node initialization
  ros::init(argc, argv, "distance_surface_node");
  ros::NodeHandle n("~");

  n.param<double>("req_thr", req_thr, REQ_THR);
  n.param<int>("mapping", mapping, MAPPING);

  if(req_thr <= 0 || mapping < 0) {
    ROS_ERROR("No req_thr/mapping given, exiting!");
    exit(EXIT_FAILURE);
  }

  data = monitorAttach("distance_surface", req_thr, mapping, 1, true);

  // Collection of the distance surface formulation to use (linear / bound / logarithmic /
  // exponential)
  n.param<string>(
    "distance_surface_formulation", dist_surface_formulation,
    DISTANCE_SURFACE_FORMULATION_DEFAULT);

  // Collection of the subscriber/publisher queues size
  int sub_pub_queues_size;
  n.param<int>("queues_size", sub_pub_queues_size, QUEUES_SIZE_DEFAULT);

  if (data->useGPU) {
    dist_surface_function = distance_surface_gpu;
  } else {
    dist_surface_function = distance_surface_cpu;
  }
  last_use_gpu = data->useGPU;

  // Configuration of the denoised & filled edge image subscriber and the distance surface
  // publisher
  ros::Subscriber sub_edges = n.subscribe(
      "/edge_image", sub_pub_queues_size, callback_edges_image);  
  dist_surface_pub = n.advertise<sensor_msgs::Image>("/distance_surface", sub_pub_queues_size);

#if FPS_MEASUREMENT
  myfile.open(string(FPS_MEASUREMENT_FOLDER_PATH) + "/ds.txt");
#endif

  // Call to ros::spin() to process incomming events until Ctrl+C is pressed
  ros::spin();

  if (gpu_initialized) {
    free_gpu_memory();
  }

#if FPS_MEASUREMENT
  myfile.close();
#endif

  return 0;
}
