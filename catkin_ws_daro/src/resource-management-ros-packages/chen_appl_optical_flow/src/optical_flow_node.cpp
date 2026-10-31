/**
 * optical_flow_node.cpp
 * This code corresponds to the ROS node which goal is to compute the optical flow from the
 * distance surface images.
 * This is the fourth node of the pipeline architecture.
 */

#include "optical_flow/defines.hpp"
#include "optical_flow/includes.hpp"

#include "optical_flow/optical_flow_gpu.hpp"
#include "optical_flow/optical_flow_cpu.hpp"

#include "../../chen_resource_controller/controller/ApplicationMonitor.h"
#include "../../chen_resource_controller/controller/MappingConsts.h"

typedef std::chrono::high_resolution_clock Clock;

#if FPS_MEASUREMENT
#include <fstream>
ofstream myfile;
bool gpu_mem_initialized = true;
#endif

/**
 * Global variables definition
 */

// ROS publisher, used to send the distance surface image once computed
ros::Publisher optical_flow_pub;

// Number of pyramidal levels for the optical flow library
int pyramidal_flow_levels;

// Number of pyramidal levels for the optical flow library
float max_flow;

// Regularization parameters for each layer of the pyramidal optical flow computation module
vector<float> gammas;

// Number of smoothing iterations for each layer of the pyramidal optical flow computation module
vector<int> smooth_iterations;

data_t *data = nullptr;
double req_thr = 25;
int mapping = MAP_LITTLE_ID;

// Runtime switching state
std::mutex switch_mutex;
bool gpu_initialized = false;
bool last_use_gpu = false;
cv::Size last_gpu_image_size;

/* -------------------------------------------------------------------------- */
/*                         CPU / GPU SAFE SWITCHING                            */
/* -------------------------------------------------------------------------- */
inline void updateImplementation(const cv::Mat& dist_surface)
{
  std::lock_guard<std::mutex> lock(switch_mutex);

  if (!data) return;

  const bool want_gpu = data->useGPU;

  /* Switch implementation */
  if (want_gpu != last_use_gpu) {
    if (want_gpu) {
      ROS_INFO("Switching optical flow to GPU");
      init_gpu_memory(dist_surface, pyramidal_flow_levels);
      last_gpu_image_size = dist_surface.size();
      gpu_initialized = true;
    } else {
      ROS_INFO("Switching optical flow to CPU");
      if (gpu_initialized) {
        free_gpu_memory();
        gpu_initialized = false;
      }
    }
    last_use_gpu = want_gpu;
    return;
  }

  /* Same backend → only reallocate if image size changed */
  if (want_gpu && gpu_initialized) {
    if (dist_surface.size() != last_gpu_image_size) {
      init_gpu_memory(dist_surface, pyramidal_flow_levels);
      last_gpu_image_size = dist_surface.size();
    }
  }
}

/**
 * \brief ROS callback, called when a new distance surface image arrives.
 * From this image, its optical flow is computed using the flowfilter library, and then sent for
 * further operations with it.
 */
void callback_dist_surface(const sensor_msgs::Image::ConstPtr& img)
{
// #if FPS_MEASUREMENT
//   ros::Time begin = ros::Time::now();
// #endif
  auto t_init = Clock::now();
  // Converting the message to an usable OpenCV Mat
  Mat dist_surface;
  try {
    dist_surface = cv_bridge::toCvShare(img)->image;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
    return;
  }

  // Runtime CPU / GPU switch
  updateImplementation(dist_surface);
  // Computing the optical flow from the distance surface image, using the flowfilter GPU library
  Mat optical_flow;
  if (data->useGPU) {
    optical_flow = optical_flow_gpu(
      dist_surface, 
      pyramidal_flow_levels, 
      max_flow, 
      gammas, 
      smooth_iterations);
  } else {
    optical_flow = optical_flow_cpu(
      dist_surface, 
      pyramidal_flow_levels, 
      max_flow, 
      gammas, 
      smooth_iterations);
  }

  if (optical_flow.empty()) return;

  // Converting the optical_flow Mat to a ROS compatible msg and publishing it
  cv_bridge::CvImage optical_flow_msg;
  optical_flow_msg.encoding = "32FC2";
  optical_flow_msg.image = optical_flow;
  optical_flow_pub.publish(optical_flow_msg.toImageMsg());

// #if FPS_MEASUREMENT
//   if(gpu_mem_initialized) {
//     ros::Duration duration = ros::Time::now() - begin;
//     myfile << duration.toSec()*1000 << endl;
//   } else {
//     gpu_mem_initialized = true;
//   }
// #endif

  auto t_end = Clock::now();
  double elapsed_ms =
  std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_init).count();

  std::cout << "[Optical Flow] processing time: " << elapsed_ms << " ms" << std::endl;

  monitorTick(data);
  setUnitThProcessingTime(data, 1.0e-3*elapsed_ms);
}

/**
 * \brief Main function.
 */
int main(int argc, char* argv[])
{
  // ROS node initialization
  ros::init(argc, argv, "optical_flow_node");
  ros::NodeHandle n("~");

  n.param<double>("req_thr", req_thr, REQ_THR);
  n.param<int>("mapping", mapping, MAPPING);
  if(req_thr <= 0 || mapping < 0) {
    ROS_ERROR("No req_thr/mapping given, exiting!");
    exit(EXIT_FAILURE);
  }

  data = monitorAttach("optical_flow", req_thr, mapping, 1, true);

  // Collection of the number of pyramidal flow levels
  n.param<int>("pyramidal_flow_levels", pyramidal_flow_levels, PYRAMIDAL_FLOW_LEVELS_DEFAULT);

  // Collection of the maximum optical flow magnitude
  n.param<float>("max_flow", max_flow, MAX_FLOW_DEFAULT);

  // Collection of the gamma values for the optical flow config
  n.param("gammas", gammas, GAMMAS_DEFAULT);

  // Collection of the smooth iterations for the optical flow config
  n.param("smooth_iterations", smooth_iterations, SMOOTH_ITERATIONS_DEFAULT);

  // Collection of the subscriber/publisher queues size
  int sub_pub_queues_size;
  n.param<int>("queues_size", sub_pub_queues_size, QUEUES_SIZE_DEFAULT);

  last_use_gpu = data->useGPU;

  // Configuration of the distance surface subscriber and the optical flow publisher
  ros::Subscriber sub_dist_surface = n.subscribe(
    "/edge_image", sub_pub_queues_size, callback_dist_surface);
  optical_flow_pub = n.advertise<sensor_msgs::Image>("/optical_flow", sub_pub_queues_size);

#if FPS_MEASUREMENT
  myfile.open(string(FPS_MEASUREMENT_FOLDER_PATH) + "/of.txt");
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
