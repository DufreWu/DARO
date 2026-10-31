/**
 * optical_flow_node.cpp
 * This code corresponds to the ROS node which goal is to compute the optical flow from the
 * distance surface images.
 * This is the fourth node of the pipeline architecture.
 */

#include "rt_of_low_high_res_event_cameras/defines.hpp"
#include "rt_of_low_high_res_event_cameras/includes.hpp"

#include "rt_of_low_high_res_event_cameras/optical_flow_gpu.hpp"

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

data_t *data;
double req_thr = 25;
int mapping = 1;

std::chrono::seconds::rep clock_to_s(std::chrono::time_point<std::chrono::system_clock> tp) {
    return std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch()).count();
}

std::chrono::nanoseconds::rep clock_to_ns(std::chrono::time_point<std::chrono::system_clock> tp) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
}

std::string clock_to_s_str(std::chrono::time_point<std::chrono::system_clock> tp) {
    std::time_t curr_time_c = std::chrono::system_clock::to_time_t(tp);
    std::string curr_time_s = std::string(std::ctime(&curr_time_c));
    curr_time_s = curr_time_s.substr(0, curr_time_s.size()-1);
    return curr_time_s;
}

/**
 * \brief ROS callback, called when a new distance surface image arrives.
 * From this image, its optical flow is computed using the flowfilter library, and then sent for
 * further operations with it.
 */
void callback_dist_surface(const sensor_msgs::Image::ConstPtr& img)
{
  ROS_INFO("Callback dist with %lu events", img->data.size());
  auto t_init = Clock::now();
#if FPS_MEASUREMENT
  ros::Time begin = ros::Time::now();
#endif

  // Converting the message to an usable OpenCV Mat
  Mat dist_surface;
  try {
    dist_surface = cv_bridge::toCvShare(img)->image;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
    return;
  }

  // Computing the optical flow from the distance surface image, using the flowfilter GPU library
  Mat optical_flow = optical_flow_gpu(
    dist_surface, pyramidal_flow_levels, max_flow, gammas, smooth_iterations);

  // Converting the optical_flow Mat to a ROS compatible msg and publishing it
  cv_bridge::CvImage optical_flow_msg;
  optical_flow_msg.encoding = "32FC2";
  optical_flow_msg.image = optical_flow;
  optical_flow_pub.publish(optical_flow_msg.toImageMsg());

#if FPS_MEASUREMENT
  if(gpu_mem_initialized) {
    ros::Duration duration = ros::Time::now() - begin;
    myfile << duration.toSec()*1000 << endl;
  } else {
    gpu_mem_initialized = true;
  }
#endif
  
  auto t_init_ns = clock_to_ns(t_init);
  auto t_init_s = clock_to_s(t_init);
  auto t_init_s_str = clock_to_s_str(t_init);
  
  auto t_end = Clock::now();
  auto t_end_ns = clock_to_ns(t_end);

  float processing_time = (1.0e-9*(t_end_ns - t_init_ns));
  monitorTick(data);
  setUnitThProcessingTime(data, processing_time);
}


/**
 * \brief Main function.
 */
int main(int argc, char* argv[])
{
  // ROS node initialization
  ros::init(argc, argv, "optical_flow_node");
  ros::NodeHandle n("~");
  string ns = "/rt_of_low_high_res_event_cameras";

  n.param<double>("req_thr", req_thr, REQ_THR);
  n.param<int>("mapping", mapping, MAPPING);
  if(req_thr <= 0 || mapping < 0) {
    ROS_ERROR("No req_thr/mapping given, exiting!");
    exit(EXIT_FAILURE);
  }
  std::cout << "optical_flow_node, throughput " << req_thr << ", mapping " << mapping << std::endl;
  data = monitorAttach(argv[0], req_thr, mapping, 1, true);

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

  // Configuration of the distance surface subscriber and the optical flow publisher
  ros::Subscriber sub_dist_surface = n.subscribe(
    ns+"/distance_surface", sub_pub_queues_size, callback_dist_surface);
  optical_flow_pub = n.advertise<sensor_msgs::Image>(ns+"/optical_flow", sub_pub_queues_size);

#if FPS_MEASUREMENT
  myfile.open(string(FPS_MEASUREMENT_FOLDER_PATH) + "/of.txt");
#endif

  // Call to ros::spin() to process incomming events until Ctrl+C is pressed
  ros::spin();

#if FPS_MEASUREMENT
  myfile.close();
#endif

  return 0;
}
