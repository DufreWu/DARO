/**
 * denoising_filling_node.cpp
 * This code corresponds to the ROS node which goal is to apply the denoising and filling step onto
 * the edge images.
 * This is the second node of the pipeline architecture.
 */

#include "rt_of_low_high_res_event_cameras/defines.hpp"
#include "rt_of_low_high_res_event_cameras/includes.hpp"

#include "rt_of_low_high_res_event_cameras/denoising_filling_cpu.hpp"
#include "rt_of_low_high_res_event_cameras/denoising_filling_gpu.hpp"

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

// The function that will be used for denoising and filling of the edge images
function<Mat(Mat, int, int)> denoising_filling_function;

// Thresholds for the denoising and filling steps (N_d and N_f in the article)
int denoising_min_neighbours, filling_min_neighbours;

// ROS publisher, used to send the denoised & filled edge image once computed
ros::Publisher df_edges_pub;

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
 * \brief ROS callback, called when a new edge image message arrives.
 * This image is denoised and filled, using the method described in the article, and then sent for
 * further operations with it.
 */
void callback_edges(const sensor_msgs::Image::ConstPtr& img)
{
  ROS_INFO("Callback edges with %lu events", img->data.size());
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

  // Computing the denoised & filled edge image, using the correct CPU or GPU function, and
  // checking that it is not empty (if this is the case, we simply discard it to avoid useless
  // computations down the pipeline)
  Mat df_edges = denoising_filling_function(
    edges, denoising_min_neighbours, filling_min_neighbours);
  if(sum(df_edges)[0] == 0) {
    return;
  }

  // Converting the OpenCV Mat to a ROS compatible msg and publishing it
  cv_bridge::CvImage denoised_filled_edges_msg;
  denoised_filled_edges_msg.encoding = "mono8";
  denoised_filled_edges_msg.image = df_edges;
  df_edges_pub.publish(denoised_filled_edges_msg.toImageMsg());

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
  ros::init(argc, argv, "denoising_filling_node");
  ros::NodeHandle n("~");
  string ns = "/rt_of_low_high_res_event_cameras";

  n.param<double>("req_thr", req_thr, REQ_THR);
  n.param<int>("mapping", mapping, MAPPING);
  if(req_thr <= 0 || mapping < 0) {
    ROS_ERROR("No req_thr/mapping given, exiting!");
    exit(EXIT_FAILURE);
  }
  std::cout << "denoising_filling_node, throughput " << req_thr << ", mapping " << mapping << std::endl;
  data = monitorAttach(argv[0], req_thr, mapping, 1, true);

  // Collection of the CPU/GPU parameter
  bool use_gpu_version;
  n.param<bool>("use_gpu_version", use_gpu_version, USE_GPU_VERSION_DEFAULT);
  std::cout << "denoising_filling_node, use_gpu_version " << use_gpu_version << std::endl;
  if(use_gpu_version) {
    denoising_filling_function = denoising_filling_gpu;
  } else {
    denoising_filling_function = denoising_filling_cpu;
  }

  // Collection of the denoising & filling settings (N_d and N_f in the paper).
  // Note that if a value < 1 or > 4 is provided, then it simply disables the corresponding
  // operation (no denoising and/or no filling)
  n.param<int>(
    "denoising_min_neighbours", denoising_min_neighbours, DENOISING_MIN_NEIGHBOURS_DEFAULT);
  n.param<int>(
    "filling_min_neighbours", filling_min_neighbours, FILLING_MIN_NEIGHBOURS_DEFAULT);

  // Collection of the subscriber/publisher queues size
  int sub_pub_queues_size;
  n.param<int>("queues_size", sub_pub_queues_size, QUEUES_SIZE_DEFAULT);

  // Configuration of the edge image subscriber and the denoised & filled edge image publisher
  ros::Subscriber sub_edges = n.subscribe(ns+"/edge_image", sub_pub_queues_size, callback_edges);
  df_edges_pub = n.advertise<sensor_msgs::Image>(
    ns+"/denoised_filled_edge_image", sub_pub_queues_size);

#if FPS_MEASUREMENT
  myfile.open(string(FPS_MEASUREMENT_FOLDER_PATH) + "/df.txt");
#endif

  // Call to ros::spin() to process incomming events until Ctrl+C is pressed
  ros::spin();

  // Freeing GPU memory before leaving if needed
  if(use_gpu_version) {
    free_gpu_memory();
  }

#if FPS_MEASUREMENT
  myfile.close();
#endif

  return 0;
}
