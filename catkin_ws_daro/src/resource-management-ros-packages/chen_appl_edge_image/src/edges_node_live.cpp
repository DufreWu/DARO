/**
 * edges_node_live.cpp
 * This code corresponds to the ROS node which goal is to receive the events from the camera in
 * real-time, and to compute the edge images from them.
 */

#include "edge_image/defines.hpp"
#include "edge_image/utils.hpp"
#include "edge_image/edges_cpu.hpp"
#include "edge_image/edges_gpu.hpp"

#include <dvs_msgs/EventArray.h>
#include <thread>
#include <mutex>
#include <chrono>

#include "../../chen_resource_controller/controller/ApplicationMonitor.h"
#include "../../chen_resource_controller/controller/MappingConsts.h"

typedef std::chrono::high_resolution_clock Clock;

/**
 * Global variables definition
 */
array<vector<dvs_msgs::Event>, 2> evt_buffers;
bool buffer_used = 0;
mutex evt_buf_mutex;

int height, width;
Mat undistortion_matrix;
ros::Publisher edges_pub;

// Resource management and runtime switching
data_t *data;
double req_thr = 25;
int mapping = 0; // 1 for CPU, 2 for GPU

// Throughput measurement
double current_throughput = 0.0;
long total_events_processed = 0;

/**
 * \brief Processing thread that switches between CPU and GPU at runtime.
 */
void accumulation_timing_thread(int accumulation_window)
{
  ros::NodeHandle nh_p("~");
  long events_in_window = 0;

  std::cout << "accumulation_timing_thread started: " << accumulation_window << "ms\n";

  while(ros::ok()) {
    // Check for parameter updates at runtime
    nh_p.getParam("mapping", mapping);

    {
      lock_guard<mutex> lock(evt_buf_mutex);
      buffer_used = !buffer_used;
    }

    size_t num_events = evt_buffers[!buffer_used].size();
    Mat edge_image;
    thread evts_processing_thread;

    // auto t_start_proc = Clock::now();

    // Runtime Switching Logic
    if (data->useGPU) {
      evts_processing_thread = thread(
        evts_to_edge_image_gpu, std::ref(evt_buffers[!buffer_used]), &edge_image, height, width,
        std::ref(undistortion_matrix));
    } else {
      evts_processing_thread = thread(
        evts_to_edge_image_cpu, std::ref(evt_buffers[!buffer_used]), &edge_image, height, width,
        std::ref(undistortion_matrix));
    }

    // Sleep for the remainder of the accumulation window
    this_thread::sleep_for(chrono::milliseconds(accumulation_window));
    
    if(evts_processing_thread.joinable()) {
        evts_processing_thread.join();
    }

    // Clean processed buffer
    evt_buffers[!buffer_used].clear();

    // Publish results
    if(!edge_image.empty()) {
      cv_bridge::CvImage edges_msg;
      edges_msg.header.stamp = ros::Time::now();
      edges_msg.encoding = "mono8";
      edges_msg.image = edge_image;
      edges_pub.publish(edges_msg.toImageMsg());
    }

    // auto t_end_proc = Clock::now();
    
    // // Throughput Calculation (Events per second)
    // events_in_window = num_events;
    // double elapsed_sec = chrono::duration<double>(t_end_proc - t_start_proc).count();
    // if (elapsed_sec > 0) {
    //     current_throughput = events_in_window / elapsed_sec;
    // }

    // // Update resource monitor
    // monitorTick(data);
    // setUnitThProcessingTime(data, elapsed_sec);
  }
}

/**
 * \brief ROS callback for incoming events
 */
void EventMsgCallback(const dvs_msgs::EventArray::ConstPtr& msg)
{
  auto t_init = Clock::now();
  lock_guard<mutex> lock(evt_buf_mutex);
  evt_buffers[buffer_used].insert(evt_buffers[buffer_used].end(), begin(msg->events), end(msg->events));

  auto t_end = Clock::now();
  double elapsed_ms =
  std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_init).count();

  std::cout << "[Edge Image] processing time: " << elapsed_ms << " ms" << std::endl;

  // Update resource monitor
  monitorTick(data);
  setUnitThProcessingTime(data, 1.0e-3*elapsed_ms);
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "edges_node");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  // Parameters
  nh_private.param<double>("req_thr", req_thr, REQ_THR);
  nh_private.param<int>("mapping", mapping, MAPPING);
  
  int accumulation_window = -1;
  nh_private.getParam("accumulation_window", accumulation_window);
  
  nh_private.param<int>("height", height, -1);
  nh_private.param<int>("width", width, -1);
  
  int queues_size;
  nh_private.param<int>("queues_size", queues_size, QUEUES_SIZE_DEFAULT);

  if(height <= 0 || width <= 0 || accumulation_window <= 0) {
    ROS_ERROR("Missing mandatory parameters (height, width, window). Exiting.");
    return -1;
  }

  // Monitor initialization
  data = monitorAttach("edge_image", req_thr, mapping, 1, true);

  // Subscribers and Publishers
  ros::Subscriber evts_sub = nh.subscribe("/dvs/events", queues_size, EventMsgCallback);
  edges_pub = nh.advertise<sensor_msgs::Image>("/edge_image", queues_size);

  // Start timing thread
  thread acc_timing_thread(accumulation_timing_thread, accumulation_window);

  ros::spin();

  if(acc_timing_thread.joinable()) {
      acc_timing_thread.join();
  }

  return 0;
}