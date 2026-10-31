/**
 * edges_node_live.cpp
 * This code corresponds to the ROS node which goal is to receive the events from the camera in
 * real-time, and to compute the edge images from them.
 * This is the first node of the pipeline architecture.
 */

#include "rt_of_low_high_res_event_cameras/defines.hpp"
#include "rt_of_low_high_res_event_cameras/utils.hpp"

#include "rt_of_low_high_res_event_cameras/edges_cpu.hpp"

#include <dvs_msgs/EventArray.h>
#include <thread>
#include <mutex>

#include "../../chen_resource_controller/controller/ApplicationMonitor.h"
#include "../../chen_resource_controller/controller/MappingConsts.h"

typedef std::chrono::high_resolution_clock Clock;

/**
 * Global variables definition
 */

// The buffers of events, which function constantly alternates: while one is used for storing
// incomming events, the other one serves for creating the edge image from the events that were
// received
array<vector<dvs_msgs::Event>, 2> evt_buffers;

// A boolean to switch between which of the two event buffers should be used
bool buffer_used = 0;

// A mutex, used to lock the event buffers
mutex evt_buf_mutex;

// Height and width of the event sensor
int height, width;

// Undistortion matrix, used if a calibration file of the camera is given
Mat undistortion_matrix;

// ROS publisher, used to send the edge image once computed
ros::Publisher edges_pub;


/**
 * \brief Events processing function, which is just a global timer to indicate the end of the
 * accumulation window Δt, and to call the events -> edge image function accordingly.
 *
 * \param accumulation_window The time window Δt, during which events should be accumulated to form
 * the edge images (in ms)
 */
void accumulation_timing_thread(int accumulation_window)
{
  std::cout << "accumulation_timing_thread: " << accumulation_window << "\n";
  // This thread runs until we exit the node
  while(ros::ok()) {
    // Before processing events, we switch the current buffer used to store the received events.
    // This way, the newly received events will be stored in the other buffer safely.
    {
      lock_guard<mutex> lock(evt_buf_mutex);
      buffer_used = !buffer_used;
    }

    // We call the events to edge image function (in a new thread to respect the accumulation
    // window)
    Mat edge_image;
    thread evts_processing_thread(
      evts_to_edge_image, evt_buffers[!buffer_used], &edge_image, height, width,
      undistortion_matrix);

    // We sleep during the processing
    this_thread::sleep_for(chrono::milliseconds(accumulation_window));

    // Before looping and processing events again, we have to make sure the previous iteration is
    // finished (which should be the case almost all the time).
    evts_processing_thread.join();

    // We clean the queue of its (now processed) events
    evt_buffers[!buffer_used].clear();

    // And we publish the computed edge image (if not empty)
    if(!edge_image.empty()) {
      cv_bridge::CvImage edges_msg;
      edges_msg.encoding = "mono8";
      edges_msg.image = edge_image;
      edges_pub.publish(edges_msg.toImageMsg());
    }
  }
}

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
 * \brief ROS callback, called when a new DVS EventArray message arrives.
 * The events it contains are simply stored in the correct buffer, to be processed later on by the
 * events processing thread.
 *
 * \note The events from the message can be either DVS- or Prophesee-formatted: we don't care, the
 * messages have the same fields, so the same checksum, and ROS therefore considers them as the
 * same type of message
 */
void EventMsgCallback(const dvs_msgs::EventArray::ConstPtr& msg)
{
  auto t_init = Clock::now();
  // We first lock the event buffers, to avoid any collision
  lock_guard<mutex> lock(evt_buf_mutex);

  // We then append the received events at the back of the current buffer
  evt_buffers[buffer_used].insert(
  evt_buffers[buffer_used].end(), begin(msg->events), end(msg->events));

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
  ros::init(argc, argv, "edges_node");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  string ns = "/rt_of_low_high_res_event_cameras";

  nh_private.param<double>("req_thr", req_thr, REQ_THR);
  nh_private.param<int>("mapping", mapping, MAPPING);
  if(req_thr <= 0 || mapping < 0) {
    ROS_ERROR("No req_thr/mapping given, exiting!");
    exit(EXIT_FAILURE);
  }

  std::cout << "edges_node, throughput " << req_thr << ", mapping " << mapping << std::endl;
  data = monitorAttach(argv[0], req_thr, mapping, 1, true);

  // Collection of the input topic
  // string topic;
  // n.param<string>("topic", topic, "");
  // if(topic.empty()) {
  //   ROS_ERROR("No input topic selected, exiting!");
  //   exit(EXIT_FAILURE);
  // }

  // Collection of the duration of the accumulation window (Δt)
  int accumulation_window=-1;
  nh_private.getParam("accumulation_window", accumulation_window);
  if(accumulation_window <= 0) {
    ROS_ERROR("No accumulation window given, exiting!");
    exit(EXIT_FAILURE);
  }

  std::cout << "accumulation_window " << accumulation_window << std::endl;

  // Collection of the image height and width, passed as mandatory parameters
  nh_private.param<int>("height", height, -1);
  nh_private.param<int>("width", width, -1);
  if(height <= 0 || width <= 0) {
    ROS_ERROR("Invalid height/width parameters, exiting!");
    exit(EXIT_FAILURE);
  }

  // Collection of the subscriber/publisher queues size
  int queues_size;
  nh_private.param<int>("queues_size", queues_size, QUEUES_SIZE_DEFAULT);

  std::cout << "queues_size " << queues_size << std::endl;

  // // If given, collection of the calibration file, and preparation of the undistortion matrix
  // string calibration_file = n.param<string>("calibration_file", "");
  // if(!calibration_file.empty()) {
  //   undistortion_matrix = init_undistort_mat(calibration_file, height, width);
  // }

  // Configuration of the events subscriber and the edge image publisher
  // ros::Subscriber evts_sub = nh.subscribe("events", queues_size, &EventMsgCallback);
  ros::Subscriber evts_sub = nh.subscribe("/dvs/events", queues_size, &EventMsgCallback);
  edges_pub = nh.advertise<sensor_msgs::Image>(ns+"/edge_image", queues_size);

  // Launch of the events processing thread
  thread acc_timing_thread(accumulation_timing_thread, accumulation_window);

  // Call to ros::spin() to process incomming events until Ctrl+C is pressed
  ros::spin();

  // Before exiting, we make sure join the separate thread
  acc_timing_thread.join();

  return 0;
}
