#include <dvs_msgs/EventArray.h>

// Allocate / initialize GPU memory (called once or on switch)
void init_gpu_memory(
    const dvs_msgs::EventArray::ConstPtr& msg);

// Free GPU memory
void free_gpu_memory();

// GPU event processing entry point
void tracker_gpu_events(
    const dvs_msgs::EventArray::ConstPtr& msg);
