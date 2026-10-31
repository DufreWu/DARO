#include <dvs_msgs/EventArray.h>
#include <sensor_msgs/Image.h>


/**
 * Initialize CPU tracker state.
 * Safe to call multiple times.
 */
void tracker_cpu_init();
void tracker_cpu_free();

/**
 * Process events (CPU).
 */
void tracker_cpu_events(const dvs_msgs::EventArray::ConstPtr& msg);
