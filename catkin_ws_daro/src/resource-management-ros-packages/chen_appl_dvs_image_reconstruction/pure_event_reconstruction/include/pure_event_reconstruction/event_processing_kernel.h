#ifndef EVENT_PROCESSING_KERNEL_H
#define EVENT_PROCESSING_KERNEL_H

__global__ void processEventsKernel(double* d_ts_array, double* d_ts_array_on, double* d_ts_array_off,
                                    double* d_log_intensity_state, double* d_leaky_event_count_on,
                                    double* d_leaky_event_count_off, int* d_x_coords, int* d_y_coords,
                                    double* d_timestamps, uint8_t* d_polarities, int num_events, int width,
                                    int height, double cutoff_frequency_global, double cutoff_frequency_per_event,
                                    double event_count_cutoff_frequency, double contrast_threshold_on,
                                    double contrast_threshold_off);

#endif // EVENT_PROCESSING_KERNEL_H