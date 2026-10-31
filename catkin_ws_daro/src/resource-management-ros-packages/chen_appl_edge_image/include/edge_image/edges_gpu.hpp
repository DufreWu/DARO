/**
 * edges_gpu.hpp
 * Header file for the events -> edge image computation, on GPU.
 */
#ifndef EDGES_GPU_HPP
#define EDGES_GPU_HPP

#pragma once

#include "edge_image/includes.hpp"
#include <dvs_msgs/Event.h>

/**
 * \brief Edge image creation function, which should be called at the end of the accumulation
 * window Δt. It consumes the events from the given buffer to create the edge image. Once done, the
 * created image is sent for further processing.
 *
 * \param evts_to_process Buffer of events to use for creating the edge image
 * \param edge_image_ptr Pointer to the output edge image OpenCV matrix
 * \param height Height of the edge image
 * \param width Width of the edge image
 * \param undistortion_matrix Matrix for undistorting the events (if needed)
 */

void evts_to_edge_image_gpu(
  const vector<dvs_msgs::Event>& evts_to_process, 
  Mat* edge_image_ptr, 
  int height, 
  int width,
  const Mat& undistortion_matrix);

/**
 * \brief Frees the memory allocated on GPU.
 * It must be called manually at the end of the execution!
 */
void free_gpu_memory();

#endif // EDGES_GPU_HPP
