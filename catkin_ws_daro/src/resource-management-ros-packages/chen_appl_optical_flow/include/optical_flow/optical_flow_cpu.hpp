/**
 * optical_flow_cpu.hpp
 * Header file for the computation of the optical flow from the distance surface images, on CPU.
 */

#pragma once

#include "optical_flow/includes.hpp"


/**
 * \brief Computes optical flow from the given distance surface image, using a CPU-based optical
 * flow algorithm, and applies the adequate mask to only return flow results for the pixels that
 * have received at least an event.
 *
 * \param dist_surface The distance surface image
 * \param pyramidal_levels The number of pyramidal levels the optical flow algorithm should be
 * configured with
 * \param max_flow The maximum optical flow (in pixels) the optical flow algorithm should be
 * configured with
 * \param gamma The regularization weights for each pyramidal level (kept for interface consistency)
 * \param smooth_iterations The regularization weights for each pyramidal level of the optical flow
 * algorithm
 *
 * \return The optical flow results as an OpenCV matrix, where only the edge pixels have a value
 */
Mat optical_flow_cpu(
  Mat& dist_surface, int pyramidal_levels, float max_flow, const vector<float>& gamma,
  const vector<int>& smooth_iterations);
