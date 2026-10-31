/**
 * optical_flow_cpu.cpp
 * This code corresponds to the computation of the optical flow from the distance surface images,
 * on CPU.
 */

#include "optical_flow/optical_flow_cpu.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 * Global variables definition
 */

// Previous distance surface image
static Mat prev_dist_surface;

// Optical flow matrix
static Mat optical_flow;

// Optical flow matrix masked on event pixels
static Mat optical_flow_masked;

// Flag to check initialization
static bool initialized = false;


/**
 * \brief Computes optical flow from the given distance surface image on CPU using OpenCV,
 * and applies the adequate mask to only return flow results for the pixels that have received
 * at least an event.
 *
 * \param dist_surface The distance surface image
 * \param pyramidal_levels Number of pyramid levels
 * \param max_flow Maximum expected optical flow (pixels)
 * \param gamma Regularization weights (unused here, kept for API compatibility)
 * \param smooth_iterations Number of smoothing iterations (mapped to Farneback iterations)
 *
 * \return The optical flow results as an OpenCV matrix, where only the edge pixels have a value
 */
Mat optical_flow_cpu(
  Mat& dist_surface,
  int pyramidal_levels,
  float max_flow,
  const vector<float>& gamma,
  const vector<int>& smooth_iterations) 
{
  (void)gamma;
  (void)pyramidal_levels;
  // Initialize on first call
  if(!initialized) {
    dist_surface.convertTo(prev_dist_surface, CV_32F);
    optical_flow = Mat::zeros(dist_surface.rows, dist_surface.cols, CV_32FC2);
    optical_flow_masked = Mat::zeros(dist_surface.rows, dist_surface.cols, CV_32FC2);
    initialized = true;
    return optical_flow_masked;
  }

  // Convert current distance surface to float
  Mat curr_dist_surface;
  dist_surface.convertTo(curr_dist_surface, CV_32F);

  // Parameters mapping
  int iterations = smooth_iterations.empty() ? 3 : smooth_iterations[0];
  double pyr_scale = 0.5;
  int winsize = static_cast<int>(max_flow * 2.0f);
  winsize = std::max(winsize, 5);

  // Compute optical flow using Farneback
  calcOpticalFlowFarneback(
    prev_dist_surface,
    curr_dist_surface,
    optical_flow,
    pyr_scale,
    3,            // levels
    winsize = 9,
    iterations = 2,
    3,            // poly_n
    1.1,          // poly_sigma
    0
  );

  // Apply mask: only keep flow where an event occurred
  optical_flow_masked.setTo(Scalar(0, 0));
  optical_flow.copyTo(optical_flow_masked, dist_surface == 0);

  // Update previous frame
  curr_dist_surface.copyTo(prev_dist_surface);

  return optical_flow_masked;
}
