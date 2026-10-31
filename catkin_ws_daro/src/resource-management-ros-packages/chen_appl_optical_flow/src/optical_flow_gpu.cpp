/**
 * optical_flow_gpu.cpp
 * GPU optical flow computation from distance surface images.
 */

#include "optical_flow/optical_flow_gpu.hpp"

#include <flowfilter/gpu/flowfilter.h>
#include <mutex>

/**
 * Global GPU state
 */

// Flowfilter module
static flowfilter::gpu::PyramidalFlowFilter filter;

// Image wrappers
static flowfilter::image_t dist_surface_wrapper = {};
static flowfilter::image_t optical_flow_wrapper = {};

// OpenCV matrices
static cv::Mat optical_flow;
static cv::Mat optical_flow_masked;

// Runtime state
static std::atomic<bool> gpu_initialized(false);
static int gpu_rows = -1;
static int gpu_cols = -1;
static int gpu_pyramid_levels = -1;

/* Cached parameters (avoid per-frame reconfiguration) */
static float last_max_flow = -1.0f;
static std::vector<float> last_gamma;
static std::vector<int> last_smooth_iterations;

/**
 * Wrap OpenCV Mat into flowfilter image_t
 */
static inline void wrap_cv_matrix(cv::Mat& cv_matrix, flowfilter::image_t& img)
{
  img.height   = cv_matrix.rows;
  img.width    = cv_matrix.cols;
  img.depth    = cv_matrix.channels();
  img.pitch    = cv_matrix.step;
  img.itemSize = cv_matrix.elemSize1();
  img.data     = cv_matrix.ptr();
}

/**
 * Initialize GPU memory (image-aware, idempotent)
 */
void init_gpu_memory(const cv::Mat& dist_surface, int pyramidal_levels)
{
  const int rows = dist_surface.rows;
  const int cols = dist_surface.cols;

  // Already initialized with correct size
  if (gpu_initialized && rows == gpu_rows && cols == gpu_cols && pyramidal_levels == gpu_pyramid_levels) {
    return;
  }

  // Reinitialize
  free_gpu_memory();

  // Allocate OpenCV buffers
  optical_flow.create(rows, cols, CV_32FC2);
  optical_flow_masked.create(rows, cols, CV_32FC2);

  // Wrap matrices
  wrap_cv_matrix(const_cast<cv::Mat&>(dist_surface), dist_surface_wrapper);
  wrap_cv_matrix(optical_flow, optical_flow_wrapper);

  // Create flowfilter
  filter = flowfilter::gpu::PyramidalFlowFilter(
    rows, 
    cols, 
    pyramidal_levels
  );

  gpu_rows = rows;
  gpu_cols = cols;
  gpu_pyramid_levels = pyramidal_levels;
  gpu_initialized = true;
}

/**
 * Free GPU memory (safe to call multiple times)
 */
void free_gpu_memory()
{
  if (!gpu_initialized) return;

  // Reset wrappers
  dist_surface_wrapper = {};
  optical_flow_wrapper = {};

  // Release OpenCV matrices
  optical_flow.release();
  optical_flow_masked.release();

  // Destroy filter
  filter = flowfilter::gpu::PyramidalFlowFilter();

  gpu_rows = -1;
  gpu_cols = -1;
  gpu_initialized = false;

  last_max_flow = -1.0f;
  last_gamma.clear();
  last_smooth_iterations.clear();
}

/**
 * GPU optical flow computation
 */
cv::Mat optical_flow_gpu(
  cv::Mat& dist_surface,
  int pyramidal_levels,
  float max_flow,
  const std::vector<float>& gamma,
  const std::vector<int>& smooth_iterations)
{
  (void)pyramidal_levels;
  // Ensure GPU is initialized
  init_gpu_memory(dist_surface, pyramidal_levels);

  // std::cout << "[Optical Flow GPU] Computing optical flow on GPU..." << std::endl;

  // Update wrapper pointer (new frame)
  dist_surface_wrapper.data = dist_surface.ptr();

  // std::cout << "[Optical Flow GPU] Configuration: "
  //           << "max_flow=" << max_flow << ", "
  //           << "gamma=[";

  /* Configure filter ONLY if changed */
  if (max_flow != last_max_flow) {
    filter.setMaxFlow(max_flow);
    last_max_flow = max_flow;
  }

  if (gamma != last_gamma) {
    filter.setGamma(gamma);
    last_gamma = gamma;
  }

  if (smooth_iterations != last_smooth_iterations) {
    filter.setSmoothIterations(smooth_iterations);
    last_smooth_iterations = smooth_iterations;
  }

  // std::cout << "], smooth_iterations=[";

  // Compute optical flow
  filter.loadImage(dist_surface_wrapper);
  filter.compute();
  filter.downloadFlow(optical_flow_wrapper);

  // std::cout << "]" << std::endl;

  // Mask non-edge pixels
  optical_flow_masked.setTo(cv::Scalar::all(0));
  cv::Mat mask;
  dist_surface.convertTo(mask, CV_8U);
  optical_flow.copyTo(optical_flow_masked, mask);

  // std::cout << "[Optical Flow GPU] Optical flow computation done." << std::endl;

  return optical_flow_masked;
}
