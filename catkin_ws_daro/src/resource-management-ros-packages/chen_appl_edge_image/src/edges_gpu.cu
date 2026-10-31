/**
 * edges_gpu.cu
 * This code corresponds to the events -> edge image computation, on CUDA-capable GPU.
 */

#include "edge_image/edges_gpu.hpp"
#include "edge_image/types.hpp"

static dvs_msgs::Event* d_events = nullptr;
static size_t d_events_capacity = 0;

static uint8_t* d_edge_image = nullptr;
static size_t d_edge_image_size = 0;

static GpuPoint2i* d_undistortion = nullptr;
static size_t d_undistortion_size = 0;

/* ============================
 * CUDA Kernel
 * ============================ */
__global__ void evts_to_edge_image_kernel(
    const dvs_msgs::Event* events,
    int num_events,
    uint8_t* edge_image,
    int height,
    int width,
    const GpuPoint2i* undistortion,
    bool use_undistortion)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_events) return;

  int x = events[idx].x;
  int y = events[idx].y;

  if (use_undistortion) {
    GpuPoint2i p = undistortion[y * width + x];
    x = p.x;
    y = p.y;

    if (x < 0 || x >= width || y < 0 || y >= height) {
      return;
    }
  }

  edge_image[y * width + x] = 255;
}

void evts_to_edge_image_gpu(
    const vector<dvs_msgs::Event>& evts_to_process,
    Mat* edge_image_ptr,
    int height,
    int width,
    const Mat& undistortion_matrix)
{
  if (evts_to_process.empty()) {
    return;
  }

  const size_t num_events = evts_to_process.size();
  const size_t events_bytes = num_events * sizeof(dvs_msgs::Event);
  const size_t image_bytes = height * width * sizeof(uint8_t);

  /* ----------------------------
   * Allocate / resize GPU buffers
   * ---------------------------- */
  if (events_bytes > d_events_capacity) {
    cudaFree(d_events);
    cudaMalloc(&d_events, events_bytes);
    d_events_capacity = events_bytes;
  }

  if (image_bytes > d_edge_image_size) {
    cudaFree(d_edge_image);
    cudaMalloc(&d_edge_image, image_bytes);
    d_edge_image_size = image_bytes;
  }

  if (!undistortion_matrix.empty()) {
    size_t undist_bytes = height * width * sizeof(GpuPoint2i);
    if (undist_bytes > d_undistortion_size) {
      cudaFree(d_undistortion);
      cudaMalloc(&d_undistortion, undist_bytes);
      d_undistortion_size = undist_bytes;
    }
    cudaMemcpy(
        d_undistortion,
        undistortion_matrix.ptr<GpuPoint2i>(),
        undist_bytes,
        cudaMemcpyHostToDevice);
  }

  /* ----------------------------
   * Copy events + clear image
   * ---------------------------- */
  cudaMemcpy(d_events, evts_to_process.data(), events_bytes, cudaMemcpyHostToDevice);
  cudaMemset(d_edge_image, 0, image_bytes);

  /* ----------------------------
   * Launch kernel
   * ---------------------------- */
  const int threads = 256;
  const int blocks = (num_events + threads - 1) / threads;

  evts_to_edge_image_kernel<<<blocks, threads>>>(
      d_events,
      num_events,
      d_edge_image,
      height,
      width,
      undistortion_matrix.empty() ? nullptr : d_undistortion,
      !undistortion_matrix.empty());

  cudaDeviceSynchronize();

  /* ----------------------------
   * Copy back to cv::Mat
   * ---------------------------- */
  *edge_image_ptr = Mat::zeros(height, width, CV_8U);
  cudaMemcpy(
      edge_image_ptr->data,
      d_edge_image,
      image_bytes,
      cudaMemcpyDeviceToHost);
}


/* ============================
 * Cleanup
 * ============================ */
void free_gpu_memory()
{
  cudaFree(d_events);
  cudaFree(d_edge_image);
  cudaFree(d_undistortion);

  d_events = nullptr;
  d_edge_image = nullptr;
  d_undistortion = nullptr;

  d_events_capacity = 0;
  d_edge_image_size = 0;
  d_undistortion_size = 0;
}