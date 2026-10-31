#include <cuda_runtime.h>
#include <vector>
#include <mutex>

#include <dvs_msgs/EventArray.h>
#include <std_msgs/Header.h> // Added missing dependency
#include <sensor_msgs/Image.h>

#include <gflags/gflags.h>

#include "tracker_gpu.h"
#include "optimizer.h"
#include "patch.h"
#include "types.h"

// ===============================
// gflags (CPU side ONLY)
// ===============================
DECLARE_int32(patch_size);
DECLARE_double(displacement_px);

// ===============================
// CUDA kernel interface
// ===============================
extern "C" {

// Must match tracker_gpu_kernels.cu
struct GPUEvent {
    int x;
    int y;
};

struct GPUPatch {
    float cx;
    float cy;
    float radius;
    int event_count;
};

void launch_update_patch(
    const GPUEvent* d_events,
    int num_events,
    GPUPatch* d_patches,
    int num_patches);
}

// ===============================
// GPU state 
// ===============================
static bool gpu_initialized = false;
static uint8_t* d_event_frame = nullptr;
static size_t d_event_frame_bytes = 0;

static GPUEvent* d_events = nullptr;
static size_t d_events_capacity = 0;

static GPUPatch* d_patches = nullptr;
static size_t d_patches_capacity = 0;

static nlls::Optimizer optimizer;

// ===============================
// Init / Free
// ===============================
void tracker_gpu_init(const dvs_msgs::EventArray::ConstPtr& msg)
{
    if (!gpu_initialized)
        gpu_initialized = true;

    const size_t needed = msg->events.size();

    if (needed > d_events_capacity)
    {
        if (d_events)
            cudaFree(d_events);

        cudaMalloc(&d_events, needed * sizeof(GPUEvent));
        d_events_capacity = needed;
    }
}

void tracker_gpu_free()
{
    if (d_events)
        cudaFree(d_events);

    if (d_patches)
        cudaFree(d_patches);

    d_events = nullptr;
    d_patches = nullptr;
    d_events_capacity = 0;
    d_patches_capacity = 0;
    gpu_initialized = false;
}

// ===============================
// Event path (GPU accelerated)
// ===============================
void tracker_gpu_events(
    const dvs_msgs::EventArray::ConstPtr& msg)
{
    if (!gpu_initialized || msg->events.empty())
        return;

    const int num_events = static_cast<int>(msg->events.size());

    // ---------------------------------
    // Copy events to GPU
    // ---------------------------------
    std::vector<GPUEvent> h_events(num_events);

    for (int i = 0; i < num_events; ++i)
    {
        h_events[i].x = msg->events[i].x;
        h_events[i].y = msg->events[i].y;
    }

    cudaMemcpy(
        d_events,
        h_events.data(),
        num_events * sizeof(GPUEvent),
        cudaMemcpyHostToDevice);

    // ---------------------------------
    // Mirror patches from CPU
    // ---------------------------------
    extern std::vector<Patch> patches;   // defined in tracker_cpu.cpp

    const int num_patches = static_cast<int>(patches.size());
    if (num_patches == 0)
        return;

    if (num_patches > d_patches_capacity)
    {
        if (d_patches)
            cudaFree(d_patches);

        cudaMalloc(&d_patches, num_patches * sizeof(GPUPatch));
        d_patches_capacity = num_patches;
    }

    const float radius = float((FLAGS_patch_size - 1) / 2);

    std::vector<GPUPatch> h_patches(num_patches);
    for (int i = 0; i < num_patches; ++i)
    {
        h_patches[i].cx = patches[i].center_.x;
        h_patches[i].cy = patches[i].center_.y;
        h_patches[i].radius = radius;
        h_patches[i].event_count = 0;
    }

    cudaMemcpy(
        d_patches,
        h_patches.data(),
        num_patches * sizeof(GPUPatch),
        cudaMemcpyHostToDevice);

    // ---------------------------------
    // Launch CUDA kernel
    // ---------------------------------
    launch_update_patch(
        d_events,
        num_events,
        d_patches,
        num_patches);

    cudaDeviceSynchronize();
}

void init_gpu_memory(
    const dvs_msgs::EventArray::ConstPtr& msg)
{
    if (gpu_initialized)
        return;

    ROS_INFO("[tracker_gpu] Initializing GPU memory");

    // If no message yet, delay allocation
    if (!msg)
        return;

    const int width  = msg->width;
    const int height = msg->height;

    if (width <= 0 || height <= 0)
        return;

    d_event_frame_bytes = width * height * sizeof(uint8_t);

    cudaMalloc(&d_event_frame, d_event_frame_bytes);
    cudaMemset(d_event_frame, 0, d_event_frame_bytes);

    gpu_initialized = true;
}

void free_gpu_memory()
{
    if (!gpu_initialized)
        return;

    ROS_INFO("[tracker_gpu] Freeing GPU memory");

    if (d_event_frame)
    {
        cudaFree(d_event_frame);
        d_event_frame = nullptr;
    }

    d_event_frame_bytes = 0;
    gpu_initialized = false;
}
