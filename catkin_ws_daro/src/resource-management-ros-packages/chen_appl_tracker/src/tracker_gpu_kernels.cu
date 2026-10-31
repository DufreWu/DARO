#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {

// ======================================
// Plain POD structs only
// ======================================

struct GPUPatch
{
    float cx, cy;
    float radius;
    int event_count;
};

struct GPUEvent
{
    int x, y;
};

// ======================================
// CUDA kernels
// ======================================

__global__ void kernel_update_patch(
    const GPUEvent* events,
    int num_events,
    GPUPatch* patches,
    int num_patches)
{
    int e = blockIdx.x;
    int p = threadIdx.x;

    if (e >= num_events || p >= num_patches)
        return;

    GPUEvent ev = events[e];
    GPUPatch& patch = patches[p];

    float dx = ev.x - patch.cx;
    float dy = ev.y - patch.cy;

    if (dx * dx + dy * dy <= patch.radius * patch.radius)
    {
        atomicAdd(&patch.event_count, 1);
    }
}

// ======================================
// C ABI launchers (callable from C++)
// ======================================

void launch_update_patch(
    const GPUEvent* d_events,
    int num_events,
    GPUPatch* d_patches,
    int num_patches)
{
    dim3 blocks(num_events);
    dim3 threads(num_patches);

    kernel_update_patch<<<blocks, threads>>>(
        d_events, num_events, d_patches, num_patches);
}

} // extern "C"
