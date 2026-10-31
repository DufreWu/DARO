#include "fa_harris_detector_gpu.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

namespace fa_harris_gpu {

// CUDA kernel to compute Harris scores
__global__ void computeHarrisScoresKernel(float *time_surface, float *harris_scores, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Example Harris score computation (simplified)
    float score = 0.0f;
    if (x > 1 && x < width - 1 && y > 1 && y < height - 1) {
        float center = time_surface[idx];
        float left = time_surface[idx - 1];
        float right = time_surface[idx + 1];
        float top = time_surface[idx - width];
        float bottom = time_surface[idx + width];

        // Simplified Harris score computation
        score = (center - left) * (center - right) + (center - top) * (center - bottom);
    }

    harris_scores[idx] = score;
}

// Constructor
FAHarrisDetector::FAHarrisDetector() : width_(240), height_(180), d_time_surface_(nullptr), d_harris_scores_(nullptr) {}

// Destructor
FAHarrisDetector::~FAHarrisDetector() {
    freeCudaMemory();
}

// Initialize CUDA memory
void FAHarrisDetector::initializeCudaMemory(int width, int height) {
    std::cout << "Initializing CUDA Memory: Width=" << width << ", Height=" << height << std::endl;
    if (width <= 0 || height <= 0) {
        std::cerr << "Error: Invalid dimensions for CUDA memory allocation." << std::endl;
        return;
    }

    width_ = width;
    height_ = height;

    cudaMalloc(&d_time_surface_, width * height * sizeof(float));
    cudaMalloc(&d_harris_scores_, width * height * sizeof(float));

    std::cout << "CUDA Memory initialized successfully." << std::endl;
}

// Free CUDA memory
void FAHarrisDetector::freeCudaMemory() {
    if (d_time_surface_) cudaFree(d_time_surface_);
    if (d_harris_scores_) cudaFree(d_harris_scores_);
}

// Process events on the GPU
void FAHarrisDetector::processEventsOnGpu(const dvs_msgs::EventArray &events) {
    if (width_ <= 0 || height_ <= 0) {
        std::cerr << "Error: FAHarrisDetector is not initialized. Call initializeCudaMemory() first." << std::endl;
        return;
    }
    
    std::vector<float> time_surface(width_ * height_, 0.0f);

    std::cout << "events.size() " << events.events.size() << std::endl;
    for (const auto &e : events.events) {
        int idx = e.y * width_ + e.x;
        if (idx >= 0 && idx < width_ * height_) {
            time_surface[idx] = static_cast<float>(e.ts.toSec());
        } else {
            std::cerr << "Invalid index: " << idx << " (Width: " << width_ << ", Height: " << height_ << ")" << std::endl;
        }
    }
    
    cudaMemcpy(d_time_surface_, time_surface.data(), width_ * height_ * sizeof(float), cudaMemcpyHostToDevice);

    // **Check if gridSize is valid before kernel launch**
    dim3 blockSize(16, 16);
    dim3 gridSize((width_ + blockSize.x - 1) / blockSize.x, (height_ + blockSize.y - 1) / blockSize.y);
    
    if (gridSize.x == 0 || gridSize.y == 0) {
        std::cerr << "Error: Invalid gridSize (" << gridSize.x << ", " << gridSize.y << ")" << std::endl;
        return;
    }

    computeHarrisScoresKernel<<<gridSize, blockSize>>>(d_time_surface_, d_harris_scores_, width_, height_);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
}

} // namespace fa_harris_gpu
