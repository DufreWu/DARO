#ifndef FA_HARRIS_DETECTOR_GPU_H
#define FA_HARRIS_DETECTOR_GPU_H

#include <ros/ros.h>
#include "dvs_msgs/EventArray.h"
#include <cuda_runtime.h>
#include <vector>

namespace fa_harris_gpu
{

class FAHarrisDetector
{
public:
    FAHarrisDetector();
    ~FAHarrisDetector();

    bool isCorner(const dvs_msgs::Event &e);
    bool isCornerWithoutFilter(const dvs_msgs::Event &e);
    bool addNewEventToTimeSurface(const dvs_msgs::Event &e);
    bool isCornerCandidate(const dvs_msgs::Event &e);
    bool isCornerCandidateRefined(const dvs_msgs::Event &e);
    double getHarrisScore(int img_x, int img_y, bool polarity);
    bool checkPatch(const dvs_msgs::Event &e);

    // Replace Eigen::MatrixXi with a standard C++ 2D vector
    std::vector<std::vector<int>> getPatch();

    // CUDA-specific methods
    void initializeCudaMemory(int width, int height);
    void freeCudaMemory();
    void processEventsOnGpu(const dvs_msgs::EventArray &events);

private:
    int width_ = 240;
    int height_ = 180;
    float *d_time_surface_; // Device memory for the time surface
    float *d_harris_scores_; // Device memory for Harris scores
};

} // namespace fa_harris_gpu

#endif // FA_HARRIS_DETECTOR_GPU_H