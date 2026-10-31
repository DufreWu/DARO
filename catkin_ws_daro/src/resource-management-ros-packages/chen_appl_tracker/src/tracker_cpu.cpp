#include <gflags/gflags.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <map>
#include <vector>
#include <mutex>
#include <fstream>

#include "tracker_cpu.h"
#include "optimizer.h"
#include "patch.h"
#include "types.h"

// ================================
// FLAGS
// ================================
DECLARE_int32(max_corners);
DECLARE_int32(min_corners);
DECLARE_int32(patch_size);
DECLARE_int32(batch_size);
DECLARE_int32(min_distance);
DECLARE_int32(lk_window_size);
DECLARE_int32(num_pyramidal_layers);
DECLARE_int32(block_size);

DECLARE_double(displacement_px);
DECLARE_double(first_image_t);
DECLARE_double(quality_level);
DECLARE_double(k);
DECLARE_double(tracking_quality);

DECLARE_string(bootstrap);
DECLARE_string(tracks_file_txt);

// ================================
// FORWARD DECLARATIONS
// ================================
void extractPatches(
    Patches& patches,
    const int& num_patches,
    const ImageBuffer::iterator& image_it);

void setBatchSize(Patch& patch, const cv::Mat& I_x, const cv::Mat& I_y, const double& d);
void updatePatch(Patch& patch, const dvs_msgs::Event& ev);

void bootstrapAllPossiblePatches(
    std::vector<Patch>& patches,
    const std::map<ros::Time, cv::Mat>::iterator& image_it);

void addFeatures(
    std::vector<int>& lost_indices,
    const std::map<ros::Time, cv::Mat>::iterator& image_it);

void resetPatches(
    Patches& new_patches,
    std::vector<int>& lost_indices,
    const ImageBuffer::iterator& image_it);

void bootstrapFeatureEvents(Patch& patch, const cv::Mat& event_frame);
void bootstrapAllPossiblePatches(Patches& patches, const ImageBuffer::iterator& image_it);
void bootstrapFeatureKLT(Patch& patch, const cv::Mat& last_image, const cv::Mat& current_image);
bool updateFirstImageBeforeTime(ros::Time t_start, ImageBuffer::iterator& current_image_it);
bool shouldDiscard(Patch& patch);

// ================================
// GLOBAL STATE (CPU owns this)
// ================================

bool initialized = false;
bool got_first_image = false;
static nlls::Optimizer optimizer;

cv::Size sensor_size(0, 0);

using ImageBuffer = std::map<ros::Time, cv::Mat>;
ImageBuffer images;
ImageBuffer::iterator current_image_it;

// static Patches patches;
std::vector<Patch> patches;
std::vector<int> lost_indices;
std::ofstream tracks_file_;
std::map<int, std::pair<cv::Mat, cv::Mat>> patch_gradients_;

std::mutex images_mutex;

// ================================
// INIT / FREE
// ================================

void tracker_cpu_init()
{
    if (initialized)
        return;

    patches.clear();
    lost_indices.clear();
    images.clear();

    got_first_image = false;
    initialized = true;
}

void tracker_cpu_free()
{
    patches.clear();
    lost_indices.clear();
    images.clear();
    initialized = false;
}

// ================================
// IMAGE CALLBACK (CPU)
// ================================

void tracker_cpu_image(
    const sensor_msgs::Image::ConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;

    try
    {
        cv_ptr = cv_bridge::toCvCopy(
            msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    if (cv_ptr->header.stamp.toSec() < FLAGS_first_image_t)
        return;

    if (sensor_size.width <= 0)
        sensor_size = cv::Size(msg->width, msg->height);

    std::lock_guard<std::mutex> lock(images_mutex);
    images.emplace(msg->header.stamp, cv_ptr->image.clone());

    if (!got_first_image)
    {
        current_image_it = images.begin();
        got_first_image = true;

        extractPatches(
            patches,
            FLAGS_max_corners,
            current_image_it);
    }
}

// ================================
// EVENTS CALLBACK (CPU)
// ================================

void tracker_cpu_events(const dvs_msgs::EventArray::ConstPtr& msg) {
    if (!initialized || !got_first_image)
        return;

    if (sensor_size.width <= 0)
        sensor_size = cv::Size(msg->width, msg->height);

    std::lock_guard<std::mutex> lock(images_mutex);

    for (const auto& ev : msg->events)
    {
        for (auto& patch : patches)
        {
            updatePatch(patch, ev);
        }

        if (updateFirstImageBeforeTime(
                ev.ts, current_image_it))
        {
            if (FLAGS_bootstrap == "klt")
            {
                bootstrapAllPossiblePatches(
                    patches,
                    current_image_it);
            }

            if (lost_indices.size() >
                static_cast<size_t>(
                    FLAGS_max_corners - FLAGS_min_corners))
            {
                addFeatures(
                    lost_indices,
                    current_image_it);
            }

            if (current_image_it != images.begin())
            {
                auto old_it = current_image_it;
                --old_it;
                images.erase(old_it);
            }
        }
    }
}

void extractPatches(Patches &patches, const int& num_patches, const ImageBuffer::iterator& image_it) {extractPatches(patches, static_cast<int>(lost_indices.size()), image_it);extractPatches(patches, static_cast<int>(lost_indices.size()), image_it);
    std::vector<cv::Point2d> features;

    // mask areas which are within a distance min_distance of other features or along the border.
    int hp = (FLAGS_patch_size - 1 ) / 2;
    int h = sensor_size.height;
    int w = sensor_size.width;
    cv::Mat mask = cv::Mat::ones(sensor_size, CV_8UC1);
    mask.rowRange(0, hp).colRange(0, w-1).setTo(0);
    mask.rowRange(h - hp, h - 1).colRange(0, w-1).setTo(0);
    mask.rowRange(0, h-1).colRange(0, hp).setTo(0);
    mask.rowRange(0, h-1).colRange(w - hp, w - 1).setTo(0);

    const int& min_distance = FLAGS_min_distance;
    for (Patch &patch: patches)
    {
        if (patch.lost_) continue;

        double min_x = std::fmax(patch.center_.x - min_distance, 0);
        double max_x = std::fmin(patch.center_.x + min_distance, w-1);
        double min_y = std::fmax(patch.center_.y - min_distance, 0);
        double max_y = std::fmin(patch.center_.y + min_distance, h-1);
        mask.rowRange(min_y, max_y).colRange(min_x, max_x).setTo(0);
    }

    // extract harris corners which are suitable
    // since they correspond to strong edges which also generate alot of events.
    VLOG(2) << "Harris corner detector with N=" << num_patches << " quality=" << FLAGS_quality_level
            << " min_dist=" << FLAGS_min_distance << " block_size=" << FLAGS_block_size << " k=" << FLAGS_k
            << " image_depth=" << image_it->second.depth() << " mask_ratio=" << cv::sum(mask)[0]/(mask.cols*mask.rows);

    cv::goodFeaturesToTrack(image_it->second, features, num_patches,
                            FLAGS_quality_level,
                            FLAGS_min_distance, mask,
                            FLAGS_block_size,
                            true,
                            FLAGS_k);

    // initialize patches centered at the features with an initial pixel warp
    VLOG(1) << "Extracted " << features.size() << " new features on image at t=" << std::setprecision(15) << image_it->first.toSec() << " s.";
    for (int i=0; i<features.size(); i++)
    {
        const cv::Point2f &feature = features[i];
        patches.emplace_back(feature, image_it->first);
        Patch &patch = patches[patches.size()-1];
        if (tracks_file_.is_open())
            tracks_file_ << patch.id_ << " " << patch.t_curr_ << " " << patch.center_.x << " " << patch.center_.y << std::endl;
    }
}

void addFeatures(std::vector<int>& lost_indices, const ImageBuffer::iterator& image_it) {
    // find new patches to replace them lost features
    Patches patches;
    extractPatches(patches, lost_indices.size(), image_it);

    if (patches.size() != 0)
    {
        // pass the new image to the optimizer to use for future optimizations
        optimizer.precomputeLogImageArray(patches, image_it);

        // reset all lost features with newly initialized ones
        resetPatches(patches, lost_indices, image_it);
    }
}

void updatePatch(Patch& patch, const dvs_msgs::Event &event) {
    // if patch is lost or event does not fall within patch
    // or the event has occurred before the most recent patch timestamp
    // or the patch has not been bootstrapped yet, do not process event
    if (patch.lost_ ||
    (FLAGS_bootstrap == "klt" && !patch.initialized_) ||
    !patch.contains(event.x, event.y) ||
    patch.t_curr_>event.ts)
        return;

    patch.insert(event);

    // start optimization if there are update_rate new events in the patch
    int update_rate = std::min<int>(patch.update_rate_, patch.batch_size_);	    
    if (patch.event_buffer_.size() < patch.batch_size_ || patch.event_counter_ < update_rate)	
        return;

    // compute event frame according to equation (2) in the paper
    cv::Mat event_frame;
    patch.getEventFramesAndReset(event_frame);

    // bootstrap using the events
    ros::Time t_prev = patch.t_curr_;
    if (!patch.initialized_ && FLAGS_bootstrap == "events")
        bootstrapFeatureEvents(patch, event_frame);

    // update feature position and recompute the adaptive batchsize
    optimizer.optimizeParameters(event_frame, patch);

    if (tracks_file_.is_open())
        tracks_file_ << patch.id_ << " " << patch.t_curr_ << " " << patch.center_.x << " " << patch.center_.y << std::endl;

    setBatchSize(patch, patch_gradients_[&patch - &patches[0]].first, patch_gradients_[&patch - &patches[0]].second, FLAGS_displacement_px);

    if (!shouldDiscard(patch))
        return;

    // if the patch has been lost record it in lost_indices_
    patch.lost_ = true;
    lost_indices.push_back(&patch - &patches[0]);
}

bool updateFirstImageBeforeTime(ros::Time t_start, ImageBuffer::iterator& current_image_it) {
    bool next_image = false;
    auto next_image_it = current_image_it;

    while (next_image_it->first < t_start)
    {
        ++next_image_it;
        if (next_image_it == images.end())
            break;

        if (next_image_it->first < t_start)
        {
            next_image = true;
            current_image_it = next_image_it;
        }
    }

    return next_image;
}

void setBatchSize(Patch& patch, const cv::Mat& I_x, const cv::Mat& I_y, const double& d) {
    // implements the equation (15) of the paper
    cv::Mat gradient = d * std::cos(patch.flow_angle_) * I_x + d * std::sin(patch.flow_angle_) * I_y;
    patch.batch_size_ = std::min<double>(cv::norm(gradient, cv::NORM_L1), FLAGS_batch_size);
    patch.batch_size_ = std::max<int>(5, patch.batch_size_);
}

void bootstrapFeatureEvents(Patch& patch, const cv::Mat& event_frame)
{
    // Implement a bootstrapping mechanism for computing the optical flow direction via
    // \nabla I \cdot v= - \Delta E --> v = - \nabla I ^ \dagger \Delta E (assuming no translation or rotation
    // of the feature.
    int index = &patch - &patches[0];

    cv::Mat& I_x = patch_gradients_[index].first;
    cv::Mat& I_y = patch_gradients_[index].second;

    double s_I_xx = cv::sum(I_x.mul(I_x))[0];
    double s_I_yy = cv::sum(I_y.mul(I_y))[0];
    double s_I_xy = cv::sum(I_x.mul(I_y))[0];
    double s_I_xt = cv::sum(I_x.mul(event_frame))[0];
    double s_I_yt = cv::sum(I_y.mul(event_frame))[0];

    cv::Mat M = (cv::Mat_<double>(2,2) << s_I_xx, s_I_xy, s_I_xy, s_I_yy);
    cv::Mat b = (cv::Mat_<double>(2,1) << s_I_xt, s_I_yt);

    cv::Mat v = -M.inv() * b;

    patch.flow_angle_ = std::atan2(v.at<double>(0,0), v.at<double>(1,0));
    patch.initialized_ = true;
}

bool shouldDiscard(Patch& patch)
{
    bool out_of_fov = (patch.center_.y < 0 || patch.center_.y >= sensor_size.height || patch.center_.x < 0 || patch.center_.x >= sensor_size.width);
    bool exceeded_error = patch.tracking_quality_ < FLAGS_tracking_quality;

    return exceeded_error || out_of_fov;
}

void resetPatches(Patches& new_patches, std::vector<int>& lost_indices, const ImageBuffer::iterator& image_it)
{
    const int& p = (FLAGS_patch_size-1)/2;
    cv::Mat I_x, I_y, I_x_padded, I_y_padded;

    optimizer.getLogGradients(image_it->second, I_x, I_y);
    cv::copyMakeBorder(I_x, I_x_padded, p, p, p, p, cv::BORDER_REFLECT);
    cv::copyMakeBorder(I_y, I_y_padded, p, p, p, p, cv::BORDER_REFLECT);

    for (int i=new_patches.size()-1; i>=0; i--)
    {
        int index = lost_indices[i];
        // for each lost feature decrement the ref counter of the optimizer (will free image gradients when no more
        // features use the image with timestamp patches[index].t_init_
        optimizer.decrementCounter(patches[index].t_init_);

        // reset lost patches with new ones
        Patch& reset_patch = new_patches[i];
        patches[index].reset(reset_patch.center_, reset_patch.t_init_);
        lost_indices.erase(lost_indices.begin()+i);

        // reinitialize the image gradients of new features
        const int x_min = reset_patch.center_.x-p;
        const int y_min = reset_patch.center_.y-p;
        cv::Mat p_I_x = I_x_padded.rowRange(y_min, y_min+2*p+1).colRange(x_min, x_min+2*p+1);
        cv::Mat p_I_y = I_y_padded.rowRange(y_min, y_min+2*p+1).colRange(x_min, x_min+2*p+1);
        patch_gradients_[index] = std::make_pair(p_I_x.clone(), p_I_y.clone());
        setBatchSize(reset_patch, patch_gradients_[index].first, patch_gradients_[index].second, FLAGS_displacement_px);
    }
}

void bootstrapAllPossiblePatches(Patches& patches, const ImageBuffer::iterator& image_it)
{
    for (int i=0; i<patches.size(); i++)
    {
        Patch &patch = patches[i];

        // if a patch is already bootstrapped, lost or has just been extracted
        if (patch.initialized_ || patch.lost_ || patch.t_init_ == image_it->first)
            continue;

        // perform bootstrapping using KLT and the first 2 frames, and compute the adaptive batch size
        // with the newly found parameters
        bootstrapFeatureKLT(patch, images[patch.t_init_], image_it->second);
        setBatchSize(patch, patch_gradients_[i].first, patch_gradients_[i].second, FLAGS_displacement_px);
    }
}

void bootstrapFeatureKLT(Patch& patch, const cv::Mat& last_image, const cv::Mat& current_image)
{
    // bootstrap feature by initializing its warp and optical flow with KLT on successive images
    std::vector<cv::Point2f> points = {patch.init_center_};
    std::vector<cv::Point2f> next_points;

    // track feature for one frame
    std::vector<float> error;
    std::vector<uchar> status;
    cv::Size window(FLAGS_lk_window_size, FLAGS_lk_window_size);
    cv::calcOpticalFlowPyrLK(last_image, current_image, points, next_points, status, error, window, FLAGS_num_pyramidal_layers);

    // compute optical flow angle as direction where the feature moved
    double opt_flow_angle = std::atan2(next_points[0].y - points[0].y, next_points[0].x - points[0].x);
    patch.flow_angle_ = opt_flow_angle;

    // initialize warping as pure translation to new point
    patch.warping_.at<double>(0,2) = -(next_points[0].x - points[0].x);
    patch.warping_.at<double>(1,2) = -(next_points[0].y - points[0].y);
    patch.warpPixel(patch.init_center_, patch.center_);

    // check if new patch has been lost due to leaving the fov
    bool should_discard = bool (patch.center_.y < 0 || patch.center_.y >= sensor_size.height || patch.center_.x < 0 || patch.center_.x >= sensor_size.width);
    if (should_discard)
    {
        patch.lost_ = true;
        lost_indices.push_back(&patch - &patches[0]);
    }
    else
    {
        patch.initialized_ = true;
        patch.t_curr_ = current_image_it->first;
        if (tracks_file_.is_open())
            tracks_file_ << patch.id_ << " " << patch.t_curr_ << " " << patch.center_.x << " " << patch.center_.y << std::endl;
    }
}


