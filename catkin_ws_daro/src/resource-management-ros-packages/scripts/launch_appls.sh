#!/bin/bash

#######################################
function launch_event_publisher() { # pass number of events as the first arg.
    echo "Launching event publisher..."
    command="
        source ../../devel/setup.bash;
        rosrun chen_rnd_event_publisher publish_events_fixed_track --track-length 100 --n-events $1 --init-speed $speed --log-dir $log_dir --hz 30 --loop;
        exec sh;
    "
    screen -S chen_rosbag -md bash -c "$command"
    echo "Event publisher launched!"
}

#######################################
function launch_event_counter() { # args: ()
    echo "Launching event counter..."
    command="
        source ../../devel/setup.bash;
        roslaunch chen_appl_null_event_counter/launch/counter.launch;
        exec sh;
    "
    screen -S chen_event_counter -md bash -c "$command"
    echo "Event counter launched!"
}

#######################################
function launch_fa_harris() {
    echo "Launching fa_harris.."
    command="
        source ../../devel/setup.bash; 
        roslaunch fa_harris corner.launch > $log_dir/fa_harris.log 2>&1;
        exec sh;
    "
    screen -S chen_fa_harris -dm bash -c "$command"
    echo "fa_harris launched!"
}

#######################################
function launch_img_reconstruction() {
    echo "Launching img_reconstruction.."
    command="
        source ../../devel/setup.bash; 
        roslaunch pure_event_reconstruction no_popup.launch > $log_dir/img_reconstruction.log 2>&1;
        exec sh;
    "
    screen -S chen_img_reconstruction -dm bash -c "$command"
    echo "img_reconstruction launched!"
}

#######################################
function launch_eklt() {
    echo "Launching eklt.."
    command="
        source ../../devel/setup.bash; 
        roslaunch eklt eklt.launch > $log_dir/eklt.log 2>&1;
        exec sh;
    "
    screen -S chen_eklt -dm bash -c "$command"
    echo "eklt launched!"
}

#######################################
function launch_edge_image() {
    echo "Launching edge_image.."
    command="
        source ../../devel/setup.bash; 
        roslaunch edge_image davis346_edge_image.launch > $log_dir/edge_image.log 2>&1;
        exec sh;
    "
    screen -S chen_edge_image -dm bash -c "$command"
    echo "edge_image launched!"
}

#######################################
function launch_denoising_filling() {
    echo "Launching denoising_filling.."
    command="
        source ../../devel/setup.bash; 
        roslaunch denoising_filling davis346_denoising_filling.launch > $log_dir/denoising_filling.log 2>&1;
        exec sh;
    "
    screen -S chen_denoising_filling -dm bash -c "$command"
    echo "denoising_filling launched!"
}

#######################################
function launch_distance_surface() {
    echo "Launching distance_surface.."
    command="
        source ../../devel/setup.bash; 
        roslaunch distance_surface davis346_distance_surface.launch > $log_dir/distance_surface.log 2>&1;
        exec sh;
    "
    screen -S chen_distance_surface -dm bash -c "$command"
    echo "distance_surface launched!"
}

#######################################
function launch_optical_flow() {
    echo "Launching optical_flow.."
    command="
        source ../../devel/setup.bash; 
        roslaunch optical_flow davis346_optical_flow.launch > $log_dir/optical_flow.log 2>&1;
        exec sh;
    "
    screen -S chen_optical_flow -dm bash -c "$command"
    echo "optical_flow launched!"
}

#######################################
function launch_tracker() {
    echo "Launching tracker.."
    command="
        source ../../devel/setup.bash; 
        roslaunch tracker tracker.launch > $log_dir/tracker.log 2>&1;
        exec sh;
    "
    screen -S chen_tracker -dm bash -c "$command"
    echo "tracker launched!"
}