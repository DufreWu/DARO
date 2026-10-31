#!/bin/bash
source /opt/ros/melodic/setup.bash

log_dir="./temp/exp-9999-cpu-gpu-all-mission"
speed=0.5
# 345600 499200 652800 806400 960000 1113600 1267200 1420800 1574400 1728000 1881600 2035200 
# 114750 216750 318750 420750 522750 624750 726750 854250 930750 1032750 1122000 1236750 1300500
freq_l=499200
freq_b=499200
freq_g=216750
sleep_time=30
total_time=300
num_ev_base=10000000
# print the command
echo "$0 $@"

src_dir=$( dirname -- "$0")
. $src_dir/parse_options.sh
. $src_dir/launch_appls.sh
. $src_dir/launch_controller.sh
. $src_dir/kill_funcs.sh

trap 'kill_all_screens; exit 0;' SIGTERM SIGINT EXIT

mkdir -p $log_dir;

log_dir=$(realpath $log_dir);

echo "Logging in $log_dir"

function start_roscore() { # args: ($log_dir)
    echo "Starting roscore...";
    rosparam set /chen/log_dir $log_dir;
    if [ $? -ne 0 ]; then
        screen -S chen_core -dm bash -c "source ../../devel/setup.bash; roscore; exec sh;"
        rosparam set /chen/log_dir $log_dir; 
        while [ $? -ne 0 ]; do
            sleep 1;
            rosparam set /chen/log_dir $log_dir;
        done;
    fi
    echo "roscore ready!" 
}

function mv_result_files() { # args: (name)
    name=$1
    mv -f $log_dir/{sys,${name}_sys}.csv;
    mv -f $log_dir/{appls,${name}_appls}.csv;
    mv -f $log_dir/{hc,${name}_hc}.csv;
    mv -f $log_dir/{latest_model,${name}_latest_model}.csv;
}

function run_event_publisher() {
    complexity=$1
    echo "Complexity is: $complexity"
    # launch_event_counter;
    if [[ "$complexity" == "0" ]]; then
        num_ev_base=10000000
        launch_event_publisher $num_ev_base
    elif [[ "$complexity" == "1" ]]; then
        num_ev_base=20000000
        launch_event_publisher $num_ev_base
    elif [[ "$complexity" == "2" ]]; then
        num_ev_base=35000000
        launch_event_publisher $num_ev_base
    else 
        launch_event_publisher $num_ev_base
    fi
}

function run_scenario_avg_speed() {
    echo "############# AVG #############"
    name="constavg"; speed=2.5; freq_b=2035200, freq_l=2035200, freq_g=1300500;
    launch_const_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps $1 $2;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;    
}

function run_scenario_search_seq() {
    echo "############# SEARCH #############"
    name="search"; speed=0.5; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_search_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps $1 $2;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_so_seq() {
    echo "############# SO #############"
    name="so"; speed=2; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_const_speed_interactive_freq_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps $1 $2;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_allconst() {
    echo "############# ALL CONST #############"
    name="allconst"; speed=1.0; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_all_consts_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps $1 $2;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_pdqn() {
    echo "############# PDQN #############"
    name="pdqn"; speed=0.5; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_pdqn_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps $1 $2;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_apps() {
    run_event_publisher $1
    sleep 1

    run_apps_repeatedly $2

    kill_edge_image
    sleep 2
}

function run_apps_repeatedly() {
    mission=$1
    for i in $(seq 1 1 1); do
        run_apps_once $mission
    done
}

function run_apps_once() {
    mission=$1

    if [[ "$mission" == "1" ]]; then
        run_apps_once_mission_1
    elif [[ "$mission" == "2" ]]; then
        run_apps_once_mission_2
    elif [[ "$mission" == "3" ]]; then
        run_apps_once_mission_3
    else
        echo "Unknown mission $mission"
        exit 1
    fi
}

function run_apps_once_mission_1() {
    # t = 0s : launch_edge_image (APP-EI)
    launch_edge_image;

    # t = 10s : launch_distance_surface (APP-DS)
    sleep 10;
    launch_distance_surface;

    # t = 20s : launch_optical_flow (APP-OF)
    sleep 10;
    launch_optical_flow;

    # t = 45s : kill_optical_flow
    sleep 25;
    kill_optical_flow;

    # t = 65s : launch_optical_flow
    sleep 20;
    launch_optical_flow;

    # t = 85s : kill_distance_surface
    sleep 20;
    kill_distance_surface;

    # t = 95s : kill_optical_flow
    sleep 10;
    kill_optical_flow;

    # t = 105s : launch_distance_surface
    sleep 10;
    launch_distance_surface;

    # t = 120s : launch_optical_flow
    sleep 15;
    launch_optical_flow;

    # t = 160s : kill_optical_flow
    sleep 40;
    kill_optical_flow;

    # t = 175s : kill_distance_surface
    sleep 15;
    kill_distance_surface;

    # t = 180s : mission end
    sleep 5;
    kill_edge_image;

    # sleep for a while before next round
    sleep 5;
}

function run_apps_once_mission_2() {
    # t = 0s : launch_edge_image
    launch_edge_image;

    # t = 15s : launch_denoising_filling (APP-DF)
    sleep 15;
    launch_denoising_filling;

    # t = 40s : launch_tracker (APP-FT)
    sleep 25;
    launch_tracker;

    # t = 50s : kill_denoising_filling
    sleep 10;
    kill_denoising_filling;

    # t = 75s : launch_denoising_filling
    sleep 25;
    launch_denoising_filling;

    # t = 95s : kill_tracker
    sleep 20;
    kill_tracker;

    # t = 105s : launch_tracker
    sleep 10;
    launch_tracker;

    # t = 115s : kill_denoising_filling
    sleep 10;
    kill_denoising_filling;

    # t = 135s : launch_denoising_filling
    sleep 20;
    launch_denoising_filling;

    # t = 165s : kill_denoising_filling
    sleep 30;
    kill_denoising_filling;

    # t = 175s : kill_tracker
    sleep 10;
    kill_tracker;

    # t = 180s : mission end
    sleep 5;
    kill_edge_image;

    # sleep for a while before next round
    sleep 5;
}

function run_apps_once_mission_3() {
    # t = 0s : launch_distance_surface (APP-DS)
    launch_edge_image;
    launch_distance_surface;

    # t = 10s : launch_tracker (APP-FT)
    sleep 10;
    launch_tracker;

    # t = 35s : kill_tracker
    sleep 25;
    kill_tracker;

    # t = 50s : launch_tracker
    sleep 15;
    launch_tracker;

    # t = 75s : kill_tracker
    sleep 25;
    kill_tracker;

    # t = 90s : launch_tracker
    sleep 15;
    launch_tracker;

    # t = 100s : kill_distance_surface
    sleep 10;
    kill_distance_surface;

    # t = 115s : kill_tracker + relaunch_distance_surface
    sleep 15;
    kill_tracker;
    launch_distance_surface;

    # t = 130s : launch_tracker
    sleep 15;
    launch_tracker;

    # t = 160s : kill_tracker
    sleep 30;
    kill_tracker;

    # t = 180s : kill_distance_surface
    sleep 20;
    kill_distance_surface;
    kill_edge_image;

    # sleep for a while before next round
    sleep 5;
}

sudo cgdelete -r cpuset:mycontroller
start_roscore;

cx=2

for mission in $(seq 1 1 2); do
    echo "Running mission $mission, complexity $cx ..."
    # run_scenario_pdqn $cx $mission;
    # run_scenario_search_seq $cx $mission;
    # run_scenario_so_seq $cx $mission;
    run_scenario_avg_speed $cx $mission;

    mission_log_dir="$log_dir/$mission";
    mkdir -p $mission_log_dir;
    mv -f "$log_dir"/*.csv "$log_dir"/*.log "$log_dir"/*.txt "$mission_log_dir";
done

exit 0
