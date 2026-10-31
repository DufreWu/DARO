#!/bin/bash
source /opt/ros/melodic/setup.bash

log_dir="./temp/exp-9999-cpu-gpu-s2"
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
        num_ev_base=30000000
        launch_event_publisher $num_ev_base
    else 
        launch_event_publisher $num_ev_base
    fi
}

function run_scenario_max_speed() {
    echo "############# MAX #############"
    name="constmax"; speed=5; freq_b=2035200, freq_l=2035200, freq_g=216750;
    launch_const_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps $1;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_min_speed() {
    echo "############# MIN #############"
    name="constmin"; speed=0.5; freq_b=2035200, freq_l=2035200, freq_g=216750;
    launch_const_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps $1;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_avg_speed() {
    echo "############# AVG #############"
    name="constavg"; speed=2.5; freq_b=2035200, freq_l=2035200, freq_g=1300500;
    launch_const_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps $1;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;    
}

function run_scenario_hc() {
    echo "############# HC #############"
    name="hc"; speed=0.5; freq_b=2035200, freq_l=2035200, freq_g=522750;
    launch_hc_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps $1;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_search_seq() {
    echo "############# SEARCH #############"
    name="search"; speed=0.5; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_search_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps $1;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_so_seq() {
    echo "############# SO #############"
    name="so"; speed=2; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_const_speed_interactive_freq_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps $1;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_allconst() {
    echo "############# ALL CONST #############"
    name="allconst"; speed=1.0; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_all_consts_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps $1;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_pdqn() {
    echo "############# PDQN #############"
    name="pdqn"; speed=0.5; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_pdqn_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps $1;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_sac() {
    echo "############# SAC #############"
    name="sac"; speed=0.5; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_sac_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps $1;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_apps() {
    run_event_publisher $1;
    sleep 1;

    # run_apps_repeatedly & sleep $total_time ; kill $!;
    run_apps_repeatedly; kill $!;

    kill_edge_image;
    sleep 2;
}

function run_apps_repeatedly() {
    # while true; do
    for i in $(seq 0 1 5); do
        run_apps_once;
    done;
}

function run_apps_once() {
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

sudo cgdelete -r cpuset:mycontroller
start_roscore;

cxs=$(seq 0 1 2)
for cx in $cxs; do
    echo "Running complexity $cx ..."
    run_scenario_pdqn $cx;
    # run_scenario_search_seq $cx;
    # run_scenario_so_seq $cx;
    # run_scenario_avg_speed $cx;

    # run_scenario_sac $cx;
    # run_scenario_max_speed $cx;
    # run_scenario_min_speed $cx;
    # run_scenario_hc $cx;

    cx_log_dir="$log_dir/$cx";
    mkdir -p $cx_log_dir;
    mv -f "$log_dir"/*.csv "$log_dir"/*.log "$log_dir"/*.txt "$cx_log_dir";
done

exit 0
