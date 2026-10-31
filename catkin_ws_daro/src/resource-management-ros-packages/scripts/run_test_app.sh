#!/bin/bash
source /opt/ros/melodic/setup.bash

log_dir="./temp/exp-9999-data-test"
speed=0.5
# 345600 499200 652800 806400 960000 1113600 1267200 1420800 1574400 1728000 1881600 2035200 
# 114750 216750 318750 420750 522750 624750 726750 854250 930750 1032750 1122000 1236750 1300500
freq_l=499200
freq_b=499200
freq_g=216750
sleep_time=30
total_time=300
all_test_time=7500
# num_ev_base=10000000
num_ev_base=30000000
cxs=$(seq 0 1 3) 
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

#######################################
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

######################################
function kill_all_screens() {
    for session in $(screen -ls | grep -o '[0-9]*\.chen_[[:alnum:]]*' | grep -v chen_core); do 
        echo "Killing screen session: $session"
        screen -S "${session}" -X quit;
    done
}

function mv_result_files() { # args: (name)
    name=$1
    mv -f $log_dir/{sys,${name}_sys}.csv;
    mv -f $log_dir/{appls,${name}_appls}.csv;
    mv -f $log_dir/{fa_harris_stat,${name}_fa_harris_stat}.csv;
    mv -f $log_dir/{event_counter,${name}_event_counter}.csv;
    mv -f $log_dir/{model,${name}_model}.csv;
    mv -f $log_dir/{hc,${name}_hc}.csv;
    mv -f $log_dir/{latest_model,${name}_latest_model}.csv;
}

function run_apps() {
    launch_event_publisher $num_ev_base;
    sleep 1;

    # launch_edge_image;
    # sleep 1000;
    # kill_edge_image;

    launch_edge_image;
    sleep 1000;
    # launch_denoising_filling;
    # sleep 1000;
    # kill_denoising_filling;

    # launch_edge_image;
    # sleep 1;
    # launch_distance_surface;
    # sleep 1000;
    # kill_distance_surface;

    # launch_edge_image;
    # sleep 1;
    # launch_optical_flow $log_dir;
    # sleep 10;
    # kill_optical_flow;

    # launch_tracker;
    # sleep 1000;
    # kill_tracker;

    # sleep $total_time;
    # sleep $all_test_time;
}

function run_scenario_max_speed() {
    echo "############# MAX #############"
    name="constmax"; speed=5; freq_b=2035200, freq_l=2035200, freq_g=216750;
    launch_const_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_min_speed() {
    echo "############# MIN #############"
    name="constmin"; speed=0.5; freq_b=2035200, freq_l=2035200, freq_g=216750;
    launch_const_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

# 345600 499200 652800 806400 960000 1113600 1267200 1420800 1574400 1728000 1881600 2035200 
# 114750 216750 318750 420750 522750 624750 726750 854250 930750 1032750 1122000 1236750 1300500
function run_scenario_avg_speed() {
    echo "############# AVG #############"
    name="constavg"; speed=2.5; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_const_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;    
}

function run_scenario_search_seq() {
    echo "############# SEARCH #############"
    name="search"; speed=0.5; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_search_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_so_seq() {
    echo "############# SO #############"
    name="so"; speed=2; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_const_speed_interactive_freq_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_allconst() {
    echo "############# ALL CONST #############"
    name="allconst"; speed=1.0; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_all_consts_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_pdqn() {
    echo "############# PDQN #############"
    name="pdqn"; speed=0.5; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_pdqn_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_sac() {
    echo "############# SAC #############"
    name="sac"; speed=0.5; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_sac_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

#####

sudo cgdelete -r cpuset:mycontroller
start_roscore;

# run_scenario_sac;
# run_scenario_pdqn
# run_scenario_allconst;
# run_scenario_hc;
run_scenario_so_seq;
# run_scenario_search_seq;
# run_scenario_avg_speed;
# run_scenario_max_speed;
# run_scenario_min_speed;
exit 0

start_roscore;

echo "#############"
launch_controller;
sleep 1;
launch_event_publisher $num_ev_base;
launch_optical_flow
# launch_eklt;
# launch_img_reconstruction;
# launch_fa_harris;
sleep 60;
kill_all_screens;
exit 0;

echo "#############"
launch_controller;
sleep $sleep_time;
kill_all_screens;
mv $log_dir/{sys,sys_idle}.csv 


echo "#############"
launch_controller;
launch_event_publisher $num_ev_base;
sleep $sleep_time;
kill_all_screens;
mv $log_dir/{sys,sys_ev_pub}.csv;

echo "#############"
launch_controller;
launch_event_publisher $num_ev_base;
launch_fa_harris;
sleep $sleep_time;
kill_all_screens;
mv $log_dir/{sys,sys_app}.csv;
mv $log_dir/{appls,appls_app}.csv;
mv $log_dir/{fa_harris_stat,fa_harris_stat_app}.csv;


echo "#############"
launch_controller;
launch_event_publisher $(( $num_ev_base * 10 ));
sleep $sleep_time;
kill_all_screens;
mv $log_dir/{sys,sys_ev_pub_heavy}.csv;

echo "#############"
launch_controller;
launch_event_publisher $(( $num_ev_base * 10 ));
launch_fa_harris;
sleep $sleep_time;
kill_all_screens;
mv $log_dir/{sys,sys_app_heavy}.csv;
mv $log_dir/{appls,appls_app_heavy}.csv;
mv $log_dir/{fa_harris_stat,fa_harris_stat_app_heavy}.csv;
