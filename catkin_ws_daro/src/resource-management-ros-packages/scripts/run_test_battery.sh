#!/bin/bash
source /opt/ros/melodic/setup.bash

log_dir="./temp/exp-9999-battery-soh-1.0"
speed=0.5
# 345600 499200 652800 806400 960000 1113600 1267200 1420800 1574400 1728000 1881600 2035200 
# 114750 216750 318750 420750 522750 624750 726750 854250 930750 1032750 1122000 1236750 1300500
freq_l=499200
freq_b=499200
freq_g=216750
sleep_time=30
total_time=120
all_test_time=1500
num_ev_base=10000000
cxs=$(seq 0 1 2) 
# print the command
echo "$0 $@"

src_dir=$( dirname -- "$0")
. $src_dir/parse_options.sh
. $src_dir/util_funcs.sh

trap 'kill_all_screens; exit 0;' SIGTERM SIGINT EXIT

mkdir -p $log_dir;

log_dir=$(realpath $log_dir);

echo "Logging in $log_dir"

#######################################
function start_roscore() {
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
#######################################
# rosrun --prefix 'sudo -E' --debug chen_resource_controller controller -p 15 -s $speed -f $freq --log-dir $log_dir | tee $log_dir/controller.log; 
function launch_controller() {
    echo "Launching controller.."
    command="
    source ../../devel/setup.bash; 
    # rosrun chen_resource_controller controller -p 15 -s $speed --fb $freq_b --fl $freq_l --fg $freq_g --log-dir $log_dir | tee $log_dir/controller.log; 
    rosrun chen_resource_controller controller -p 15 -s $speed --fb $freq_b --fl $freq_l --fg $freq_g --log-dir $log_dir 
    exec sh;
    "
    screen -S chen_controller -dm bash -c "$command"
    echo "controller launched!"
}

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
function launch_optical_flow() {
    echo "Launching optical_flow.."
    command="
        source ../../devel/setup.bash; 
        roslaunch rt_of_low_high_res_event_cameras davis240_live.launch > $log_dir/optical_flow.log 2>&1;
        exec sh;
    "
    screen -S chen_optical_flow -dm bash -c "$command"
    echo "optical_flow launched!"
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

function run_event_publisher() {
    complexity=$1
    echo "Complexity is: $complexity"
    # launch_event_counter;
    if [[ "$complexity" == "0" ]]; then
        num_ev_base=10000000
        launch_event_publisher $num_ev_base
    elif [[ "$complexity" == "1" ]]; then
        num_ev_base=30000000
        launch_event_publisher $num_ev_base
    elif [[ "$complexity" == "2" ]]; then
        num_ev_base=50000000
        launch_event_publisher $num_ev_base
    else 
        launch_event_publisher $num_ev_base
    fi
}

function run_apps() {
    run_event_publisher $1;

    launch_fa_harris;
    sleep 20;
    
    run_apps_repeatedly & sleep $total_time ; kill $!;

    rosnode kill "fa_harris";
    sleep 1;
    screen -S "chen_fa_harris" -X quit; 
    echo "fa_harris killed!" 
}

function run_apps_repeatedly() {
    while true; do
        run_apps_once;
    done;
}

function run_apps_once() {
    launch_img_reconstruction
    sleep 20;
    # launch_edge_image;
    # sleep 20;

    rosnode kill "reconstruction_node";
    sleep 1;
    screen -S "chen_img_reconstruction" -X quit; 
    echo "chen_img_reconstruction killed!"

    # rosnode kill "rt_of_low_high_res_event_cameras/edges";
    # sleep 1;
    # screen -S "chen_edge_image" -X quit; 
    # echo "chen_edge_image killed!"

    # sleep 20;
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

function run_scenario_avg_speed() {
    echo "############# AVG #############"
    name="constavg"; speed=2.5; freq_b=2035200, freq_l=2035200, freq_g=216750;
    launch_const_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;    
}

function run_scenario_hc() {
    echo "############# HC #############"
    name="hc"; speed=0.5; freq_b=2035200, freq_l=2035200, freq_g=522750;
    launch_hc_controller $log_dir $speed $freq_b $freq_l $freq_g;
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

function run_scenario_mrl_seq() {
    echo "############# MRL #############"
    name="mrl"; speed=2.0; freq_b=499200, freq_l=499200, freq_g=216750; 
    launch_mapping_rl_controller $log_dir $speed $freq_b $freq_l $freq_g;
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
    name="pdqn"; speed=0.25; freq_b=499200, freq_l=499200, freq_g=216750;
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

function run_scenario_dqn() {
    echo "############# DQN #############"
    name="dqn"; speed=0.5; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_dqn_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_ppo() {
    echo "############# PPO #############"
    name="ppo"; speed=0.5; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_ppo_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_drqn() {
    echo "############# DRQN #############"
    name="drqn"; speed=0.5; freq_b=499200, freq_l=499200, freq_g=216750;
    launch_drqn_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 2;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

#####

start_roscore;

echo $cxs;
for cx in $cxs; do
    echo "Running complexity $cx ..."
    # run_scenario_ppo;
    # run_scenario_pdqn;
    # run_scenario_so_seq;
    # run_scenario_dqn;
    # run_scenario_search_seq;
    run_scenario_sac;
    run_scenario_avg_speed;
    # run_scenario_drqn;
    # run_scenario_hc;
    # run_scenario_max_speed;
    # run_scenario_min_speed;
    # run_scenario_allconst;
    cx_log_dir="$log_dir/$cx";
    mkdir -p $cx_log_dir;
    mv -f "$log_dir"/*.csv "$log_dir"/*.log "$log_dir"/*.txt "$cx_log_dir";
done
exit 0
