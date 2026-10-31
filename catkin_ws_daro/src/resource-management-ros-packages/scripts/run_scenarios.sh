#!/bin/bash

source /opt/ros/melodic/setup.bash

log_dir="./temp/exp-9999-test"
sleep_time=120
num_ev_base=10000000
cxs=$(seq 0 1 3) 

# print the command
echo "$0 $@"

src_dir=$( dirname -- "$0")
. $src_dir/parse_options.sh
. $src_dir/util_funcs.sh

trap 'kill_all_screens; exit 0;' SIGTERM SIGINT EXIT

curr_time=`date +'%Y%m%d-%H%M%S'`;
# log_dir="$log_dir-$curr_time"
# rm -rf $log_dir;
mkdir -p $log_dir;
log_dir=$(realpath $log_dir);
echo "Logging in $log_dir"


######################################
######################################

start_roscore $log_dir;

function run_scenario_hc() {
    echo "############# HC #############"
    name="hc"; speed=0.5; freq=2035200;
    launch_hc_controller $log_dir $speed $freq;
    # launch_event_counter;
    launch_fa_harris $log_dir;
    sleep $sleep_time;
    kill_all_screens;
    mv  $log_dir/{sys,${name}_sys}.csv;
    mv  $log_dir/{appls,${name}_appls}.csv;
    mv $log_dir/{fa_harris_stat,${name}_fa_harris_stat}.csv;
    mv $log_dir/{hc,${name}_hc}.csv;
    sleep 2;
}

function run_scenario_max_speed() {
    echo "############# MAX #############"
    name="constmax"; speed=5; freq=2035200; 
    launch_const_controller $log_dir $speed $freq;
    # launch_tegra_speed_controller $log_dir $speed $freq;
    # launch_event_counter;
    launch_fa_harris $log_dir;
    sleep 120;
    kill_all_screens;
    mv  $log_dir/{sys,${name}_sys}.csv;
    mv  $log_dir/{appls,${name}_appls}.csv;
    mv $log_dir/{fa_harris_stat,${name}_fa_harris_stat}.csv;
    sleep 2;
}
function run_scenario_min_speed() {
    echo "############# MIN #############"
    name="constmin"; speed=0.5; freq=499200; 
    launch_const_controller $log_dir $speed $freq;
    # launch_tegra_speed_controller $log_dir $speed $freq;
    # launch_event_counter;
    launch_fa_harris $log_dir;
    sleep 120;
    kill_all_screens;
    mv $log_dir/{sys,${name}_sys}.csv;
    mv $log_dir/{appls,${name}_appls}.csv;
    mv $log_dir/{fa_harris_stat,${name}_fa_harris_stat}.csv;
    sleep 2;
}
function run_scenario_avg_speed() {
    echo "############# AVG #############"
    name="constavg"; speed=2.5; freq=1420800; 
    launch_const_controller $log_dir $speed $freq;
    # launch_tegra_speed_controller $log_dir $speed $freq;
    # launch_event_counter;
    launch_fa_harris $log_dir;
    sleep 120;
    kill_all_screens;
    mv $log_dir/{sys,${name}_sys}.csv;
    mv $log_dir/{appls,${name}_appls}.csv;
    mv $log_dir/{fa_harris_stat,${name}_fa_harris_stat}.csv;
}

function run_scenario_search() {
    echo "############# SEARCH #############"
    name="search"; speed=0.5; freq=2035200; 
    launch_search_controller $log_dir $speed $freq;
    # launch_event_counter;
    launch_fa_harris $log_dir;
    sleep $sleep_time;
    kill_all_screens;
    mv $log_dir/{sys,${name}_sys}.csv;
    mv $log_dir/{appls,${name}_appls}.csv;
    mv $log_dir/{fa_harris_stat,${name}_fa_harris_stat}.csv;
    mv $log_dir/{model,${name}_model}.csv;
    mv $log_dir/{hc,${name}_hc}.csv;
    mv $log_dir/{latest_model,${name}_latest_model}.csv;
    sleep 2;
}

#####

run_scenario_hc;

echo $cxs;
for cx in $cxs; do
    echo "Running complexity $cx ..."
    set_complexity $cx;
    run_scenario_hc;
    # run_scenario_search;
    # run_scenario_max_speed;
    # run_scenario_avg_speed;
    # run_scenario_min_speed;
    

    cx_log_dir="$log_dir/$cx";
    mkdir -p $cx_log_dir;
    mv "$log_dir"/*.csv "$log_dir"/*.log "$cx_log_dir";
done

exit 0;

# echo "#############"
# name="constsp-all-speeds"; init_speed=1; freq=499200;
# launch_tegra_speed_controller $log_dir $init_speed $freq;
# launch_event_counter;
# launch_fa_harris $log_dir;

# sleep $sleep_time;
# kill_all_screens;
# mv  $log_dir/{sys,${name}_sys}.csv;
# mv  $log_dir/{appls,${name}_appls}.csv;
# mv $log_dir/{fa_harris_stat,${name}_fa_harris_stat}.csv;
# sleep 2;

# exit 0

# ######################################
# ######################################

# start_roscore $log_dir;
# speeds=$(seq 0.5 0.25 5) # "2.5 4.0 4.5 5.0" #
# for speed in $speeds; do
#     echo "#############"
#     echo $speed
#     name="constsp-$speed"; freq=499200; 
#     launch_tegra_speed_controller $log_dir $speed $freq;
#     launch_event_counter;
#     launch_fa_harris $log_dir;

#     sleep $sleep_time;
#     kill_all_screens;
#     mv  $log_dir/{sys,${name}_sys}.csv;
#     mv  $log_dir/{appls,${name}_appls}.csv;
#     mv $log_dir/{fa_harris_stat,${name}_fa_harris_stat}.csv;
#     sleep 2;
# done

# exit 0