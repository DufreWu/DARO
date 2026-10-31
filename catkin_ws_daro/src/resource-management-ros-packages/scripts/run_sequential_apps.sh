#!/bin/bash

source /opt/ros/melodic/setup.bash

log_dir="./temp/exp-9999-20250615"
total_time=600
sleep_time=30
num_ev_base=10000000
cxs=$(seq 0 1 3) 
comment=
# print the command
echo "$0 $@"

src_dir=$( dirname -- "$0")
. $src_dir/parse_options.sh
. $src_dir/util_funcs.sh

trap 'kill_all_screens; exit 0;' SIGTERM SIGINT EXIT

curr_time=`date +'%Y%m%d-%H%M%S'`;
log_dir=$(realpath $log_dir);
mkdir -p $log_dir;
echo "Logging in $log_dir"

echo $comment >> $log_dir/exp_comment.info

######################################
######################################

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

# function launch_fa_harris_custom() { # args: (log_dir, launch_file_name)
#     echo "Launching fa_harris.."
#     command="
#         source ../../devel/setup.bash; 
#         roslaunch fa_harris $2 > $1/log_$2.log 2>&1;
#         exec sh;
#     "
#     screen -S chen_$2 -dm bash -c "$command"
#     echo "$2 launched!"
# }

function run_apps() {
    # launch_event_counter;

    launch_fa_harris $log_dir;
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
    # launch_img_reconstruction $log_dir;
    # sleep $sleep_time;

    launch_optical_flow $log_dir;
    sleep 2;

    launch_eklt $log_dir;
    sleep 12;

    # rosnode kill "eklt_node";
    sleep 1;
    screen -S "chen_eklt" -X quit; 
    echo "chen_eklt killed!"
    sleep 13;

    launch_eklt $log_dir;
    sleep 12;

    # rosnode kill "eklt_node";
    sleep 1;
    screen -S "chen_eklt" -X quit; 
    echo "chen_eklt killed!"
    sleep 10;

    # rosnode kill "reconstruction_node";
    # sleep 1;
    # screen -S "chen_img_reconstruction" -X quit; 
    # echo "chen_img_reconstruction killed!"
    # sleep $sleep_time;

    # rosnode kill "rt_of_low_high_res_event_cameras/denoising_filling";
    # rosnode kill "rt_of_low_high_res_event_cameras/distance_surface";
    # rosnode kill "rt_of_low_high_res_event_cameras/edges";
    sleep 1;
    screen -S "chen_optical_flow" -X quit; 
    echo "chen_optical_flow killed!"
    sleep $sleep_time;
}

function run_scenario_max_speed() {
    echo "############# MAX #############"
    name="constmax"; speed=5; freq_b=2035200, freq_l=2035200, freq_g=216750000;
    launch_const_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 1;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_min_speed() {
    echo "############# MIN #############"
    name="constmin"; speed=0.5; freq_b=2035200, freq_l=2035200, freq_g=216750000;
    launch_const_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 1;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_avg_speed() {
    echo "############# AVG #############"
    name="constavg"; speed=2.5; freq_b=2035200, freq_l=2035200, freq_g=216750000;
    launch_const_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 1;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;    
}

function run_scenario_hc() {
    echo "############# HC #############"
    name="hc"; speed=0.5; freq_b=2035200, freq_l=2035200, freq_g=216750000;
    launch_hc_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 1;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_search_seq() {
    echo "############# SEARCH #############"
    name="search"; speed=0.5; freq_b=499200, freq_l=499200, freq_g=216750000;
    launch_search_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 1;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

function run_scenario_so_seq() {
    echo "############# SO #############"
    name="so"; speed=2; freq_b=499200, freq_l=499200, freq_g=216750000; 
    launch_const_speed_interactive_freq_controller $log_dir $speed $freq_b $freq_l $freq_g;
    sleep 1;
    run_apps;
    kill_all_screens;
    mv_result_files $name;
    sleep 2;
}

#####

start_roscore $log_dir;

echo $cxs;
for cx in $cxs; do
    echo "Running complexity $cx ..."
    set_complexity $cx;
    # run_scenario_avg_speed;
    # run_scenario_hc
    # run_scenario_search_seq;
    run_scenario_so_seq;
    # run_scenario_max_speed;
    # run_scenario_min_speed;

    cx_log_dir="$log_dir/$cx";
    mkdir -p $cx_log_dir;
    mv -f "$log_dir"/*.csv "$log_dir"/*.log "$log_dir"/*.txt "$cx_log_dir";
done

exit 0;
