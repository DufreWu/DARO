#!/bin/bash

# enum {
#   NO_POLICY = 0,
#   ONLY_SENSORS = 1,
#   ODROID_POLICY = 2,
#   ODROID_LOG_POLICY = 3,
#   ODROID_PROF_POLICY = 4,
#   TEGRA_POLICY = 5,
#   TEGRA_LOG_POLICY = 6,
#   RPI3_POLICY = 11,
#   TEGRA_SPEED_LOG_POLICY = 13,
#   TEGRA_SPEED_HILLCLIMBING_LOG_POLICY = 14,
#   TEGRA_SPEED_CONSTANT_LOG_POLICY = 15,
#   TEGRA_SPEED_MODELBASED_HILLCLIMBING_LOG_POLICY = 16,
#   TEGRA_SPEED_MODELBASED_LIN3_HILLCLIMBING_LOG_POLICY = 17,
#   TEGRA_SPEED_CONSTANT_FREQ_INTERACTIVE=18,
#   TEGRA_SPEED_PREDICTIVE_SEARCH_LOG_POLICY = 19,
#   TEGRA_SPEED_ALL_CONSTS_POLICY_LOG_POLICY = 20,
#   TEGRA_SPEED_PDQN_POLICY=22,
#   TEGRA_SPEED_SAC_POLICY=23,
#   TEGRA_SPEED_DQN_POLICY=24,
#   TEGRA_SPEED_PPO_POLICY=25,
#   MAX_POLICY = 27,
# };


#######################################
function launch_const_controller() { # args: ($log_dir, $speed, $freq_l, $freq_b, $freq_g)
    # rosrun --prefix 'sudo -E' --debug chen_resource_controller controller -p 15 -s $2 -fl $3 --fb $3 --fg $4 --log-dir $1 | tee $1/log_controller.log; 
    echo "Launching SPEED_CONSTANT: log_dir: $1, speed: $2, freq_l: $3, freq_b: $4, freq_g: $5"
    command="
    source ../../devel/setup.bash; 
    rosrun chen_resource_controller controller -p 15 -s $2 --fb $3 --fl $4 --fg $5 --log-dir $1 | tee $1/controller.log; 
    exec sh;
    "
    screen -S chen_controller -dm bash -c "$command"
    echo "SPEED_CONSTANT launched!"
}

#######################################
function launch_all_consts_controller() { # args: ($log_dir, $speed, $freq_l, $freq_b, $freq_g)
    # rosrun --prefix 'sudo -E' --debug chen_resource_controller controller -p 20 -s $2 -f $3 --fb $3 --log-dir $1 | tee $1/log_controller.log; 
    echo "Launching ALL_CONSTS: log_dir: $1, speed: $2, freq_l: $3, freq_b: $4, freq_g: $5"
    echo "log_dir: $1, speed: $2, freq_l: $3, freq_b: $4, freq_g: $5"
    command="
    source ../../devel/setup.bash;
    mkdir -p $1;
    rosrun chen_resource_controller controller -p 20 -s $2 --fb $3 --fl $4 --fg $5 --log-dir $1 | tee $1/controller.log; 
    exec sh;
    "
    screen -S chen_controller -dm bash -c "$command"
    echo "ALL_CONSTS launched!"
}

#######################################
function launch_const_speed_interactive_freq_controller() { # args: ($log_dir, $speed, $freq_l, $freq_b, $freq_g)
    # rosrun --prefix 'sudo -E' --debug chen_resource_controller controller -p 18 -s $2 --f $3 --fb $3 --log-dir $1 | tee $1/log_controller.log; 
    echo "Launching SPEED_CONSTANT_FREQ_INTERACTIVE: log_dir: $1, speed: $2, freq_l: $3, freq_b: $4, freq_g: $5"
    command="
    source ../../devel/setup.bash; 
    rosrun chen_resource_controller controller -p 18 -s $2 --fb $3 --fl $4 --fg $5 --log-dir $1 | tee $1/controller.log; 
    exec sh;
    "
    screen -S chen_controller -dm bash -c "$command"
    echo "SPEED_CONSTANT_FREQ_INTERACTIVE launched!"
}

#######################################
function launch_tegra_speed_controller() { # args: ($log_dir, $speed, $freq_l, $freq_b, $freq_g)
    # rosrun --prefix 'sudo -E' --debug chen_resource_controller controller -p 13 -s $2 -f $3 --fb $3 --log-dir $1 | tee $1/log_controller.log; 
    echo "Launching SPEED_LOG: log_dir: $1, speed: $2, freq_l: $3, freq_b: $4, freq_g: $5"
    echo "log_dir: $1, speed: $2, freq_l: $3, freq_b: $4, freq_g: $5"
    command="
    source ../../devel/setup.bash; 
    rosrun chen_resource_controller controller -p 13 -s $2 --fb $3 --fl $4 --fg $5 --log-dir $1 | tee $1/controller.log; 
    exec sh;
    "
    screen -S chen_controller -dm bash -c "$command"
    echo "SPEED_LOG launched!"
}

#######################################
function launch_hc_controller() { # args: ($log_dir, $speed, $freq_l, $freq_b, $freq_g)
    # rosrun --prefix 'sudo -E' --debug chen_resource_controller controller -p 14 -s $2 -f $3 --fb $3 --log-dir $1 | tee $1/log_controller.log; 
    echo "Launching HILLCLIMBING_LOG_POLICY: log_dir: $1, speed: $2, freq_l: $3, freq_b: $4, freq_g: $5"
    command="
    source ../../devel/setup.bash; 
    rosrun chen_resource_controller controller -p 14 -s $2 --fb $3 --fl $4 --fg $5 --log-dir $1 | tee $1/controller.log; 
    exec sh;
    "
    screen -S chen_controller -dm bash -c "$command"
    echo "HILLCLIMBING_LOG_POLICY launched!"
}

#######################################
function launch_search_controller() { # args: ($log_dir, $speed, $freq_l, $freq_b, $freq_g)
    # rosrun --prefix 'sudo -E' --debug chen_resource_controller controller -p 19 -s $2 -f $3 --fb $3 --log-dir $1 --model-dir $model_dir | tee $1/log_controller.log; 
    echo "Launching PREDICTIVE_SEARCH: log_dir: $1, speed: $2, freq_l: $3, freq_b: $4, freq_g: $5"
    model_dir="/home/chen2/catkin_ws_chen/src/resource-management-ros-packages/chen_resource_controller/models"
    command="
    source ../../devel/setup.bash; 
    rosrun chen_resource_controller controller -p 19 --model-dir $model_dir -s $2 --fb $3 --fl $4 --fg $5 --log-dir $1 | tee $1/controller.log; 
    exec sh;
    "
    screen -S chen_controller -dm bash -c "$command"
    echo "PREDICTIVE_SEARCH launched!"
}

#######################################
function launch_pdqn_controller() { # args: ($log_dir, $speed, $freq_l, $freq_b, $freq_g)
    # rosrun --prefix 'sudo -E' --debug chen_resource_controller controller -p 19 -s $2 -f $3 --fb $3 --log-dir $1 --model-dir $model_dir | tee $1/log_controller.log; 
    echo "Launching PDQN_POLICY: log_dir: $1, speed: $2, freq_l: $3, freq_b: $4, freq_g: $5"
    model_dir="/home/chen2/catkin_ws_chen/src/resource-management-ros-packages/chen_resource_controller/models"
    command="
    source ../../devel/setup.bash; 
    rosrun chen_resource_controller controller -p 22 --model-dir $model_dir -s $2 --fb $3 --fl $4 --fg $5 --log-dir $1 | tee $1/controller.log; 
    exec sh;
    "
    screen -S chen_controller -dm bash -c "$command"
    echo "PDQN_POLICY launched!"
}

#######################################
function launch_sac_controller() { # args: ($log_dir, $speed, $freq_l, $freq_b, $freq_g)
    # rosrun --prefix 'sudo -E' --debug chen_resource_controller controller -p 19 -s $2 -f $3 --fb $3 --log-dir $1 --model-dir $model_dir | tee $1/log_controller.log; 
    echo "Launching SAC_POLICY: log_dir: $1, speed: $2, freq_l: $3, freq_b: $4, freq_g: $5"
    model_dir="/home/chen2/catkin_ws_chen/src/resource-management-ros-packages/chen_resource_controller/models"
    command="
    source ../../devel/setup.bash; 
    rosrun chen_resource_controller controller -p 23 --model-dir $model_dir -s $2 --fb $3 --fl $4 --fg $5 --log-dir $1 | tee $1/controller.log; 
    exec sh;
    "
    screen -S chen_controller -dm bash -c "$command"
    echo "SAC_POLICY launched!"
}

function launch_dqn_controller() { # args: ($log_dir, $speed, $freq_l, $freq_b, $freq_g)
    # rosrun --prefix 'sudo -E' --debug chen_resource_controller controller -p 19 -s $2 -f $3 --fb $3 --log-dir $1 --model-dir $model_dir | tee $1/log_controller.log; 
    echo "Launching DQN_POLICY: log_dir: $1, speed: $2, freq_l: $3, freq_b: $4, freq_g: $5"
    model_dir="/home/chen2/catkin_ws_chen/src/resource-management-ros-packages/chen_resource_controller/models"
    command="
    source ../../devel/setup.bash; 
    rosrun chen_resource_controller controller -p 24 --model-dir $model_dir -s $2 --fb $3 --fl $4 --fg $5 --log-dir $1 | tee $1/controller.log; 
    exec sh;
    "
    screen -S chen_controller -dm bash -c "$command"
    echo "DQN_POLICY launched!"
}

function launch_ppo_controller() { # args: ($log_dir, $speed, $freq_l, $freq_b, $freq_g)
    # rosrun --prefix 'sudo -E' --debug chen_resource_controller controller -p 19 -s $2 -f $3 --fb $3 --log-dir $1 --model-dir $model_dir | tee $1/log_controller.log; 
    echo "Launching PPO_POLICY: log_dir: $1, speed: $2, freq_l: $3, freq_b: $4, freq_g: $5"
    model_dir="/home/chen2/catkin_ws_chen/src/resource-management-ros-packages/chen_resource_controller/models"
    command="
    source ../../devel/setup.bash; 
    rosrun chen_resource_controller controller -p 25 --model-dir $model_dir -s $2 --fb $3 --fl $4 --fg $5 --log-dir $1 | tee $1/controller.log; 
    exec sh;
    "
    screen -S chen_controller -dm bash -c "$command"
    echo "PPO_POLICY launched!"
}

######################################
function set_complexity() { # args (complexity)
    cx=$1
    ncx=$(( (cx+1)%3 ));
    echo $ncx > ../../../chen_resource_controller_complexity.txt;
    sleep 1;
    echo $cx > ../../../chen_resource_controller_complexity.txt; 
}

function find_avg_time() {
    cat temp/exp-scenarios-14-search-2/0/log_controller.log | \
        grep "CYCLE-TIME (TOTAL)" | \
        cut -d " " -f 3 | \
        awk '{ total += $1; count++ } END { print total/count }'
}
