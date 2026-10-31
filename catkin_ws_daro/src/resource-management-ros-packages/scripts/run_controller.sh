#!/bin/bash

# --- Configuration ---
ROS_SETUP="/opt/ros/melodic/setup.bash"
WS_SETUP="$HOME/catkin_ws_chen/devel/setup.bash"
LOG_DIR="./temp/exp-9999-data-test/edge_image"
CG_NAME="mycontroller"

# --- Environment Setup ---
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost
mkdir -p "$LOG_DIR"

echo "--- Initializing Environment ---"

# 1. Force move any PIDs out of existing mycontroller subgroups to the root
# This is the "Magic Fix" for the 'Device or resource busy' error
if [ -d "/sys/fs/cgroup/cpuset/$CG_NAME" ]; then
    echo "Moving orphaned processes to root cgroup..."
    find "/sys/fs/cgroup/cpuset/$CG_NAME" -name tasks -exec sh -c '
        for pid in $(cat "$1"); do 
            echo "$pid" | sudo tee /sys/fs/cgroup/cpuset/tasks > /dev/null 2>&1
        done' -- {} \;
fi

# 2. Delete existing groups across all relevant controllers
echo "Cleaning up CGroup hierarchies..."
sudo cgdelete -r "cpuset,cpu,freezer:/$CG_NAME" 2>/dev/null

# # 3. Check if roscore is running
# if ! rostopic list >/dev/null 2>&1; then
#     echo "ERROR: roscore not detected. Please start roscore in another terminal."
#     exit 1
# fi

# 4. Run Controller with sudo -E (Preserve Environment)
echo "Starting Resource Controller..."
sudo -E bash -c "
    source $ROS_SETUP
    source $WS_SETUP
    rosrun chen_resource_controller controller \
    -p 13 \
    --model-dir \"/home/chen2/catkin_ws_chen/src/resource-management-ros-packages/chen_resource_controller/models\" \
    -s 1.0 --fb 499200 --fl 499200 --fg 216750 \
    --log-dir \"$LOG_DIR\"
" | tee "$LOG_DIR/controller.log"