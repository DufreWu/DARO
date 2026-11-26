#!/usr/bin/env bash
# DARO HIL: start the PC nodes (event camera + mission runner) against a running
# Isaac Sim server and TX2 agent. Extra arguments go to mission_node.py.
#   TX2_HOST=192.168.1.20 bash pc/launch_pc_nodes.sh --policies BA-EPD Lag-RLS \
#       --seeds 0 1 2 --distance 1000 --soh 1.0 --c-label 2 --align --out ../logs/hil/nominal_c2
set -eo pipefail
ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
TX2_HOST="${TX2_HOST:-192.168.1.20}"
PC_PYTHON="${PC_PYTHON:-python3}"
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${PC_PYTHON}" "${DIR}/event_camera_node.py" --tx2-host "${TX2_HOST}" --match-table &
EVENT_PID=$!
trap 'kill ${EVENT_PID} 2>/dev/null || true' EXIT
"${PC_PYTHON}" "${DIR}/mission_node.py" --tx2-host "${TX2_HOST}" "$@"
