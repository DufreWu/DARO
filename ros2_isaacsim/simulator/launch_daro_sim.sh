#!/usr/bin/env bash
# DARO HIL: start the Isaac Sim server (PC).
#   bash simulator/launch_daro_sim.sh medium                 # one complexity on the whole route
#   bash simulator/launch_daro_sim.sh zones --zone-order low,high,medium,high
#   bash simulator/launch_daro_sim.sh high --headless --route-scale 3
set -eo pipefail

COMPLEXITY="medium"
if [[ $# -gt 0 && "$1" != -* ]]; then
    COMPLEXITY="$1"
    shift
fi
case "${COMPLEXITY}" in
    low|medium|high|zones) ;;
    *) echo "complexity must be low, medium, high or zones" >&2; exit 2 ;;
esac

ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
[[ -f "${ROS_SETUP}" ]] || { echo "ROS 2 setup not found: ${ROS_SETUP}" >&2; exit 1; }
# shellcheck disable=SC1090
source "${ROS_SETUP}"
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export LD_LIBRARY_PATH="/opt/ros/${ROS_DISTRO_NAME}/lib:/opt/ros/${ROS_DISTRO_NAME}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-python}"

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}  complexity=${COMPLEXITY}"
exec "${ISAAC_PYTHON}" "${SCRIPT_DIR}/daro_sim.py" \
    --complexity "${COMPLEXITY}" \
    --control-mode hybrid \
    --disable-depth \
    "$@"
