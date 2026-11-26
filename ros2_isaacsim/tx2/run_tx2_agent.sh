#!/usr/bin/env bash
# DARO HIL: start the TX2 agent (on the Jetson TX2, as root for DVFS).
# Replace the placeholder applications with APP-DT / APP-OP (see apps/reference_apps.py):
#   sudo APPS="APP-DT=app_dt:AppDT APP-OP=app_op:AppOP" bash tx2/run_tx2_agent.sh --verbose
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APPS="${APPS:-APP-DT=reference_apps:RefDetect APP-OP=reference_apps:RefFlow}"
APP_ARGS=()
for a in ${APPS}; do APP_ARGS+=(--app "$a"); done
# Fixed fan speed avoids thermal throttling changing the throughput during a run.
if [[ -w /sys/devices/pwm-fan/target_pwm ]]; then echo 255 > /sys/devices/pwm-fan/target_pwm; fi
exec python3 "${DIR}/daro_tx2_agent.py" "${APP_ARGS[@]}" "$@"
