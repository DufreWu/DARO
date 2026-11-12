#!/usr/bin/env bash
# Train all DARO variants (DARO + ablations) in parallel, one background job per variant.
#
#   ./train_all.sh                 # seeds 0 1 2
#   SEEDS="0 1 2 3 4" ./train_all.sh
#   EXTRA="--steps 200000" ./train_all.sh
#
# Each variant trains its seeds in parallel (--workers = number of seeds),
# so total processes = 5 x number of seeds. Console output of each variant goes
# to logs/train_all/<name>.log; per-seed output to logs/<variant>/seed_<s>/train_stdout.log.
# Ctrl+C stops all jobs.

set -u
cd "$(dirname "$0")"

SEEDS=${SEEDS:-"0 1 2 3 4"}
EXTRA=${EXTRA:-}
NSEEDS=$(wc -w <<< "$SEEDS")
OUT=logs/train_all
mkdir -p "$OUT"

# name | extra args
VARIANTS=(
  "daro|"
  "rw_no_stress|--reward no_stress"
  "rl_soc|--obs soc"
  "rl_soh|--obs soh"
  "rl_soc_rw_no_stress|--obs soc --reward no_stress"
)

pids=()
names=()
trap 'echo; echo "Stopping all jobs..."; kill 0 2>/dev/null; exit 130' INT TERM

for v in "${VARIANTS[@]}"; do
  name=${v%%|*}
  args=${v#*|}
  # shellcheck disable=SC2086
  python sac_training.py --seeds $SEEDS --workers "$NSEEDS" $args $EXTRA \
    > "$OUT/$name.log" 2>&1 &
  pids+=($!)
  names+=("$name")
  echo "started $name (pid $!) -> $OUT/$name.log"
done

echo "Running ${#pids[@]} variants x $NSEEDS seeds. Progress: tail -f logs/<variant>/seed_<s>/train_stdout.log"

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[done] ${names[$i]}"
  else
    echo "[FAILED] ${names[$i]} (see $OUT/${names[$i]}.log)"
    fail=1
  fi
done
exit $fail
