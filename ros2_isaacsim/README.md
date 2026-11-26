# DARO Hardware-in-the-Loop: Isaac Sim + Jetson TX2

Isaac Sim simulates the rover motion and renders the front camera. The PC turns the
frames into an event stream, runs the battery plant and the DARO controllers, and the
Jetson TX2 runs the perception applications (APP-DT, APP-OP) under the CPU DVFS level
chosen by the controller. Throughput and board power are **measured** on the TX2;
the battery (2RC ECM, thermal, BMS, Xu degradation) is the same model as in the paper's
Python evaluation.

```
┌──────────────────────── PC ────────────────────────┐                ┌──────── TX2 ────────┐
│ simulator/daro_sim.py   (Isaac Sim 6)              │                │ tx2/daro_tx2_agent  │
│   Leatherback, route, scene complexity, camera     │                │   APP-DT, APP-OP    │
│     │ /front_camera/rgb        ▲ /daro/speed_cmd   │  events (TCP)  │   (one process each)│
│     ▼                          │                   │ ─────────────▶ │   CPU DVFS (sysfs)  │
│ pc/event_camera_node.py  ──────┼───────────────────│                │   INA3221 VDD_IN    │
│ pc/mission_node.py  ◀── /odom, /daro/scene_zone    │  set_freq /    │                     │
│   HilRobotEnv (battery, BMS, reward, Xu metrics)   │ ◀── status ──▶ │                     │
│   DARO controller (methods/*)                      │  (TCP, 1 Hz)   │                     │
└────────────────────────────────────────────────────┘                └─────────────────────┘
```

The PC–TX2 link is plain TCP (`common/daro_protocol.py`, stdlib + numpy, Python 3.6
compatible), so the TX2 (JetPack 4.x) needs no ROS 2.

## What is measured, what is simulated

| Quantity | Source |
|---|---|
| Speed, distance, acceleration | Isaac Sim odometry (the commanded speed is tracked by the simulated Leatherback; corners and acceleration limits are real) |
| Motor power | DARO motor model (fit to robot measurements) evaluated on the measured speed profile, incl. motor thermal state and Mismatch plant overrides. PhysX wheel power is logged (`wheel_mech_w`) but not used: PhysX has no rolling resistance. |
| Scene complexity | Isaac Sim scene: object density along the route (`--complexity`), not a number |
| Events | ESIM-style emulation from the rendered frames (346×260, C = 0.2, 40 ms batches = 25 batch/s) |
| Throughput `q` per app | **TX2**: batches processed / busy time (processing capacity, as in the offline profiling) |
| Compute power | **TX2** INA3221 `VDD_IN`, × `compute_scale` of the plant |
| CPU frequency | **TX2** cpufreq, all policies pinned to the selected DARO level |
| Battery, BMS, sensor noise, cell variation, reward, ΔSOH | `HilRobotEnv` = `envs.robot_env.RobotEnv` (tested step-by-step identical) |

## Setup

**PC**: Ubuntu 24.04, ROS 2 Jazzy, Isaac Sim 6 (`ISAAC_PYTHON` = its Python), the DARO
repository (this folder lives in it). The PC nodes run with the system ROS Python
(`/usr/bin/python3`, numpy 1.26 is fine).

**TX2**: copy `tx2/` and `common/` to the TX2 (same relative layout), `python3 -m pip
install numpy` if missing. DVFS needs root.

Network: PC and TX2 on the same wired LAN (the event stream is a few MB/s).

## Running

1. **TX2** (replace the placeholder applications first, see below):
   ```bash
   sudo APPS="APP-DT=app_dt:AppDT APP-OP=app_op:AppOP" bash tx2/run_tx2_agent.sh --verbose
   ```
2. **PC, Isaac Sim** (one complexity for the whole run):
   ```bash
   ISAAC_PYTHON=~/miniconda3/envs/env_isaacsim/bin/python \
     bash simulator/launch_daro_sim.sh medium --headless
   ```
3. **PC, nodes** (event camera + missions; arguments go to `pc/mission_node.py`):
   ```bash
   TX2_HOST=192.168.1.20 bash pc/launch_pc_nodes.sh \
     --policies BA-EPD Lag-RLS "DA-Lag-RLS (w=4000)" --seeds 0 1 2 \
     --distance 1000 --soh 1.0 --c-label 2 --align --out ../logs/hil/nominal_c2
   ```

Each mission appends one line to `<out>/runs.jsonl` (the metrics of
`evaluation.metrics.run_episode` plus decision time and real-time factor) and writes
the step trajectory to `<out>/traj/<policy>_seed<k>.csv` (incl. measured throughput,
TX2 frequency, events per batch, zone, wheel power).

### Scenario families

| Family | Isaac Sim | `mission_node.py` |
|---|---|---|
| Nominal | `launch_daro_sim.sh low\|medium\|high` | `--distance 1000 --soh {1,0.9,0.8} --c-label {1,2,3}` |
| Tight | same | `--distance 1000 --soh 0.8 --tight-k 1.0 --c-label …` |
| Switch | `launch_daro_sim.sh zones --zone-order low,high,medium,high` | `--distance 1000 --align` |
| Mismatch | same as Nominal | `--plant payload+50% \| rough_v3x2 \| smooth_v3x0.7 \| compute_x1.3 \| compute_x0.8` |

`--c-label` is only used for the charge-limited start (`--tight-k`) and in the logs; the
controllers estimate the complexity from the measured throughput as in the paper.
In the Switch family the complexity changes with the position on the route
(switching period = zone length / speed); `--align` starts every mission at the lap
start so that all methods see the same zone sequence.

Oracles are not available in HIL (they need the future workload).

### Workload calibration (Table II)

The measured workload is the number of events per batch (paper Table II,
`pc/daro_hil/workload.py`):

| Complexity | 1 m/s | 2 m/s | 3 m/s | 4 m/s | 5 m/s |
|---|---|---|---|---|---|
| low (c = 1) | 12k | 18k | 24k | 30k | 36k |
| medium (c = 2) | 16k | 24k | 32k | 40k | 48k |
| high (c = 3) | 30k | 40k | 50k | 60k | 70k |

`pc/calibrate_events.py` drives the rover at 1–5 m/s in a running `daro_sim.py`
scene and reports the emulated events per batch for several contrast thresholds next
to the table (and the best threshold):

```bash
bash simulator/launch_daro_sim.sh low --headless &
python3 pc/calibrate_events.py --out ../logs/hil/calib_low.json
```

For the experiments, run the event node with `--match-table`: every batch is
resampled (subsampled, or replicated with ±1 px / time jitter) to the Table II count
at the measured speed and the zone complexity, so the TX2 load equals the profiled
workload while the spatial and temporal event structure comes from the rendered scene.
`/daro/events_per_batch_raw` keeps the scene's own count for reference.

## Plugging in APP-DT / APP-OP

`tx2/apps/reference_apps.py` contains **placeholders** with the same load shape (fixed
cost per batch + cost per event); they are only for bring-up. Wrap the real
applications in the same interface and pass them with `APPS=…`:

```python
class AppDT(object):
    def __init__(self, width, height): ...
    def process(self, events, t_start, batch_dt): ...   # events: t, x, y, p (structured array)
```

The application order must match the controller's (`qos_ref`, two applications at
25 batch/s).

## Testing without Isaac Sim / TX2

```bash
python3 tests/test_offline.py        # HilRobotEnv == RobotEnv, events, protocol, TX2 agent (dry run)
```

End-to-end with a fake simulator (synthetic camera) and the TX2 agent in dry-run mode:

```bash
source /opt/ros/jazzy/setup.bash
python3 tx2/daro_tx2_agent.py --dry-run &
python3 tests/fake_isaac_node.py --zone-order medium &
python3 pc/event_camera_node.py --tx2-host 127.0.0.1 &
python3 pc/mission_node.py --tx2-host 127.0.0.1 --policies Lag-RLS --seeds 0 --distance 150 --out /tmp/hil_test
```

`pc/mission_node.py --no-tx2` replaces the TX2 by the throughput and power models
(software-in-the-loop with Isaac Sim dynamics).

## Notes

* Real time: the TX2 processes events in wall time, so `daro_sim.py` paces simulation
  time to wall time; the achieved real-time factor is logged per step (`rtf`) and per
  mission (`mean_rtf`). Results with `mean_rtf` well below 1 are not valid.
* Thermal throttling of the TX2 changes the throughput; `run_tx2_agent.sh` fixes the fan
  at full speed. Do not run other loads on the TX2.
* The original AsyncMEC launchers (`fixed_complexity.py`, `asyncmec_demo.py`) are kept
  unchanged; `daro_sim.py` is derived from `asyncmec_demo.py`.
* Defaults: ROS domain 20, `rmw_fastrtps_cpp`, TX2 ports 5600 (events) / 5601 (control).
