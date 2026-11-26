#!/usr/bin/env python3
"""DARO HIL mission runner (PC, ROS 2).

Closes the loop Isaac Sim -> plant (battery, BMS) -> controller -> Isaac Sim / TX2:

  /odom (Isaac Sim, ~60 Hz)       measured speed, distance and acceleration
  /daro/scene_zone                complexity zone label (logged only)
  /daro/route_progress            lap progress in [0, 1) (start alignment)
  /daro/wheel_power               wheel mechanical power from PhysX (logged only)
  TX2 status (TCP, 1 Hz)          throughput per application, board power, frequency
  ->  /daro/speed_cmd (10 Hz)     speed command for the Isaac Sim hybrid mode
  ->  TX2 set_freq                CPU DVFS level

Every control period (1 s of simulation time) the plant (HilRobotEnv) is advanced
with the measurements of the period, and the controller selects the next action
from the same observation vector as in the Python evaluation. Missions run for
each (seed, policy) pair; between missions the rover stops for --rest-s and, with
--align, drives to the lap start so that every mission sees the same zone sequence.

    python3 pc/mission_node.py --tx2-host 192.168.1.20 --policies BA-EPD Lag-RLS \
        "DA-Lag-RLS (w=4000)" --seeds 0 1 2 --distance 1000 --soh 1.0 --c-label 2 --out logs/hil/nominal
    python3 pc/mission_node.py --no-tx2 ...   # software-in-the-loop: TX2 replaced by the models
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from daro_hil.hil_env import HilRobotEnv, DARO_ROOT  # noqa: E402
from daro_hil.tx2_link import Tx2Control  # noqa: E402

sys.path.insert(0, str(DARO_ROOT))
from envs.robot_env import (ECMParams, StressParams, CPU_CLOCK_KHZ, N_FREQ_LEVELS,  # noqa: E402
                            compute_throughput, non_motor_power, tight_soc_init)
from envs.scenarios import PLANTS  # noqa: E402
from envs.test_env import EVAL_CONFIG, make_robot  # noqa: E402
from evaluation.benchmark import POLICIES  # noqa: E402
from evaluation.metrics import xu_mission_fd  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from std_msgs.msg import Float32, String  # noqa: E402

ZONE_C = {'low': 1.0, 'medium': 2.0, 'high': 3.0}


class OdomAccumulator:
    """Integrates the odometry of one control period (simulation time)."""

    def __init__(self, accel_tau_s: float = 0.2):
        self.tau = accel_tau_s
        self.t = None
        self.pos = None
        self.v = 0.0
        self.a = 0.0
        self.reset_period()

    def reset_period(self):
        self.dist = 0.0
        self.e_motor = 0.0
        self.e_motor_cold = 0.0
        self.time = 0.0
        self.v_start = self.v

    def update(self, t: float, x: float, y: float, vx: float, vy: float, power_fn):
        v = math.hypot(vx, vy)
        if self.t is None:
            self.t, self.pos, self.v = t, (x, y), v
            return
        dt = t - self.t
        if dt <= 0.0:
            return
        if self.pos is not None:
            self.dist += math.hypot(x - self.pos[0], y - self.pos[1])
        a_raw = (v - self.v) / dt
        alpha = dt / (self.tau + dt)
        self.a += alpha * (a_raw - self.a)
        cold, hot = power_fn(v, self.a)
        self.e_motor += hot * dt
        self.e_motor_cold += cold * dt
        self.time += dt
        self.t, self.pos, self.v = t, (x, y), v


class MissionNode(Node):
    def __init__(self, args):
        super().__init__('daro_hil_mission')
        self.args = args
        self.out = Path(args.out)
        (self.out / 'traj').mkdir(parents=True, exist_ok=True)
        self.tx2 = None if args.no_tx2 else Tx2Control(args.tx2_host, args.tx2_ctrl_port)
        self.speed_pub = self.create_publisher(Float32, args.speed_topic, 10)
        self.create_subscription(Odometry, args.odom_topic, self._odom_cb, qos_profile_sensor_data)
        self.create_subscription(String, args.zone_topic, self._zone_cb, 10)
        self.create_subscription(Float32, args.progress_topic, self._progress_cb, 10)
        self.create_subscription(Float32, args.wheel_power_topic, self._wheel_cb, 10)
        self.odom = OdomAccumulator()
        self.t_sim = None
        self.zone = None
        self.progress = None
        self.prev_progress = None
        self.wheel_w = float('nan')
        self.wheel_sum, self.wheel_n = 0.0, 0
        self.v_cmd = 0.0
        self.f_ghz = float(CPU_CLOCK_KHZ[-1]) / 1e6
        self.runs = [(seed, name) for seed in args.seeds for name in args.policies]
        self.run_i = 0
        self.state = 'align' if args.align else 'start'
        self.t_state = time.monotonic()
        self.env = None
        self.policy = None
        self.create_timer(0.1, self._publish_speed)
        self.create_timer(0.05, self._tick)
        self.get_logger().info(f'{len(self.runs)} missions -> {self.out}')

    # ---------------- subscriptions ----------------
    def _power_fn(self, v, a):
        if self.env is None:
            return 0.0, 0.0
        return self.env.motor_power_sample(v, a)

    def _odom_cb(self, msg: Odometry):
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        self.t_sim = t
        self.odom.update(t, msg.pose.pose.position.x, msg.pose.pose.position.y,
                         msg.twist.twist.linear.x, msg.twist.twist.linear.y, self._power_fn)

    def _zone_cb(self, msg: String):
        self.zone = msg.data

    def _progress_cb(self, msg: Float32):
        self.prev_progress, self.progress = self.progress, float(msg.data)

    def _wheel_cb(self, msg: Float32):
        if math.isfinite(msg.data):
            self.wheel_sum += float(msg.data)
            self.wheel_n += 1

    def _publish_speed(self):
        self.speed_pub.publish(Float32(data=float(self.v_cmd)))

    # ---------------- mission state machine ----------------
    def _tick(self):
        if self.t_sim is None:
            return
        if self.state == 'align':
            self.v_cmd = self.args.align_speed
            if (self.progress is not None and self.prev_progress is not None
                    and self.prev_progress > 0.9 and self.progress < 0.1):
                self.state = 'start'
        elif self.state == 'start':
            self._start_mission()
        elif self.state == 'run':
            if self.odom.time >= self.robot.dt - 1e-6:
                self._control_step()
        elif self.state == 'rest':
            self.v_cmd = 0.0
            if time.monotonic() - self.t_state >= self.args.rest_s:
                if self.run_i >= len(self.runs):
                    self.get_logger().info('all missions done')
                    self.state = 'done'
                    rclpy.shutdown()
                    return
                self.state = 'align' if self.args.align else 'start'

    def _make(self, name: str):
        cls, over, _group = POLICIES[name]
        if cls == 'oracle':
            raise ValueError('oracles need the true future workload; not available in HIL')
        cfg = dict(EVAL_CONFIG, **{k: v for k, v in over.items() if k != 'observe_c'})
        cfg['mission_distance_m'] = self.args.distance
        robot = make_robot(cfg)
        env = HilRobotEnv(robot=robot, ecm=ECMParams.default(robot), stress=StressParams(),
                          soc_init=self.soc0, soh_init=self.args.soh, complexity=self.args.c_label,
                          mode='test', complexity_jitter=0.0)
        policy = cls(make_robot(cfg), cfg)
        if getattr(policy, 'needs_sensors', False):
            raise ValueError(f'{name} needs extra simulated sensors; not supported in HIL')
        return env, policy, cfg

    @property
    def soc0(self) -> float:
        if self.args.soc0 is not None:
            return self.args.soc0
        if self.args.tight_k is not None:
            return tight_soc_init(PLANTS.get(self.args.plant, {}), self.args.soh,
                                  self.args.c_label, self.args.distance, self.args.tight_k)
        return 1.0

    def _start_mission(self):
        seed_i, name = self.runs[self.run_i]
        self.name, self.seed = name, EVAL_CONFIG['seed'] + int(seed_i)
        self.env, self.policy, self.cfg = self._make(name)
        self.robot = self.env.robot
        options = {'plant': PLANTS.get(self.args.plant, {})} if self.args.plant else None
        obs, _ = self.env.reset(seed=self.seed, options=options)
        if hasattr(self.policy, 'reset'):
            self.policy.reset()
        self.ret, self.viol_steps, self.steps, self.decision_s = 0.0, 0, 0, 0.0
        self.t0_wall, self.t0_sim = time.monotonic(), self.t_sim
        self.rows = []
        self.get_logger().info(f'mission {self.run_i + 1}/{len(self.runs)}: {name} seed {seed_i} '
                               f'SOC0={self.env.soc_init:.3f} SOH={self.env.soh_init:.3f} '
                               f'L={self.args.distance:.0f} m plant={self.args.plant or "nominal"}')
        self._act(obs)
        self.odom.reset_period()
        self.wheel_sum, self.wheel_n = 0.0, 0
        self.state = 'run'

    def _act(self, obs):
        t0 = time.perf_counter()
        action = self.policy.select_action(obs, deterministic=True)
        self.decision_s += time.perf_counter() - t0
        v = float(np.clip(action[0], self.robot.v_min, self.robot.v_max))
        level = int(np.clip(np.rint(action[1]), 0, N_FREQ_LEVELS - 1))
        self.v_cmd = v
        self.f_ghz = CPU_CLOCK_KHZ[level] / 1e6
        if self.tx2 is not None:
            self.tx2.set_freq_khz(CPU_CLOCK_KHZ[level])
        self.level = level

    def _measure_compute(self, v: float, c_zone: float):
        """(throughput list, compute power, extra log fields) from the TX2 or the models."""
        if self.tx2 is None:
            c = c_zone if math.isfinite(c_zone) else self.args.c_label
            thr = [compute_throughput(v, self.f_ghz, c)] * self.robot.n_apps
            return thr, float(non_motor_power(v, self.f_ghz, c, self.robot)), {}
        st = self.tx2.latest()
        if st is None:
            raise RuntimeError('lost the TX2 control connection')
        age = time.monotonic() - st['t_rx']
        extra = {'tx2_age_s': age, 'tx2_freq_khz': st['freq_khz'][0] if st['freq_khz'] else -1,
                 'tx2_rate': st['rate'], 'events_per_batch': st['events_per_batch']}
        if age > 3.0:
            self.get_logger().warn(f'TX2 status is {age:.1f} s old')
        return list(st['thr']), float(st['power_w']), extra

    def _control_step(self):
        o = self.odom
        dt = o.time
        v_mean = o.dist / dt
        accel = (o.v - o.v_start) / dt
        motor_w, motor_cold_w = o.e_motor / dt, o.e_motor_cold / dt
        c_zone = ZONE_C.get(self.zone, float('nan'))
        thr, compute_w, extra = self._measure_compute(v_mean, c_zone)
        extra.update(zone=self.zone or '', wheel_mech_w=(self.wheel_sum / self.wheel_n
                                                         if self.wheel_n else float('nan')),
                     v_cmd=self.v_cmd, dvfs_level=self.level,
                     rtf=(self.t_sim - self.t0_sim) / max(time.monotonic() - self.t0_wall, 1e-9))
        obs, reward, term, trunc, info = self.env.step_measured(
            self.f_ghz, v_mean, accel, o.dist, dt, thr, compute_w,
            motor_w=motor_w, motor_cold_w=motor_cold_w, complexity=c_zone, extra=extra)
        o.reset_period()
        self.wheel_sum, self.wheel_n = 0.0, 0
        self.ret += reward
        self.viol_steps += int(info['qos_violation'] > 0.0)
        self.steps += 1
        self.rows.append(info)
        if self.steps % 10 == 0:
            self.get_logger().info(
                f'[{self.name}] t={info["t_s"]:.0f}s d={info["distance_m"]:.0f}m v={v_mean:.2f} '
                f'f={self.f_ghz:.2f}GHz thr={[round(x, 1) for x in thr]} P={info["power_w"]:.1f}W '
                f'SOC={info["soc"]:.3f} zone={self.zone} rtf={extra["rtf"]:.2f}')
        if term or trunc:
            self._finish(info)
        else:
            self._act(obs)

    def _finish(self, info):
        self.v_cmd = 0.0
        env = self.env
        dist_km = max(env.distance_m, 1e-9) / 1000.0
        fade = xu_mission_fd(env, self.cfg['charge_c_rate'])
        m = {
            'policy': self.name, 'seed': self.seed, 'plant': self.args.plant or 'nominal',
            'soh': env.soh_init, 'soc0': env.soc_init, 'distance_m': self.args.distance,
            'c_label': self.args.c_label, 'tx2': self.tx2 is not None,
            'return': self.ret, 'energy_kj': env.total_energy_j / 1e3,
            'wh_per_km': env.total_energy_j / 3600.0 / dist_km,
            'mission_time_s': float(info.get('t_s', 0.0)),
            'completed': float(bool(info.get('completed', False))),
            'qos_viol_pct': 100.0 * self.viol_steps / max(self.steps, 1),
            'final_soc': float(env.battery.soc),
            'reserve_viol': float(float(env.battery.soc) < env.robot.soc_reserve - 1e-9),
            'delta_soh_ppm': 1e6 * fade['delta_soh'], 'fd_cycle': fade['fd_cycle'],
            'decision_us': 1e6 * self.decision_s / max(self.steps, 1),
            'mean_rtf': float(np.mean([r.get('rtf', np.nan) for r in self.rows])) if self.rows else float('nan'),
            'steps': self.steps,
        }
        with open(self.out / 'runs.jsonl', 'a') as fh:
            fh.write(json.dumps(m) + '\n')
        tag = f'{self.name}_seed{self.seed}'.replace(' ', '_').replace('/', '-')
        keys = sorted({k for r in self.rows for k in r})
        with open(self.out / 'traj' / f'{tag}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
        self.get_logger().info(
            f'done {self.name}: R={m["return"]:.3f} completed={m["completed"]:.0f} '
            f'T={m["mission_time_s"]:.0f}s E={m["energy_kj"]:.1f}kJ QoS-viol={m["qos_viol_pct"]:.1f}% '
            f'dSOH={m["delta_soh_ppm"]:.1f}ppm rtf={m["mean_rtf"]:.2f}')
        self.run_i += 1
        self.state, self.t_state = 'rest', time.monotonic()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    names = [n for n, v in POLICIES.items() if v[0] != 'oracle']
    ap.add_argument('--policies', nargs='+', default=['BA-EPD', 'Lag-RLS', 'DA-Lag-RLS (w=4000)'],
                    choices=names)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                    help='seed offsets (added to the evaluation seed): cell variation, sensor noise')
    ap.add_argument('--distance', type=float, default=1000.0)
    ap.add_argument('--soh', type=float, default=1.0)
    ap.add_argument('--soc0', type=float, default=None)
    ap.add_argument('--tight-k', type=float, default=None,
                    help='charge-limited start: usable energy = k x energy-optimal mission energy')
    ap.add_argument('--c-label', type=float, default=2.0,
                    help='nominal complexity of the Isaac scene (tight SOC0 and logs; not simulated)')
    ap.add_argument('--plant', default='', choices=[''] + list(PLANTS))
    ap.add_argument('--out', default=str(DARO_ROOT / 'logs' / 'hil' / 'run'))
    ap.add_argument('--tx2-host', default='192.168.1.20')
    ap.add_argument('--tx2-ctrl-port', type=int, default=5601)
    ap.add_argument('--no-tx2', action='store_true', help='use the throughput/power models instead of the TX2')
    ap.add_argument('--speed-topic', default='/daro/speed_cmd')
    ap.add_argument('--odom-topic', default='/odom')
    ap.add_argument('--zone-topic', default='/daro/scene_zone')
    ap.add_argument('--progress-topic', default='/daro/route_progress')
    ap.add_argument('--wheel-power-topic', default='/daro/wheel_power')
    ap.add_argument('--align', action='store_true', help='start every mission at the lap start')
    ap.add_argument('--align-speed', type=float, default=2.0)
    ap.add_argument('--rest-s', type=float, default=5.0)
    args = ap.parse_args()
    rclpy.init()
    node = MissionNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.v_cmd = 0.0
        if rclpy.ok():
            node._publish_speed()
        if node.tx2 is not None:
            node.tx2.close()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
