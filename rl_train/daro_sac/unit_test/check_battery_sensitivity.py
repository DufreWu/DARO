"""Does the battery state change the optimal action? (sanity check for DARO)

Runs constant-speed policies (CPU at the lowest frequency meeting QoS at the
nominal complexity) for several (SOH, initial SOC) pairs and reports the
reward-optimal and energy-optimal speeds. If these speeds do not move with
SOC/SOH, battery-aware observations cannot help and DARO has no advantage
over RL-SOC-only / RL-SOH-only / QoS-EPD by construction.

Uses the environment in rl_train/daro_sac (robot_env.py). From any directory:

    python rl_train/daro_sac/unit_test/check_battery_sensitivity.py            # current env
    python rl_train/daro_sac/unit_test/check_battery_sensitivity.py --legacy   # load-side energy,
                                                                               # rated-Q stress, no low-SOC term
    python rl_train/daro_sac/unit_test/check_battery_sensitivity.py --distance 1000 --complexity 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl_train/daro_sac

import numpy as np

import evaluation  # noqa: F401  (sets sys.path)
from evaluation import EVAL_CONFIG, make_test_env, TestScenario
from evaluation.metrics import run_episode
from robot_env import throughput_freq_required

PAIRS = [(1.0, 1.0), (0.8, 1.0), (1.0, 0.6), (1.0, 0.3), (0.8, 0.3)]


class ConstantPolicy:
    def __init__(self, v, f):
        self.a = np.array([v, f], dtype=np.float32)

    def select_action(self, obs, deterministic=True):
        return self.a


def scan(soh, soc, cfg, complexity, speeds, legacy, seed=1):
    rows = []
    for v in speeds:
        env = make_test_env(cfg, TestScenario(soh, soc, complexity))
        if legacy:
            env.robot.energy_at_cell = False
            env.stress_params.c_rate_actual_capacity = False
            env.stress_params.low_soc_gain = 0.0
        q = max(env.robot.qos_ref) * 1.05
        f = float(np.clip(throughput_freq_required(v, q, complexity * 1.2),
                          env.robot.f_min, env.robot.f_max))
        m = run_episode(env, ConstantPolicy(v, f), seed=seed)
        if m['completed'] < 1.0:
            continue
        rows.append((v, m['return'], m['energy_kj'], m['stress_proxy'], m['qos_viol_pct']))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', type=float, default=1000.0)
    ap.add_argument('--complexity', type=float, default=1.0)
    ap.add_argument('--legacy', action='store_true', help='original env (load-side energy, rated-Q stress)')
    ap.add_argument('--vmin', type=float, default=1.0)
    ap.add_argument('--vmax', type=float, default=4.0)
    ap.add_argument('--dv', type=float, default=0.05)
    args = ap.parse_args()

    cfg = dict(EVAL_CONFIG, mission_distance_m=args.distance, test_complexity_jitter=0.0)
    speeds = np.arange(args.vmin, args.vmax + 1e-9, args.dv)
    print(f"{'legacy' if args.legacy else 'current'} env, {args.distance:.0f} m, C={args.complexity:g}")
    print(f"{'SOH':>5} {'SOC0':>5} | {'v*_reward':>9} {'R*':>7} {'E@v*':>7} {'S@v*':>6} "
          f"| {'v*_energy':>9} {'E_min':>7} | E@2.5m/s")
    for soh, soc in PAIRS:
        rows = scan(soh, soc, cfg, args.complexity, speeds, args.legacy)
        if not rows:
            print(f"{soh:5.2f} {soc:5.2f} | no completed run")
            continue
        rb = max(rows, key=lambda r: r[1])
        re = min(rows, key=lambda r: r[2])
        e25 = min(rows, key=lambda r: abs(r[0] - 2.5))[2]
        print(f"{soh:5.2f} {soc:5.2f} | {rb[0]:9.2f} {rb[1]:7.3f} {rb[2]:7.3f} {rb[3]:6.2f} "
              f"| {re[0]:9.2f} {re[2]:7.3f} | {e25:.3f}")


if __name__ == '__main__':
    main()
