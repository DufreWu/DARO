"""Sanity-check the reward before training: is the reward optimum the energy optimum?

Drives the test environment with simple fixed policies and compares
  1. speed sweep (constant v, CPU at the minimum QoS frequency):
     reward-optimal v  vs  energy-optimal v, per distance x complexity;
  2. CPU sweep at that speed: reward-optimal f vs minimum QoS-safe f;
  3. smoothness: steady speed vs the same mean speed with +-dv oscillation.

    python evaluation/reward_check.py
    python -m evaluation.reward_check --distances 500 5000 --complexity 1 3

Pass criteria are printed per row (OK / MISMATCH).
"""

from __future__ import annotations

if __name__ == '__main__' and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = 'evaluation'
    import evaluation  # noqa: F401

import argparse

import numpy as np

from . import EVAL_CONFIG, make_robot
from robot_env import RobotEnv, throughput_freq_required

V_GRID = np.round(np.arange(1.0, 5.01, 0.25), 2)
F_GRID = np.round(np.arange(0.5, 2.001, 0.05), 3)


class _Fixed:
    """Constant (or periodically oscillating) action."""

    def __init__(self, v, f, dv=0.0, period=4):
        self.v, self.f, self.dv, self.period, self.k = v, f, dv, period, 0

    def __call__(self, obs):
        self.k += 1
        sign = 1.0 if (self.k // (self.period // 2)) % 2 == 0 else -1.0
        return [self.v + sign * self.dv, self.f]


def rollout(distance, complexity, policy, seed=42, jitter=None):
    cfg = dict(EVAL_CONFIG, mission_distance_m=float(distance))
    robot = make_robot(cfg)
    env = RobotEnv(robot=robot, mode='test', soh_init=1.0, soc_init=1.0,
                   complexity=complexity,
                   complexity_jitter=cfg['test_complexity_jitter'] if jitter is None else jitter)
    obs, _ = env.reset(seed=seed)
    ret, viol, done, info = 0.0, 0, False, {}
    while not done:
        obs, r, te, tr, info = env.step(policy(obs))
        ret += r
        viol += int(info['qos_violation'] > 0)
        done = te or tr
    return dict(ret=ret, energy_kj=env.total_energy_j / 1e3,
                wh_km=env.total_energy_j / 3.6 / distance,
                time_s=float(info['t_s']), completed=bool(info['completed']),
                viol_pct=100.0 * viol / max(env.step_count, 1))


def qos_safe_f(v, complexity, robot, cfg):
    """Lowest frequency meeting QoS under the worst-case complexity disturbance."""
    q = max(robot.qos_ref) * cfg['baseline_qos_margin']
    c = complexity * (1.0 + cfg['test_complexity_jitter'])
    return float(np.clip(throughput_freq_required(v, q, c), robot.f_min, robot.f_max))


def speed_sweep(d, c, robot, cfg):
    rows = []
    for v in V_GRID:
        r = rollout(d, c, _Fixed(v, qos_safe_f(v, c, robot, cfg)))
        if r['completed']:
            rows.append((float(v), r))
    v_r = max(rows, key=lambda t: t[1]['ret'])
    v_e = min(rows, key=lambda t: t[1]['energy_kj'])
    return rows, v_r, v_e


def main():
    ap = argparse.ArgumentParser(description='Reward sanity check (reward optimum vs energy optimum)')
    ap.add_argument('--distances', type=float, nargs='+', default=[500, 1000, 2500, 5000])
    ap.add_argument('--complexity', type=float, nargs='+', default=[1, 2, 3])
    ap.add_argument('--verbose', action='store_true', help='Print the full speed sweep')
    args = ap.parse_args()
    cfg = dict(EVAL_CONFIG)
    robot = make_robot(cfg)
    p = robot
    print(f'Reward: w_energy={p.w_energy} e_ref={p.energy_ref_wh_per_km} Wh/km '
          f'B={p.completion_bonus} w_time={p.w_time} w_smooth={p.w_smooth} '
          f'w_stress={p.w_stress} w_qos={p.w_qos}\n')

    print('1) Speed sweep (CPU at minimum QoS-safe frequency)')
    print(f'{"dist":>6} {"C":>3} | {"v_reward":>8} {"v_energy":>8} | {"E@v_r":>7} {"E@v_e":>7} '
          f'{"dE":>6} | {"R@v_r":>7} {"R@v_e":>7} | check')
    best_v = {}
    for d in args.distances:
        for c in args.complexity:
            rows, (vr, rr), (ve, re) = speed_sweep(d, c, robot, cfg)
            best_v[(d, c)] = vr
            de = 100 * (rr['energy_kj'] / re['energy_kj'] - 1)
            ok = 'OK' if de < 1.0 else 'MISMATCH'
            print(f'{d:6.0f} {c:3g} | {vr:8.2f} {ve:8.2f} | {rr["energy_kj"]:7.2f} '
                  f'{re["energy_kj"]:7.2f} {de:+5.1f}% | {rr["ret"]:7.3f} {re["ret"]:7.3f} | {ok}')
            if args.verbose:
                for v, r in rows:
                    print(f'{"":13}v={v:4.2f} R={r["ret"]:7.3f} E={r["energy_kj"]:7.2f} '
                          f'viol={r["viol_pct"]:4.1f}%')

    print('\n2) CPU sweep at the reward-optimal speed')
    print(f'{"dist":>6} {"C":>3} | {"f_reward":>8} {"f_qos":>6} | {"E@f_r":>7} {"E@2GHz":>7} '
          f'{"R@f_r":>7} {"R@2GHz":>7} | check')
    for d in args.distances:
        for c in args.complexity:
            v = best_v[(d, c)]
            res = [(float(f), rollout(d, c, _Fixed(v, f))) for f in F_GRID]
            fr, rr = max(res, key=lambda t: t[1]['ret'])
            r2 = rollout(d, c, _Fixed(v, p.f_max))
            fq = qos_safe_f(v, c, robot, cfg)
            ok = 'OK' if (rr['viol_pct'] < 1.0 and rr['ret'] >= r2['ret']) else 'MISMATCH'
            print(f'{d:6.0f} {c:3g} | {fr:8.2f} {fq:6.2f} | {rr["energy_kj"]:7.2f} '
                  f'{r2["energy_kj"]:7.2f} {rr["ret"]:7.3f} {r2["ret"]:7.3f} | {ok}'
                  f'  (viol@f_r {rr["viol_pct"]:.1f}%)')

    print('\n3) Smoothness: steady vs +-0.5 m/s oscillation (same mean speed)')
    for d in args.distances:
        c = args.complexity[0]
        v = best_v[(d, c)]
        f = qos_safe_f(v + 0.5, c, robot, cfg)
        a = rollout(d, c, _Fixed(v, f))
        b = rollout(d, c, _Fixed(v, f, dv=0.5))
        ok = 'OK' if a['ret'] > b['ret'] else 'MISMATCH'
        print(f'{d:6.0f} C={c:g} v={v:.2f}: steady R={a["ret"]:.3f} E={a["energy_kj"]:.2f} | '
              f'oscillating R={b["ret"]:.3f} E={b["energy_kj"]:.2f} | {ok}')


if __name__ == '__main__':
    main()
