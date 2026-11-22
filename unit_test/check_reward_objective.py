"""Does the reward encode "finish fast, subject to battery charge and QoS"?

Three checks on the CURRENT reward (robot_env.RobotEnv.reward_terms / terminal_reward),
no training needed:

  A. Enough charge: constant-speed scan, CPU at the QoS frequency for complexity
     c * m (m = jitter margin). The reward-optimal speed should be clearly above
     the energy-optimal speed, and the reward-optimal m should give ~0 QoS violation.
  B. Tight charge: same scan at lower initial SOC (and SOH 0.8). The reward-optimal
     speed should drop towards the energy optimum (or below, to finish at all).
  C. Policies: battery-aware rule (fast if affordable, scenario_utils.BatteryAware)
     should score a clearly higher return than QoS-EPD (energy-optimal) and
     no worse than a fixed fast speed when the battery is tight.

    python unit_test/check_reward_objective.py          # ~2 min
    python unit_test/check_reward_objective.py --only A
"""

from __future__ import annotations

import argparse

import numpy as np

from scenario_utils import (EPD, FixedSpeed, BatteryAware, make_env, make_config, run,
                            mean_runs, nominal_robot)
from envs.robot_env import throughput_freq_required, freq_action

MARGINS = (1.0, 1.1, 1.2, 1.25)
# Initial SOC grid for the 10 km tight-charge cases (QoS-EPD just finishes at the lowest).
TIGHT_SOC = {1.0: (0.40, 0.27, 0.25, 0.23), 0.8: (0.40, 0.33, 0.31, 0.29)}


def constant_episode(distance, v, c, m, soh=1.0, soc0=1.0, seed=1):
    env, _ = make_env(distance, soh=soh, soc0=soc0, complexity=c)
    env.reset(seed=seed)
    p = env.robot
    f = float(np.clip(throughput_freq_required(v, max(p.qos_ref), c * m), p.f_min, p.f_max))
    ret, viol, n, done, info = 0.0, 0, 0, False, {}
    while not done:
        _, r, te, tr, info = env.step(freq_action(v, f))
        ret += r
        viol += int(info['qos_violation'] > 0.0)
        n += 1
        done = te or tr
    return dict(ret=ret, energy_kj=env.total_energy_j / 1e3, time_s=info['t_s'],
                viol_pct=100.0 * viol / max(n, 1), completed=bool(info['completed']),
                soc_end=float(env.battery.soc))


def scan(distance, c, soh, soc0, speeds, seeds=(1, 2)):
    """Best (v, m) by mean return; energy-optimal v among completed QoS-safe runs."""
    rows = []
    for v in speeds:
        for m in MARGINS:
            rs = [constant_episode(distance, v, c, m, soh, soc0, s) for s in seeds]
            rows.append(dict(v=v, m=m, **{k: float(np.mean([r[k] for r in rs])) for k in rs[0]}))
    best = max(rows, key=lambda r: r['ret'])
    safe = [r for r in rows if r['completed'] >= 1.0 and r['m'] == 1.2]
    e_opt = min(safe, key=lambda r: r['energy_kj']) if safe else None
    return best, e_opt


def fmt_best(b):
    return (f"v*={b['v']:.1f} m={b['m']:.2f} R={b['ret']:7.3f} viol={b['viol_pct']:4.1f}% "
            f"T={b['time_s'] / 60:5.1f}min done={b['completed']:.0%} SOC_end={b['soc_end']:.3f}")


def check_a(args):
    print("A. Enough charge (SOC0 1, SOH 1/0.8): reward optimum vs energy optimum")
    speeds = np.round(np.arange(2.5, 5.01, 0.1), 2)
    for c in (1.0, 2.0, 3.0):
        for soh in (1.0, 0.8):
            best, e_opt = scan(args.distance, c, soh, 1.0, speeds)
            ok = best['v'] >= e_opt['v'] + 0.3 or (c == 3.0 and best['v'] >= e_opt['v'])
            print(f"  C={c:g} SOH={soh:.1f}: {fmt_best(best)} | v_energy={e_opt['v']:.1f} "
                  f"| {'OK' if ok and best['viol_pct'] < 1.0 else 'CHECK'}")
    print()


def check_b(args):
    d = args.tight_distance
    print(f"B. Charge budget, {d / 1000:g} km, C=1: reward-optimal speed vs initial SOC")
    speeds = np.round(np.arange(1.5, 5.01, 0.1), 2)
    for soh in (1.0, 0.8):
        for soc0 in TIGHT_SOC[soh]:
            best, e_opt = scan(d, 1.0, soh, soc0, speeds, seeds=(1,))
            ev = f"{e_opt['v']:.1f}" if e_opt else ' - '
            print(f"  SOH={soh:.1f} SOC0={soc0:.2f}: {fmt_best(best)} | v_energy={ev}")
    print()


def check_c(args):
    print("C. Policy returns (mean over seeds): QoS-EPD vs fast vs battery-aware")
    cases = [(1000.0, 1.0, 1.0, 1.0), (1000.0, 1.0, 1.0, 3.0), (1000.0, 0.8, 0.3, 2.0)]
    cases += [(args.tight_distance, soh, soc0, 1.0) for soh in (1.0, 0.8) for soc0 in TIGHT_SOC[soh]]
    cases += [(35000.0, 0.8, 1.0, 3.0)]
    labels = ('QoS-EPD', f'fast {args.v_fast:g}', 'battery-aware')
    print("  cell: R  done%  T[min]  SOC_end  QoS viol%")
    print(f"  {'case':>24} | " + ' | '.join(f"{l:>36}" for l in labels))
    for d, soh, soc0, c in cases:
        cfg = make_config(d)
        res = {}
        for label in labels:
            runs = []
            for s in range(args.seeds):
                env, _ = make_env(d, soh=soh, soc0=soc0, complexity=c)
                if label == 'QoS-EPD':
                    pol = EPD(nominal_robot(cfg), cfg)
                elif label.startswith('fast'):
                    pol = FixedSpeed(nominal_robot(cfg), cfg, args.v_fast)
                else:
                    pol = BatteryAware(nominal_robot(cfg), cfg, v_fast=args.v_fast)
                ret = [0.0]
                orig_step = env.step

                def step(a, _o=orig_step, _r=ret):
                    out = _o(a)
                    _r[0] += out[1]
                    return out
                env.step = step
                r = run(env, pol, seed=4000 + s)
                r['ret'] = ret[0]
                runs.append(r)
            res[label] = mean_runs(runs)
        name = f"{d / 1000:g}km SOH{soh:.1f} SOC{soc0:.2f} C{c:g}"
        print(f"  {name:>24} | " + ' | '.join(
            f"{res[l]['ret']:7.3f} {100 * res[l]['completed']:4.0f}% {res[l]['time_s'] / 60:6.1f} "
            f"{res[l]['final_soc']:6.3f} {res[l]['qos_viol_pct']:5.1f}%".rjust(36) for l in labels))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['A', 'B', 'C'])
    ap.add_argument('--distance', type=float, default=1000.0)
    ap.add_argument('--tight-distance', type=float, default=10000.0)
    ap.add_argument('--v-fast', type=float, default=3.9)
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()
    for k, fn in (('A', check_a), ('B', check_b), ('C', check_c)):
        if args.only in (None, k):
            fn(args)


if __name__ == '__main__':
    main()
