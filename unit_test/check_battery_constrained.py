"""Scenario 3: the battery as a constraint (long missions, aged / partly charged pack).

Battery state does not change the energy-optimal speed for this pack
(check_battery_sensitivity.py), but it decides what is AFFORDABLE: whether the
robot can drive faster (shorter mission time), must drive energy-optimally,
or must drop QoS to finish at all. Policies:

  QoS-EPD        energy-optimal + QoS, battery-agnostic (maximum range with QoS)
  fast           constant v_fast (4 m/s) + QoS frequency, battery-agnostic (time-oriented)
  battery-aware  QoS-EPD + feasibility check on the true usable energy
                 (SOC, SOH, OCV): fast if affordable, else energy-optimal,
                 else energy-minimal with f_min (QoS dropped) to finish

Reported: completion, mission time, final SOC, QoS violation. The battery-aware
policy should match 'fast' on time when the battery allows it, and QoS-EPD
(or better, by degrading QoS) on completion when it does not. The DARO
reward now values the time saving (T_d, w_time = 1); see check_reward_objective.py.

    python unit_test/check_battery_constrained.py
    python unit_test/check_battery_constrained.py --distances 30000 45000 --soh 1.0 0.8
"""

from __future__ import annotations

import argparse

from scenario_utils import (EPD, FixedSpeed, BatteryAware, make_env, make_config, run,
                            mean_runs, nominal_robot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distances', type=float, nargs='+', default=[20000.0, 35000.0, 45000.0, 55000.0])
    ap.add_argument('--soh', type=float, nargs='+', default=[1.0, 0.8])
    ap.add_argument('--soc0', type=float, nargs='+', default=[1.0])
    ap.add_argument('--complexities', type=float, nargs='+', default=[1.0, 3.0])
    ap.add_argument('--v-fast', type=float, default=4.0)
    ap.add_argument('--seeds', type=int, default=2)
    args = ap.parse_args()

    labels = ('QoS-EPD', 'fast', 'battery-aware')
    print(f"Battery-constrained missions, jitter 0.2, {args.seeds} seeds, v_fast {args.v_fast:g} m/s.")
    print("Cells: done%  time[min]  final SOC  QoS level% (mean thr/QoS_ref)  (km driven if not done)")
    print(f"{'dist':>6} {'SOH':>4} {'SOC0':>4} {'C':>2} | " + ' | '.join(f"{l:>30}" for l in labels))
    for c in args.complexities:
        for soh in args.soh:
            for soc0 in args.soc0:
                for d in args.distances:
                    cfg = make_config(d)
                    res = {}
                    for label in labels:
                        runs = []
                        for s in range(args.seeds):
                            env, _ = make_env(d, soh=soh, soc0=soc0, complexity=c)
                            if label == 'QoS-EPD':
                                pol = EPD(nominal_robot(cfg), cfg)
                            elif label == 'fast':
                                pol = FixedSpeed(nominal_robot(cfg), cfg, args.v_fast)
                            else:
                                pol = BatteryAware(nominal_robot(cfg), cfg, v_fast=args.v_fast)
                            runs.append(run(env, pol, seed=3000 + s))
                        res[label] = mean_runs(runs)

                    def cell(r):
                        km = '' if r['completed'] >= 1.0 else f" ({r['distance_km']:.1f} km)"
                        return (f"{100 * r['completed']:3.0f}% {r['time_s'] / 60:6.1f} "
                                f"{r['final_soc']:5.3f} {r['qos_level_pct']:5.1f}%{km}").rjust(30)

                    print(f"{d / 1000:5.0f}k {soh:4.2f} {soc0:4.2f} {c:2.0f} | "
                          + ' | '.join(cell(res[l]) for l in labels))
            print()


if __name__ == '__main__':
    main()
