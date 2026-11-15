"""Scenario 2: workload complexity changes during the mission.

The nominal complexity switches every `period` seconds to a random level in
{1, 2, 3} (seeded, identical for all policies), on top of the per-step ±20 %
jitter. Policies (all use the true power model):

  QoS-EPD        online EMA complexity estimate + 20 % jitter margin
  oracle-safe    knows the current nominal level, keeps the 20 % jitter margin
  oracle-tight   knows the current nominal level, no jitter margin (accepts jitter violations)
  fixed 3.3      constant speed, CPU frequency from the EMA estimate
  AS-MF          2.5 m/s, f_max

QoS-EPD -> oracle-safe isolates the cost of not knowing the level (estimation
lag/noise); oracle-safe -> oracle-tight is the price of the safety margin.
A learner sees only throughput (like QoS-EPD) but can learn a better risk trade-off.

    python unit_test/check_varying_complexity.py
    python unit_test/check_varying_complexity.py --periods 0 60 10 --seeds 5
"""

from __future__ import annotations

import argparse

import numpy as np

from scenario_utils import (EPD, FixedSpeed, ASMFBaseline, make_env, make_config, run,
                            mean_runs, nominal_robot)

LEVELS = (1.0, 2.0, 3.0)


def schedule(seed, n_steps, period):
    """Nominal complexity per step; period 0 = one random level for the whole mission."""
    rng = np.random.default_rng(10_000 + seed)
    if period <= 0:
        return np.full(n_steps, rng.choice(LEVELS))
    n_seg = int(np.ceil(n_steps / period))
    return np.repeat(rng.choice(LEVELS, size=n_seg), period)[:n_steps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', type=float, default=3000.0)
    ap.add_argument('--periods', type=int, nargs='+', default=[0, 120, 30, 10],
                    help='seconds between complexity switches (0 = constant per mission)')
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--fixed-speed', type=float, default=3.3)
    args = ap.parse_args()

    cfg = make_config(args.distance)
    labels = ('QoS-EPD', 'oracle-safe', 'oracle-tight', 'fixed', 'AS-MF')
    print(f"Varying complexity, {args.distance:.0f} m, jitter 0.2, {args.seeds} seeds. "
          f"Cells: Wh/km  QoS violation %  mean speed")
    print(f"{'period':>7} | " + ' | '.join(f"{l:>17}" for l in labels)
          + ' | EPD vs oracle-safe | EPD vs oracle-tight')
    for period in args.periods:
        res = {}
        for label in labels:
            runs = []
            for s in range(args.seeds):
                env, _ = make_env(args.distance)
                sched = schedule(s, env.robot.max_steps + 1, period)
                if label == 'QoS-EPD':
                    pol = EPD(nominal_robot(cfg), cfg)
                elif label == 'oracle-safe':
                    pol = EPD(nominal_robot(cfg), cfg, known_c=float(sched[0]))
                elif label == 'oracle-tight':
                    pol = EPD(nominal_robot(cfg), cfg, known_c=float(sched[0]), jitter_margin=False)
                elif label == 'fixed':
                    pol = FixedSpeed(nominal_robot(cfg), cfg, args.fixed_speed)
                else:
                    pol = ASMFBaseline(nominal_robot(cfg), cfg)

                def hook(env, k, sched=sched, pol=pol):
                    env.complexity = float(sched[k])
                    if getattr(pol, 'known_c', None) is not None:
                        pol.known_c = float(sched[k])

                runs.append(run(env, pol, seed=2000 + s, hook=hook))
            res[label] = mean_runs(runs)
        cell = lambda r: f"{r['wh_per_km']:5.3f} {r['qos_viol_pct']:5.1f}% v{r['avg_speed']:.2f}"  # noqa: E731
        g1 = 100.0 * (res['QoS-EPD']['wh_per_km'] / res['oracle-safe']['wh_per_km'] - 1.0)
        g2 = 100.0 * (res['QoS-EPD']['wh_per_km'] / res['oracle-tight']['wh_per_km'] - 1.0)
        tag = 'const' if period <= 0 else f'{period} s'
        print(f"{tag:>7} | " + ' | '.join(cell(res[l]) for l in labels)
              + f" | {g1:+17.2f}% | {g2:+17.2f}%")


if __name__ == '__main__':
    main()
