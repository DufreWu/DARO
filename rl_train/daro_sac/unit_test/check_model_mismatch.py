"""Scenario 1: model mismatch. How much does QoS-EPD lose when its calibrated
power model differs from the real robot (payload, terrain, compute hardware)?

The TRUE plant is changed (robot_env parameters); QoS-EPD keeps the nominal
calibration. 'EPD-true' is the same optimiser with the true model. 'best fixed'
is the best constant speed in hindsight (scanned on the true plant, CPU at the
QoS frequency of the estimated complexity). Oracle = the better of the two; the
gap QoS-EPD -> oracle is what a model-free learner (DARO, which observes the
actual current/voltage) could at best recover. Fixed 3.3 m/s and AS-MF are references.
Note: QoS-EPD's 20 % complexity margin also distorts its speed choice, so a
well-chosen constant speed can beat it even on the nominal plant.

    python rl_train/daro_sac/unit_test/check_model_mismatch.py
    python rl_train/daro_sac/unit_test/check_model_mismatch.py --distance 2000 --seeds 3
"""

from __future__ import annotations

import argparse

import numpy as np

from scenario_utils import (EPD, FixedSpeed, ASMFBaseline, make_env, make_config, run,
                            mean_runs, nominal_robot, true_robot)

NOMINAL = dict(motor_base_w=1.1625, motor_cubic_w_per_mps3=0.10893, compute_scale=1.0)
SCENARIOS = {   # name -> true-plant overrides (relative to NOMINAL)
    'nominal':              {},
    'payload (+50% motor)': dict(motor_base_w=1.5 * 1.1625, motor_cubic_w_per_mps3=1.5 * 0.10893),
    'rough terrain (v^3 x2)': dict(motor_cubic_w_per_mps3=2.0 * 0.10893),
    'smooth floor (v^3 x0.7)': dict(motor_cubic_w_per_mps3=0.7 * 0.10893),
    'extra sensors (comp x1.3)': dict(compute_scale=1.3),
    'efficient compute (x0.8)': dict(compute_scale=0.8),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', type=float, default=2000.0)
    ap.add_argument('--complexities', type=float, nargs='+', default=[1.0, 2.0, 3.0])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--fixed-speed', type=float, default=3.3)
    ap.add_argument('--scan', type=float, nargs=3, default=[2.0, 4.5, 0.1],
                    metavar=('VMIN', 'VMAX', 'DV'), help='hindsight constant-speed scan')
    ap.add_argument('--max-viol', type=float, default=1.0,
                    help='QoS violation %% allowed for the hindsight best fixed speed')
    args = ap.parse_args()

    cfg = make_config(args.distance)
    print(f"Model mismatch, {args.distance:.0f} m, jitter 0.2, {args.seeds} seeds. "
          f"Energy in Wh/km; v = mean speed; viol = QoS violation % of steps.")
    scan = np.round(np.arange(args.scan[0], args.scan[1] + 1e-9, args.scan[2]), 3)
    print(f"{'true plant':<26} {'C':>2} | {'QoS-EPD':>14} | {'EPD-true':>14} | "
          f"{'best fixed':>14} | {'fixed ' + format(args.fixed_speed, '.1f'):>14} | {'AS-MF':>14} "
          f"| QoS-EPD vs oracle")
    gaps = []
    for name, ov in SCENARIOS.items():
        for c in args.complexities:
            res = {}
            for label in ('QoS-EPD', 'EPD-true', 'fixed', 'AS-MF'):
                runs = []
                for s in range(args.seeds):
                    env, _ = make_env(args.distance, complexity=c, **ov)
                    if label == 'QoS-EPD':
                        pol = EPD(nominal_robot(cfg), cfg)
                    elif label == 'EPD-true':
                        pol = EPD(true_robot(env), cfg, name='EPD-true')
                    elif label == 'fixed':
                        pol = FixedSpeed(nominal_robot(cfg), cfg, args.fixed_speed)
                    else:
                        pol = ASMFBaseline(nominal_robot(cfg), cfg)
                    runs.append(run(env, pol, seed=1000 + s))
                res[label] = mean_runs(runs)
            best = None
            for v in scan:
                r = []
                for s in range(args.seeds):
                    env, _ = make_env(args.distance, complexity=c, **ov)
                    r.append(run(env, FixedSpeed(nominal_robot(cfg), cfg, float(v)), seed=1000 + s))
                r = mean_runs(r)
                ok = r['completed'] >= 1.0 and r['qos_viol_pct'] <= args.max_viol
                if ok and (best is None or r['wh_per_km'] < best['wh_per_km']):
                    best = r
            res['best fixed'] = best
            oracle = min(res['EPD-true']['wh_per_km'], best['wh_per_km'])
            gap = 100.0 * (res['QoS-EPD']['wh_per_km'] / oracle - 1.0)
            gaps.append((name, c, gap))
            cell = lambda r: f"{r['wh_per_km']:5.3f} v{r['avg_speed']:.2f} {r['qos_viol_pct']:3.0f}%"  # noqa: E731
            print(f"{name:<26} {c:2.0f} | {cell(res['QoS-EPD'])} | {cell(res['EPD-true'])} | "
                  f"{cell(res['best fixed'])} | {cell(res['fixed'])} | {cell(res['AS-MF'])} | {gap:+6.2f}%")
    g = np.array([x[2] for x in gaps if x[0] != 'nominal'])
    g0 = np.array([x[2] for x in gaps if x[0] == 'nominal'])
    print(f"\nQoS-EPD extra energy vs oracle: nominal plant mean {g0.mean():+.2f}% "
          f"(margin cost only); mismatched plants mean {g.mean():+.2f}%, max {g.max():+.2f}%")


if __name__ == '__main__':
    main()
