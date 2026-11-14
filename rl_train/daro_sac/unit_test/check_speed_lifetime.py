"""Energy-optimal speed and missions to EOL vs mission distance.

For each mission distance, runs constant-speed policies (CPU at the lowest
frequency that meets QoS, as in check_battery_sensitivity.py) and reports

  * the speed with the lowest mission energy (cell side, robot_env.py),
  * the number of missions until SOH_EOL (default 0.8) at each speed,
  * final SOC after one mission (fresh battery, and at SOH_EOL),
  * final SOH after one mission + recharge, starting from SOH = 1.

Missions to EOL use the same method as evaluation/lifetime.py: every mission
starts fully charged (SOC0 = 1) and is followed by a recharge; the per-cycle
Xu fade fd_cycle (evaluation/metrics.xu_mission_fd) is measured at several SOH
levels and integrated from SOH = 1 down to SOH_EOL (missions_to_eol).

    python rl_train/daro_sac/unit_test/check_speed_lifetime.py
    python rl_train/daro_sac/unit_test/check_speed_lifetime.py --distances 500 1000 5000 --complexity 2
    python rl_train/daro_sac/unit_test/check_speed_lifetime.py --plot speed_lifetime.png --csv speed_lifetime.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl_train/daro_sac

import numpy as np

import evaluation  # noqa: F401  (sets sys.path)
from evaluation import EVAL_CONFIG, make_test_env, TestScenario
from evaluation.lifetime import missions_to_eol
from evaluation.metrics import run_episode
from robot_env import throughput_freq_required

SOH_GRID = [1.0, 0.95, 0.9, 0.85, 0.8]


class ConstantPolicy:
    def __init__(self, v, f):
        self.a = np.array([v, f], dtype=np.float32)

    def select_action(self, obs, deterministic=True):
        return self.a


def qos_frequency(env, v, complexity, margin=1.05):
    """Lowest CPU frequency meeting QoS at speed v (clipped to the CPU range)."""
    q = max(env.robot.qos_ref) * margin
    f = float(throughput_freq_required(v, q, complexity))
    return float(np.clip(f, env.robot.f_min, env.robot.f_max)), f <= env.robot.f_max + 1e-9


def evaluate_speed(distance, v, complexity, charge_c_rate, soh_eol, seed=1):
    """One constant speed: energy, final SOC/SOH at SOH 1 and missions/km to SOH_EOL.

    Runs that stop early (battery empty or voltage/current limit) are kept with
    completed = False; missions_to_eol is NaN unless every SOH level completes.
    """
    cfg = dict(EVAL_CONFIG, mission_distance_m=distance, test_complexity_jitter=0.0)
    grid = sorted(set(SOH_GRID) | {soh_eol}, reverse=True)
    runs = {}
    for soh in grid:
        env = make_test_env(cfg, TestScenario(soh, 1.0, complexity))
        f, qos_ok = qos_frequency(env, v, complexity)
        m = run_episode(env, ConstantPolicy(v, f), seed=seed, charge_c_rate=charge_c_rate)
        runs[soh] = dict(m, f=f, qos_ok=qos_ok, final_soc=float(env.battery.soc))
    first, eol = runs[grid[0]], runs[soh_eol]
    done_all = all(r['completed'] >= 1.0 for r in runs.values())
    n = (missions_to_eol(grid, [runs[h]['fd_cycle'] for h in grid], soh_eol)
         if done_all else float('nan'))
    return {
        'distance_m': distance, 'speed_mps': v, 'cpu_ghz': first['f'], 'qos_ok': first['qos_ok'],
        'completed': bool(first['completed'] >= 1.0), 'completed_eol': bool(eol['completed'] >= 1.0),
        'energy_kj': first['energy_kj'], 'wh_per_km': first['wh_per_km'],
        'mission_time_s': first['mission_time_s'],
        'final_soc': first['final_soc'],            # one mission, fresh battery (SOH 1, SOC0 1)
        'final_soc_eol': eol['final_soc'],          # one mission at SOH_EOL
        'final_soh': 1.0 - first['delta_soh_ppm'] * 1e-6,   # after mission + recharge, from SOH 1
        'delta_soh_ppm': first['delta_soh_ppm'],
        'stress_proxy': first['stress_proxy'], 'peak_temp_c': first['peak_temp_c'],
        'fd_cycle_soh1': first['fd_cycle'], 'missions_to_eol': n,
        'km_to_eol': n * distance / 1000.0,
    }


def plot(rows, distances, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for d in distances:
        r = [x for x in rows if x['distance_m'] == d and x['completed']]
        if not r:
            continue
        v = [x['speed_mps'] for x in r]
        line, = axes[0].plot(v, [x['wh_per_km'] for x in r], label=f'{d:.0f} m')
        c = line.get_color()
        axes[1].plot(v, [x['final_soc'] for x in r], color=c)
        axes[1].plot(v, [x['final_soc_eol'] if x['completed_eol'] else np.nan for x in r],
                     '--', color=c)   # gaps: mission not completed at SOH_EOL
        axes[2].plot(v, [x['delta_soh_ppm'] for x in r], color=c)
        axes[3].plot(v, [x['missions_to_eol'] for x in r], color=c)
        best = min(r, key=lambda x: x['energy_kj'])
        axes[0].plot(best['speed_mps'], best['wh_per_km'], 'o', color=c)
    axes[0].set(xlabel='speed [m/s]', ylabel='energy [Wh/km]', title='Energy per km')
    axes[1].set(xlabel='speed [m/s]', ylabel='SOC', title='Final SOC (solid SOH 1, dashed SOH_EOL)')
    axes[2].set(xlabel='speed [m/s]', ylabel='ppm', title='SOH loss per mission (1 - final SOH)')
    axes[3].set(xlabel='speed [m/s]', ylabel='missions', title='Missions to EOL', yscale='log')
    axes[0].legend(title='mission')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f'Plot -> {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distances', type=float, nargs='+', default=[10000, 20000, 30000])
    ap.add_argument('--complexity', type=float, default=1.0)
    ap.add_argument('--vmin', type=float, default=0.5)
    ap.add_argument('--vmax', type=float, default=5.0)
    ap.add_argument('--dv', type=float, default=0.25)
    ap.add_argument('--soh-eol', type=float, default=0.8)
    ap.add_argument('--charge-c-rate', type=float, default=EVAL_CONFIG['charge_c_rate'])
    ap.add_argument('--csv', type=str, default=None, help='write all (distance, speed) rows')
    ap.add_argument('--plot', type=str, default=None, help='write energy / lifetime vs speed PNG')
    args = ap.parse_args()

    speeds = np.round(np.arange(args.vmin, args.vmax + 1e-9, args.dv), 3)
    rows = []
    for d in args.distances:
        for v in speeds:
            r = evaluate_speed(d, float(v), args.complexity, args.charge_c_rate, args.soh_eol)
            if r is not None:
                rows.append(r)

    print(f"Complexity {args.complexity:g}, SOC0 = 1 per mission, charge {args.charge_c_rate:g} C, "
          f"EOL at SOH {args.soh_eol:.2f}")
    hdr = (f"{'dist[m]':>7} {'optimum':>8} | {'v':>5} {'E[kJ]':>8} {'Wh/km':>6} {'SOC_end':>7} "
           f"{'SOC_end@EOL':>11} {'SOH_end':>9} {'dSOH[ppm]':>9} | {'N_EOL':>8} {'km_EOL':>8}")
    print(hdr)
    print('-' * len(hdr))

    def soc_eol(x):
        return f"{x['final_soc_eol']:11.3f}" if x['completed_eol'] else f"{'not done':>11}"

    for d in args.distances:
        all_r = [x for x in rows if x['distance_m'] == d]
        r = [x for x in all_r if x['completed']]
        if not r:
            print(f'{d:7.0f} no completed mission at any speed')
            continue
        be = min(r, key=lambda x: x['energy_kj'])                       # energy-optimal speed
        rl = [x for x in r if np.isfinite(x['missions_to_eol'])]
        bn = max(rl, key=lambda x: x['missions_to_eol']) if rl else None  # lifetime-optimal speed
        for tag, x in (('energy', be), ('lifetime', bn)):
            if x is None:
                print(f"{d:7.0f} {tag:>8} | not completed at SOH {args.soh_eol:.2f} at any speed")
                continue
            print(f"{d:7.0f} {tag:>8} | {x['speed_mps']:5.2f} {x['energy_kj']:8.3f} {x['wh_per_km']:6.3f} "
                  f"{x['final_soc']:7.3f} {soc_eol(x)} {x['final_soh']:9.6f} {x['delta_soh_ppm']:9.2f} "
                  f"| {x['missions_to_eol']:8.0f} {x['km_to_eol']:8.0f}")
            if not x['qos_ok']:
                print('                   (QoS not reachable at this speed: CPU capped at f_max)')
        failed = [x['speed_mps'] for x in all_r if not x['completed']]
        failed_eol = [x['speed_mps'] for x in all_r if x['completed'] and not x['completed_eol']]
        if failed:
            print(f"{'':16} not completed (SOH 1) at v = {failed}")
        if failed_eol:
            print(f"{'':16} not completed at SOH {args.soh_eol:.2f} at v = {failed_eol}")
    print("energy = lowest-energy speed, lifetime = speed with most missions to EOL. "
          "SOC_end: after one mission from SOC 1 (fresh battery); SOC_end@EOL: same at SOH_EOL; "
          "SOH_end: after one mission + recharge from SOH 1.")

    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'CSV -> {args.csv}')
    if args.plot:
        plot(rows, args.distances, args.plot)


if __name__ == '__main__':
    main()
