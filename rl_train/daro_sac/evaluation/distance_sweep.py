"""Evaluate DARO / baselines for several mission distances.

    python -m evaluation.distance_sweep --baselines-only
    python -m evaluation.distance_sweep --ckpt ./checkpoints/seed_42 --distances 500 1000 2500 5000

Writes logs/distance_sweep.json (full results) and logs/distance_sweep.csv
(one row per distance x policy, overall means + lifetime).
"""

from __future__ import annotations

import argparse
import csv
import os
import time

from . import (EVAL_CONFIG, make_baselines, evaluate_policies, save_results,
               load_daro, lifetime_from_results)
from .runner import COLS, format_header, format_row

DISTANCES_M = [500.0, 1000.0, 2500.0, 5000.0]


def sweep(policies_factory, config: dict, distances, n_episodes=None) -> dict:
    out = {}
    for d in distances:
        cfg = dict(config, mission_distance_m=float(d))
        t0 = time.time()
        res = evaluate_policies(policies_factory(cfg), cfg, n_episodes=n_episodes, verbose=False)
        life = lifetime_from_results(res, cfg)
        out[float(d)] = {'results': res, 'lifetime': life}
        print(f"[distance {d:.0f} m] done in {time.time() - t0:.0f} s")
    return out


def print_sweep(sweep_res: dict, config: dict):
    cols = COLS + [('missions_to_eol', 'Miss2EOL', '{:8.0f}'), ('km_to_eol', 'km2EOL', '{:7.0f}')]
    for d, r in sweep_res.items():
        print(f"\n=== Mission {d:.0f} m (overall mean; lifetime = mean over complexity) ===")
        print(format_header(cols=cols))
        for pol, pr in r['results'].items():
            life = r['lifetime'][pol].values()
            m = dict(pr['overall'],
                     missions_to_eol=sum(v['missions_to_eol'] for v in life) / len(life),
                     km_to_eol=sum(v['km_to_eol'] for v in life) / len(life))
            print(format_row(pol, m, cols))
    # Relative to AS-MF per distance
    print("\n=== Relative to AS-MF (energy %, mission time %, km to EOL %) ===")
    for d, r in sweep_res.items():
        ref = r['results']['AS-MF']['overall']
        ref_km = sum(v['km_to_eol'] for v in r['lifetime']['AS-MF'].values())
        line = f"{d:6.0f} m "
        for pol, pr in r['results'].items():
            if pol == 'AS-MF':
                continue
            o = pr['overall']
            km = sum(v['km_to_eol'] for v in r['lifetime'][pol].values())
            line += (f" | {pol}: E {100 * (o['energy_kj'] / ref['energy_kj'] - 1):+5.1f}%"
                     f" T {100 * (o['mission_time_s'] / ref['mission_time_s'] - 1):+5.1f}%"
                     f" km {100 * (km / ref_km - 1):+5.1f}%")
        print(line)
    print()


def write_csv(sweep_res: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = [k for k, _, _ in COLS] + ['avg_speed_mps', 'peak_temp_c']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['distance_m', 'policy'] + keys + ['missions_to_eol', 'km_to_eol'])
        for d, r in sweep_res.items():
            for pol, pr in r['results'].items():
                life = list(r['lifetime'][pol].values())
                w.writerow([d, pol] + [pr['overall'][k] for k in keys]
                           + [sum(v['missions_to_eol'] for v in life) / len(life),
                              sum(v['km_to_eol'] for v in life) / len(life)])
    print(f'CSV saved -> {path}')


def main():
    ap = argparse.ArgumentParser(description='DARO mission-distance sweep')
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--baselines-only', action='store_true')
    ap.add_argument('--distances', type=float, nargs='+', default=DISTANCES_M)
    ap.add_argument('--episodes', type=int, default=5, help='Seeds per scenario')
    ap.add_argument('--log-dir', type=str, default=EVAL_CONFIG['log_dir'])
    args = ap.parse_args()

    config = dict(EVAL_CONFIG)
    agent = None
    if not args.baselines_only:
        if not args.ckpt:
            raise SystemExit('Pass --ckpt <dir> or --baselines-only.')
        agent = load_daro(args.ckpt, config)

    def factory(cfg):
        pols = {} if agent is None else {'DARO': agent}
        pols.update(make_baselines(cfg))
        return pols

    res = sweep(factory, config, args.distances, n_episodes=args.episodes)
    print_sweep(res, config)
    save_results({'config': config, 'sweep': {str(k): v for k, v in res.items()}},
                 os.path.join(args.log_dir, 'distance_sweep.json'))
    write_csv(res, os.path.join(args.log_dir, 'distance_sweep.csv'))


if __name__ == '__main__':
    main()
