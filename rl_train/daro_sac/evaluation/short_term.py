"""Short-term (single-mission) energy test: complexity x distance, or complexity x SOH.

--by distance (default): SOH and initial SOC fixed (--soh, --soc), sweep the
                         mission distances (--distances).
--by soh               : distance and initial SOC fixed (--distance, --soc),
                         sweep the SOH levels (--soh-levels).
Every policy sees the same seeded complexity disturbance (common random numbers).

    python -m evaluation.short_term --ckpts ./checkpoints/seed_{0,1,2,3,4}
    python -m evaluation.short_term --by soh --ckpts ./checkpoints/seed_{0,1,2,3,4} \
        ./checkpoints/rl_soc/seed_{0,1,2,3,4} ./checkpoints/rl_soh/seed_{0,1,2,3,4}
    python -m evaluation.short_term --by soh --baselines-only --distance 1000 --soc 0.6

--ckpts takes DARO and RL-ablation checkpoints mixed; each is labelled from its
obs_variant.json (DARO / RL-SOC-only / RL-SOH-only). A method with several
checkpoints is reported per seed ("<method> s<k>") and as "<method>" = mean
over seeds (std = spread across seeds).

Outputs (in --log-dir, default ./logs); <tag> = "" for distance, "_soh" for SOH:
    short_term_energy<tag>.json   full results
    short_term_energy<tag>.csv    one row per (distance|soh) x complexity x policy
    short_term_energy<tag>.png    grouped bars vs complexity, one panel per distance/SOH
"""

from __future__ import annotations

if __name__ == '__main__' and not __package__:
    # Allow `python evaluation/<script>.py` as well as `python -m evaluation.<script>`.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = 'evaluation'
    import evaluation  # noqa: F401  (initialises the package and its sys.path)

import argparse
import csv
import json
import os
import time
from typing import Dict, List

import numpy as np

from . import EVAL_CONFIG, make_baselines, evaluate_policies, load_rl_policies

DISTANCES_M = [1000.0]
AXES = {  # outer (panel) axis: config override, CSV column, labels
    'distance': dict(key='mission_distance_m', col='distance_m',
                     panel=lambda v: f'{v:.0f} m mission', name='mission distance',
                     head=lambda v: f'Mission {v:.0f} m'),
    'soh':      dict(key='test_soh_levels', col='soh',
                     panel=lambda v: f'SOH = {v:.2f}', name='SOH',
                     head=lambda v: f'SOH {v:.2f}'),
}
METRICS = ['energy_kj', 'wh_per_km', 'mission_time_s', 'avg_speed_mps',
           'qos_viol_pct', 'completed', 'stress_proxy', 'delta_soc']


def run(policies: Dict[str, object], config: dict, values, n_episodes: int,
        by: str = 'distance') -> dict:
    """table[value][complexity][policy] = {metric: mean, metric_std: std}.

    value = mission distance (by='distance') or SOH (by='soh').
    """
    key = AXES[by]['key']
    table: Dict[float, Dict[float, Dict[str, dict]]] = {}
    for d in values:
        cfg = dict(config, **{key: [float(d)] if by == 'soh' else float(d)})
        t0 = time.time()
        res = evaluate_policies(policies, cfg, n_episodes=n_episodes, verbose=False)
        table[float(d)] = {}
        for c in cfg['test_complexity_levels']:
            row = {}
            for pol, r in res.items():
                scen = [s for s in r['scenarios'].values() if abs(s['complexity'] - c) < 1e-9]
                assert len(scen) == 1, 'expected one scenario per complexity'
                row[pol] = {k: scen[0][k] for k in scen[0]}
            table[float(d)][float(c)] = row
        print(f'[{AXES[by]["head"](d)}] done in {time.time() - t0:.0f} s')
    return table


def add_group_means(table: dict, groups: Dict[str, List[str]]):
    """Insert '<method>' = mean over its seed checkpoints (std = across-seed spread)."""
    for label, names in reversed(list(groups.items())):
        if len(names) < 2:
            continue
        for d in table:
            for c in table[d]:
                row = table[d][c]
                m = {}
                for k in METRICS:
                    v = np.array([row[n][k] for n in names])
                    m[k], m[k + '_std'] = float(v.mean()), float(v.std())
                table[d][c] = {label: m, **row}


def print_table(table: dict, ref: str = 'AS-MF', by: str = 'distance'):
    pols = list(next(iter(next(iter(table.values())).values())))
    for d, per_c in table.items():
        print(f'\n=== {AXES[by]["head"](d)}: energy [kJ] (Wh/km) | dE vs {ref} | '
              f'time [s] | QoS viol % | done % ===')
        for c, row in per_c.items():
            print(f'-- complexity {c:g}')
            r0 = row.get(ref)
            for p in pols:
                m = row[p]
                de = (f'{100 * (m["energy_kj"] / r0["energy_kj"] - 1):+6.1f}%'
                      if r0 and p != ref else '      -')
                print(f'  {p:<10} {m["energy_kj"]:7.2f} ({m["wh_per_km"]:5.2f})  {de}  '
                      f'{m["mission_time_s"]:6.0f}  {m["qos_viol_pct"]:5.1f}  '
                      f'{100 * m["completed"]:4.0f}')
    print()


def write_csv(table: dict, path: str, by: str = 'distance'):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = [x for k in METRICS for x in (k, k + '_std')]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([AXES[by]['col'], 'complexity', 'policy'] + cols)
        for d, per_c in table.items():
            for c, row in per_c.items():
                for p, m in row.items():
                    w.writerow([d, c, p] + [m.get(k, float('nan')) for k in cols])
    print(f'CSV saved -> {path}')


def plot(table: dict, path: str, show: List[str], by: str = 'distance', subtitle: str = ''):
    """One panel per outer value (SOH or distance); grouped bars inside.

    x = workload complexity, one bar per policy, error bar = std.
    --by soh      : energy per mission [kJ] (distance is fixed).
    --by distance : energy per km [Wh/km] (comparable across distances).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_rewards import SERIES, TEXT, TEXT_2, SURFACE, _style

    metric, ylabel = (('energy_kj', 'Energy per mission [kJ]') if by == 'soh'
                      else ('wh_per_km', 'Energy per km [Wh/km]'))
    outer = list(table)
    fig, axes = plt.subplots(1, len(outer), figsize=(3.4 * len(outer) + 0.6, 4.0),
                             facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    n = len(show)
    width = 0.8 / n
    for ax, d in zip(axes, outer):
        _style(ax)
        ax.grid(False, axis='x')
        cs = list(table[d])
        x0 = np.arange(len(cs))
        for k, pol in enumerate(show):
            y = np.array([table[d][c][pol][metric] for c in cs])
            e = np.array([table[d][c][pol].get(metric + '_std', 0.0) for c in cs])
            ax.bar(x0 + (k - (n - 1) / 2) * width, y, width,
                   color=SERIES[k % len(SERIES)], edgecolor=SURFACE, linewidth=1.0,
                   yerr=e, error_kw=dict(ecolor=TEXT_2, elinewidth=0.9, capsize=2),
                   label=pol, zorder=3)
        ax.set_xticks(x0)
        ax.set_xticklabels([f'{c:g}' for c in cs])
        ax.set_xlabel('Workload complexity')
        ax.set_title(AXES[by]['panel'](d), loc='left', fontsize=10, color=TEXT)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(ylabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8.5, frameon=False, labelcolor=TEXT,
               loc='lower center', ncol=n, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f'Short-term energy by workload complexity and {AXES[by]["name"]} '
                 f'(mean \u00b1 std){subtitle}', x=0.01, ha='left', fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f'Plot saved -> {path}')


def main():
    ap = argparse.ArgumentParser(description='Short-term energy: complexity x distance / SOH')
    ap.add_argument('--by', choices=list(AXES), default='distance',
                    help='Panel axis: mission distance or SOH')
    ap.add_argument('--ckpts', type=str, nargs='+', default=[],
                    help='DARO / RL-SOC-only / RL-SOH-only checkpoint directories (mixed)')
    ap.add_argument('--baselines-only', action='store_true')
    ap.add_argument('--distances', type=float, nargs='+', default=DISTANCES_M)
    ap.add_argument('--complexity', type=float, nargs='+', default=None,
                    help='Complexity levels (default: EVAL_CONFIG levels)')
    ap.add_argument('--soh', type=float, default=1.0, help='Fixed SOH (--by distance)')
    ap.add_argument('--soh-levels', type=float, nargs='+',
                    default=EVAL_CONFIG['test_soh_levels'], help='SOH levels (--by soh)')
    ap.add_argument('--distance', type=float, default=EVAL_CONFIG['mission_distance_m'],
                    help='Fixed mission distance [m] (--by soh)')
    ap.add_argument('--soc', type=float, default=1.0, help='Initial SOC')
    ap.add_argument('--episodes', type=int, default=10, help='Seeds per cell')
    ap.add_argument('--log-dir', type=str, default=EVAL_CONFIG['log_dir'])
    args = ap.parse_args()
    if not args.baselines_only and not args.ckpts:
        raise SystemExit('Pass --ckpts <dir> [...] or --baselines-only.')
    config = dict(EVAL_CONFIG, test_soh_levels=[args.soh], test_soc_levels=[args.soc],
                  mission_distance_m=args.distance)
    if args.complexity:
        config['test_complexity_levels'] = list(args.complexity)

    policies, groups = {}, {}
    if not args.baselines_only:
        policies, groups = load_rl_policies(args.ckpts, config)
    policies.update(make_baselines(config))

    by = args.by
    values = args.distances if by == 'distance' else args.soh_levels
    table = run(policies, config, values, args.episodes, by)
    add_group_means(table, groups)
    print_table(table, by=by)

    os.makedirs(args.log_dir, exist_ok=True)
    base = os.path.join(args.log_dir, 'short_term_energy' + ('' if by == 'distance' else f'_{by}'))
    subtitle = (f', SOH {args.soh:.2f}, SOC0 {args.soc:.2f}' if by == 'distance'
                else f', {args.distance:.0f} m, SOC0 {args.soc:.2f}')
    with open(base + '.json', 'w') as f:
        json.dump({'config': config, 'args': vars(args),
                   'table': {str(d): {str(c): row for c, row in per_c.items()}
                             for d, per_c in table.items()}}, f, indent=2, default=float)
    print(f'Results saved -> {base}.json')
    write_csv(table, base + '.csv', by)
    order = ['DARO', 'RL-SOC-only', 'RL-SOH-only', 'DARO w/o D_d', 'DARO w/o E_d']
    show = sorted(groups, key=lambda g: order.index(g) if g in order else 9) + list(make_baselines(config))
    plot(table, base + '.png', show, by, subtitle)


if __name__ == '__main__':
    main()
