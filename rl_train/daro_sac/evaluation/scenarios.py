"""Claim-oriented test suite: "finish fast subject to charge and QoS, adapting to
model mismatch, workload changes and battery degradation".

Families (all in the training distribution: <= 5 km, SOH 0.8-1.0):

  nominal      1 km, full charge, SOH {1, 0.9, 0.8} x C {1, 2, 3}: speed with ample charge.
  tight        5 km, SOH {1, 0.8} x C {1, 2, 3} x k {0.9, 1.0, 1.1, 1.25}: SOC0 such that
               the usable energy above the 5 % reserve is k x the energy-optimal mission
               energy (robot_env.tight_soc_init). k = 1.25: fast is affordable; k ~ 1:
               energy-optimal only; k = 0.9: the reserve must be used.
  switch       3 km, full charge, complexity re-drawn from {1, 2, 3} with mean period
               {10, 30, 120} s (start C = 2).
  mismatch     2 km, full charge, true plant != nominal model (payload, terrain, compute),
               C {1, 2, 3}. Rule-based baselines keep the nominal model.
  tight_mismatch  5 km, charge-limited as in 'tight' but for the TRUE plant (as training's
               tight episodes): SOH {1, 0.8} x C {1, 2, 3} x k {1.0, 1.1, 1.25} x PLANTS.
               Rule-based baselines budget charge with the nominal model, so model error
               flips their fast / optimal decision (overspend -> reserve / failure, or
               underspend -> needlessly slow).
  degradation  "Same mission, same charge, ageing battery": 5 km, SOC0 fixed per C so that
               at SOH 0.8 energy-optimal driving ends at the reserve (k_eol = 1), SOH swept
               1.0 -> 0.8. Expected battery-aware behaviour: fast while young, energy-optimal
               near end of life, always completing.

Metrics per scenario: return, completion, mission time, energy, QoS violation (% steps),
final SOC, reserve violation (SOC_end < 5 %).

    python -m evaluation.scenarios --baselines-only                       # ~3 min
    python -m evaluation.scenarios --ckpts checkpoints/seed_{0..4} checkpoints/rl_soc/seed_{0..4}
    python -m evaluation.scenarios --families tight degradation --episodes 2 --baselines-only
"""

from __future__ import annotations

if __name__ == '__main__' and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = 'evaluation'
    import evaluation  # noqa: F401

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from . import EVAL_CONFIG, make_baselines, load_rl_policies
from .baselines import FastQoS
from .test_env import make_robot
from .metrics import run_episode, aggregate
from robot_env import RobotEnv, RobotParams, ECMParams, StressParams, tight_soc_init

FAMILIES = ('nominal', 'tight', 'switch', 'mismatch', 'tight_mismatch', 'degradation')
METRICS = ('return', 'completed', 'mission_time_s', 'energy_kj', 'qos_viol_pct',
           'final_soc', 'reserve_viol')

_NOM = RobotParams()
PLANTS = {   # true-plant overrides (robot_env.PLANT_KEYS), cf. unit_test/check_model_mismatch.py
    'payload+50%': {'motor_base_w': 1.5 * _NOM.motor_base_w,
                    'motor_linear_w_per_mps': 1.5 * _NOM.motor_linear_w_per_mps,
                    'motor_cubic_w_per_mps3': 1.5 * _NOM.motor_cubic_w_per_mps3},
    'rough_v3x2': {'motor_cubic_w_per_mps3': 2.0 * _NOM.motor_cubic_w_per_mps3},
    'smooth_v3x0.7': {'motor_cubic_w_per_mps3': 0.7 * _NOM.motor_cubic_w_per_mps3},
    'compute_x1.3': {'compute_scale': 1.3},
    'compute_x0.8': {'compute_scale': 0.8},
}


@dataclass
class Scenario:
    family: str
    name: str
    distance_m: float
    soh: float
    soc0: float
    complexity: float
    plant: dict = field(default_factory=dict)
    switch_period_s: float = 0.0
    tags: dict = field(default_factory=dict)   # grouping keys for the printout

    def options(self) -> dict:
        return {'plant': self.plant, 'complexity_switch_period_s': self.switch_period_s,
                'complexity_levels': [1.0, 2.0, 3.0]}


def build_suite(families=FAMILIES, k_eol: float = 1.0) -> List[Scenario]:
    out: List[Scenario] = []
    if 'nominal' in families:
        for soh in (1.0, 0.9, 0.8):
            for c in (1.0, 2.0, 3.0):
                out.append(Scenario('nominal', f'soh{soh:.2f}_c{c:g}', 1000.0, soh, 1.0, c,
                                    tags={'soh': soh, 'c': c}))
    if 'tight' in families:
        for soh in (1.0, 0.8):
            for c in (1.0, 2.0, 3.0):
                for k in (0.9, 1.0, 1.1, 1.25):
                    soc0 = tight_soc_init({}, soh, c, 5000.0, k)
                    out.append(Scenario('tight', f'soh{soh:.2f}_c{c:g}_k{k:g}', 5000.0, soh,
                                        soc0, c, tags={'k': k, 'soh': soh, 'c': c}))
    if 'switch' in families:
        for period in (10.0, 30.0, 120.0):
            out.append(Scenario('switch', f'period{period:g}s', 3000.0, 1.0, 1.0, 2.0,
                                switch_period_s=period, tags={'period_s': period}))
    if 'mismatch' in families:
        for pname, plant in PLANTS.items():
            for c in (1.0, 2.0, 3.0):
                out.append(Scenario('mismatch', f'{pname}_c{c:g}', 2000.0, 1.0, 1.0, c,
                                    plant=plant, tags={'plant': pname, 'c': c}))
    if 'tight_mismatch' in families:
        for pname, plant in PLANTS.items():
            for soh in (1.0, 0.8):
                for c in (1.0, 2.0, 3.0):
                    for k in (1.0, 1.1, 1.25):
                        soc0 = tight_soc_init(plant, soh, c, 5000.0, k)
                        out.append(Scenario('tight_mismatch', f'{pname}_soh{soh:.2f}_c{c:g}_k{k:g}',
                                            5000.0, soh, soc0, c, plant=plant,
                                            tags={'plant': pname, 'k': k, 'soh': soh, 'c': c}))
    if 'degradation' in families:
        for c in (1.0, 2.0, 3.0):
            soc0 = tight_soc_init({}, 0.8, c, 5000.0, k_eol)
            for soh in np.round(np.arange(1.0, 0.8 - 1e-9, -0.025), 3):
                out.append(Scenario('degradation', f'c{c:g}_soh{soh:.3f}', 5000.0, float(soh),
                                    soc0, c, tags={'soh': float(soh), 'c': c, 'soc0': soc0}))
    return out


def make_env(config: dict, sc: Scenario) -> RobotEnv:
    robot = make_robot(dict(config, mission_distance_m=sc.distance_m))
    return RobotEnv(robot=robot, ecm=ECMParams.default(robot), stress=StressParams(),
                    soc_init=sc.soc0, soh_init=sc.soh, complexity=sc.complexity, mode='test',
                    complexity_jitter=config['test_complexity_jitter'])


def evaluate(policies: Dict[str, object], suite: List[Scenario], config: dict,
             n_episodes: int, verbose: bool = True) -> dict:
    """res[policy][scenario name] = aggregate over seeds (common seeds across policies)."""
    res: Dict[str, Dict[str, dict]] = {p: {} for p in policies}
    for k, sc in enumerate(suite):
        t0 = time.time()
        for pname, pol in policies.items():
            env = make_env(config, sc)
            eps = [run_episode(env, pol, seed=config['seed'] + 1000 * k + i,
                               charge_c_rate=config['charge_c_rate'], options=sc.options())
                   for i in range(n_episodes)]
            res[pname][sc.name if sc.family == 'nominal' else f'{sc.family}/{sc.name}'] = \
                dict(family=sc.family, tags=sc.tags, soc0=sc.soc0, **aggregate(eps))
        if verbose:
            print(f'  [{k + 1:3d}/{len(suite)}] {sc.family}/{sc.name} ({time.time() - t0:.0f} s)')
    return res


def add_group_means(res: dict, groups: Dict[str, List[str]]) -> dict:
    """'<method>' = mean over its seed checkpoints (metric_std = across-seed spread)."""
    out = {}
    for label, names in groups.items():
        if len(names) < 2:
            continue
        out[label] = {}
        for sname, row in res[names[0]].items():
            m = dict(family=row['family'], tags=row['tags'], soc0=row['soc0'])
            for key in METRICS:
                v = np.array([res[n][sname][key] for n in names])
                m[key], m[key + '_std'] = float(v.mean()), float(v.std())
            out[label][sname] = m
    return {**out, **res}


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float('nan')


def print_report(res: dict, families, show: List[str]):
    head = f"{'policy':<14} {'R':>8} {'done%':>6} {'T[min]':>7} {'E[kJ]':>7} {'QoSv%':>6} " \
           f"{'SOC_end':>7} {'res.viol%':>9}"

    def line(name, rows):
        return (f"{name:<14} {_mean(rows, 'return'):8.3f} {100 * _mean(rows, 'completed'):6.0f} "
                f"{_mean(rows, 'mission_time_s') / 60:7.1f} {_mean(rows, 'energy_kj'):7.2f} "
                f"{_mean(rows, 'qos_viol_pct'):6.1f} {_mean(rows, 'final_soc'):7.3f} "
                f"{100 * _mean(rows, 'reserve_viol'):9.0f}")

    group_key = {'nominal': 'soh', 'tight': 'k', 'switch': 'period_s', 'mismatch': 'plant',
                 'tight_mismatch': 'plant', 'degradation': 'soh'}
    for fam in families:
        print(f'\n=== {fam}: mean over scenarios ===')
        print(head)
        for p in show:
            print(line(p, [r for r in res[p].values() if r['family'] == fam]))
        key = group_key[fam]
        vals = sorted({r['tags'][key] for r in res[show[0]].values() if r['family'] == fam},
                      key=lambda x: (-x if isinstance(x, float) and key == 'soh' else x))
        print(f'--- by {key}')
        for v in vals:
            print(f'{key} = {v}')
            for p in show:
                print('  ' + line(p, [r for r in res[p].values()
                                      if r['family'] == fam and r['tags'][key] == v]))
    print()


def write_csv(res: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = [x for k in METRICS for x in (k, k + '_std')]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['policy', 'family', 'scenario', 'soc0', 'tags'] + cols)
        for p, rows in res.items():
            for sname, r in rows.items():
                w.writerow([p, r['family'], sname, r['soc0'], json.dumps(r['tags'])]
                           + [r.get(k, float('nan')) for k in cols])
    print(f'CSV saved -> {path}')


def main():
    ap = argparse.ArgumentParser(description='Claim-oriented scenario evaluation')
    ap.add_argument('--ckpts', nargs='+', default=None, help='RL checkpoint directories')
    ap.add_argument('--baselines-only', action='store_true')
    ap.add_argument('--families', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    ap.add_argument('--episodes', type=int, default=3, help='seeds per scenario')
    ap.add_argument('--k-eol', type=float, default=1.0,
                    help='degradation family: energy factor at SOH 0.8 (1 = energy-optimal ends at the reserve)')
    ap.add_argument('--no-asmf', action='store_true', help='drop the AS-MF baseline')
    ap.add_argument('--log-dir', default=EVAL_CONFIG['log_dir'])
    args = ap.parse_args()

    config = dict(EVAL_CONFIG)
    policies, groups = {}, {}
    if not args.baselines_only:
        if not args.ckpts:
            raise SystemExit('Pass --ckpts <dir> ... or --baselines-only.')
        policies, groups = load_rl_policies(args.ckpts, config)
    base = make_baselines(config)
    if args.no_asmf:
        base.pop('AS-MF', None)
    base['Fast-QoS'] = FastQoS(make_robot(config), config)
    policies.update(base)

    suite = build_suite(args.families, args.k_eol)
    print(f'{len(suite)} scenarios x {len(policies)} policies x {args.episodes} seeds')
    res = evaluate(policies, suite, config, args.episodes)
    res = add_group_means(res, groups)
    show = [g for g in groups if len(groups[g]) > 1] + \
           [p for p in policies if not any(p in n and len(n) > 1 for n in groups.values())]
    print_report(res, args.families, show)
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, 'scenarios.json'), 'w') as f:
        json.dump({'config': config, 'families': args.families, 'k_eol': args.k_eol,
                   'results': res}, f, indent=2, default=float)
    write_csv(res, os.path.join(args.log_dir, 'scenarios.csv'))


if __name__ == '__main__':
    main()
