"""Comprehensive benchmark of the proposed Lag-RLS against rule-based baselines, ablations
and informed references on the scenario suite (evaluation/scenarios.py).

Main setting (all methods get the same information): battery_obs='bms' (SOC from a
rated-capacity coulomb counter, SOH unknown) + battery_realism (sensor noise, BMS gain
error, per-episode cell variation). References with full battery information run on
battery_obs='true' with the same realism; the oracles read the true plant / workload.

Per (policy, scenario, seed) it records the short-term metrics (return, completion,
time, energy, QoS violation, reserve violation, final SOC), the long-term metric (Xu
dSOH per mission) and the decision time. The report gives per-family means with 95 %
bootstrap CIs, paired differences against a reference method, and the km to end of life
extrapolated from the degradation family (evaluation/lifetime.missions_to_eol).

    python -m evaluation.benchmark --episodes 10 --out logs/benchmark
    python -m evaluation.benchmark --report-only --out logs/benchmark
"""

from __future__ import annotations

if __name__ == '__main__' and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = 'evaluation'
    import evaluation  # noqa: F401

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from . import EVAL_CONFIG
from methods import ASMFBaseline, QoSAwareEPD, BatteryAwareEPD, BAEPDSohEstimate, LagRLSEPD
from methods.degradation_aware import DegAwareLagRLS
from .lifetime import missions_to_eol
from .metrics import run_episode
from methods.oracle import LagrangianOracle, LAM_GRID
from .scenarios import FAMILIES, build_suite, make_env
from envs.test_env import make_robot

BMS = dict(battery_obs='bms', battery_realism=True)
FULL = dict(battery_obs='true', battery_realism=True)
# name -> (class, config overrides incl. environment setting, group)
POLICIES = {
    'AS-MF':               (ASMFBaseline, BMS, 'baseline'),
    'QoS-EPD':             (QoSAwareEPD, BMS, 'baseline'),
    'BA-EPD':              (BatteryAwareEPD, BMS, 'baseline'),          # assumes SOH = 1
    'BA-EPD+SOH':          (BAEPDSohEstimate, BMS, 'baseline'),
    'Lag-RLS':             (LagRLSEPD, BMS, 'baseline'),               # short-term objective
    'DA-Lag-RLS (w=1000)': (DegAwareLagRLS, dict(BMS, deg_weight=1000.0), 'proposed'),
    'DA-Lag-RLS (w=4000)': (DegAwareLagRLS, dict(BMS, deg_weight=4000.0), 'proposed'),
    'DA-Lag-RLS (w=16000)': (DegAwareLagRLS, dict(BMS, deg_weight=16000.0), 'proposed'),
    'Lag-RLS w/o SOH est': (LagRLSEPD, dict(BMS, lag_soh='obs'), 'ablation'),
    'Lag-RLS w/o lambda':  (LagRLSEPD, dict(BMS, lag_lambda=False), 'ablation'),
    'Lag-RLS tier select': (LagRLSEPD, dict(BMS, lag_select='tier'), 'ablation'),
    'Lag-RLS w/o RLS':     (LagRLSEPD, dict(BMS, lag_rls=False), 'ablation'),
    'Lag-RLS w/o probe':   (LagRLSEPD, dict(BMS, lag_probe=False), 'ablation'),
    'Lag-RLS w/o eta':     (LagRLSEPD, dict(BMS, lag_eta=False), 'ablation'),
    'Lag-RLS w/o probe+eta': (LagRLSEPD, dict(BMS, lag_probe=False, lag_eta=False), 'ablation'),
    'Lag-RLS w/o CD':      (LagRLSEPD, dict(BMS, lag_cd=False), 'ablation'),
    'Lag-RLS w/o QoS cost': (LagRLSEPD, dict(BMS, lag_qos='allowance'), 'ablation'),
    # extra-sensor variants (run separately: --policies ... --out logs/paper/benchmark_sensing)
    'Lag-RLS split power': (LagRLSEPD, dict(BMS, lag_power='split'), 'variant'),
    'Lag-RLS event count': (LagRLSEPD, dict(BMS, lag_c_source='events'), 'variant'),
    'Lag-RLS split+events': (LagRLSEPD, dict(BMS, lag_power='split', lag_c_source='events'), 'variant'),
    'BA-EPD (full info)':  (BatteryAwareEPD, FULL, 'reference'),
    'Lag-RLS (full info)': (LagRLSEPD, dict(FULL, lag_soh='obs'), 'reference'),
    'Oracle-obs':          ('oracle', dict(BMS, observe_c=True), 'reference'),
    'Oracle':              ('oracle', dict(BMS, observe_c=False), 'reference'),
}
METRICS = ('return', 'completed', 'mission_time_s', 'energy_kj', 'qos_viol_pct',
           'reserve_viol', 'final_soc', 'delta_soh_ppm', 'fd_cycle')


class _Timed:
    """Wraps a policy and accumulates select_action wall time."""

    def __init__(self, pol):
        self.pol, self.t, self.n = pol, 0.0, 0

    def reset(self):
        if hasattr(self.pol, 'reset'):
            self.pol.reset()

    def select_action(self, obs, deterministic=True):
        t0 = time.perf_counter()
        a = self.pol.select_action(obs, deterministic)
        self.t += time.perf_counter() - t0
        self.n += 1
        return a


def _episode(name, sc, k, i, n_seed_base):
    cls, over, _ = POLICIES[name]
    cfg = dict(EVAL_CONFIG, **{x: y for x, y in over.items() if x != 'observe_c'})
    seed = n_seed_base + 1000 * k + i
    if cls == 'oracle':
        best = None
        for lam in LAM_GRID:
            env = make_env(cfg, sc)
            m = run_episode(env, LagrangianOracle(env, lam, cfg, observe_c=over['observe_c']),
                            seed=seed, charge_c_rate=cfg['charge_c_rate'], options=sc.options())
            if best is None or m['return'] > best['return']:
                best = m
        m, dt = best, float('nan')
    else:
        pol = _Timed(cls(make_robot(cfg), cfg))
        env = make_env(cfg, sc)
        if getattr(pol.pol, 'needs_sensors', False):
            pol.pol.attach_sensors(env, seed)
        m = run_episode(env, pol, seed=seed, charge_c_rate=cfg['charge_c_rate'],
                        options=sc.options())
        dt = 1e6 * pol.t / max(pol.n, 1)
    row = {x: float(m[x]) for x in METRICS}
    row.update(policy=name, family=sc.family, scenario=sc.name, tags=sc.tags, seed_i=i,
               decision_us=dt)
    return row


def _scenario_job(args):
    k, names, n_eps, families, seed = args
    sc = build_suite(families)[k]
    return [_episode(n, sc, k, i, seed) for n in names for i in range(n_eps)]


def run(n_eps: int, out: str, workers: int, families, names, seed=None):
    suite = build_suite(families)
    seed = EVAL_CONFIG['seed'] if seed is None else seed
    jobs = [(k, names, n_eps, families, seed) for k in range(len(suite))]
    t0 = time.time()
    rows = []
    with Pool(workers) as p:
        for j, r in enumerate(p.imap_unordered(_scenario_job, jobs)):
            rows += r
            if (j + 1) % 20 == 0:
                print(f'  {j + 1}/{len(jobs)} scenarios ({time.time() - t0:.0f} s)', flush=True)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'raw.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'{len(rows)} episodes -> {out}/raw.jsonl ({time.time() - t0:.0f} s)')
    return rows


# --- report -------------------------------------------------------------------

def _boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float('nan'), float('nan')
    r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _key(r):
    return (r['scenario'], r['seed_i'])


def report(rows, families, ref='BA-EPD+SOH', out=None):
    names = [n for n in POLICIES if any(r['policy'] == n for r in rows)]
    by = {n: [r for r in rows if r['policy'] == n] for n in names}
    lines = []
    P = lines.append

    P('## Return per family (mean [95% CI]); higher is better\n')
    P('| policy | ' + ' | '.join(families) + ' | all |')
    P('|---' * (len(families) + 2) + '|')
    for n in names:
        cells = []
        for fam in list(families) + ['all']:
            x = [r['return'] for r in by[n] if fam == 'all' or r['family'] == fam]
            lo, hi = _boot_ci(x)
            cells.append(f'{np.mean(x):.3f} [{lo:.3f}, {hi:.3f}]')
        P(f'| {n} | ' + ' | '.join(cells) + ' |')

    P(f'\n## Paired difference in return vs {ref} (mean [95% CI]); > 0 = better than {ref}\n')
    P('| policy | ' + ' | '.join(families) + ' | all |')
    P('|---' * (len(families) + 2) + '|')
    refd = {_key(r): r for r in by.get(ref, [])}
    for n in names:
        if n == ref or not refd:
            continue
        cells = []
        for fam in list(families) + ['all']:
            d = [r['return'] - refd[_key(r)]['return'] for r in by[n]
                 if (fam == 'all' or r['family'] == fam) and _key(r) in refd]
            lo, hi = _boot_ci(d)
            sig = '*' if (lo > 0 or hi < 0) else ''
            cells.append(f'{np.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}')
        P(f'| {n} | ' + ' | '.join(cells) + ' |')

    P('\n## Short-term and long-term metrics (all scenarios)\n')
    P('| policy | done % | time [min] | energy [kJ] | QoS viol % | reserve viol % | final SOC | '
      'Xu dSOH [ppm/mission] | decision [us] |')
    P('|---' * 10 + '|')
    for n in names:
        m = lambda k: float(np.nanmean([r[k] for r in by[n]]))
        P(f"| {n} | {100 * m('completed'):.1f} | {m('mission_time_s') / 60:.2f} | {m('energy_kj'):.2f} | "
          f"{m('qos_viol_pct'):.2f} | {100 * m('reserve_viol'):.1f} | {m('final_soc'):.3f} | "
          f"{m('delta_soh_ppm'):.2f} | {m('decision_us'):.0f} |")

    if 'degradation' in families:
        P('\n## Long term: ageing battery, same mission and charge (degradation family)\n')
        sohs = sorted({r['tags']['soh'] for r in rows if r['family'] == 'degradation'}, reverse=True)
        pick = [s for s in sohs if any(abs(s - t) < 1e-6 for t in (1.0, 0.9, 0.85, 0.8))]
        P('| policy | ' + ' | '.join(f'SOH {s:.3f}: R / res% / T[min]' for s in pick)
          + ' | km to EOL (C=1/2/3) |')
        P('|---' * (len(pick) + 2) + '|')
        for n in names:
            d = [r for r in by[n] if r['family'] == 'degradation']
            cells = []
            for s in pick:
                x = [r for r in d if abs(r['tags']['soh'] - s) < 1e-6]
                cells.append(f"{np.mean([r['return'] for r in x]):.3f} / "
                             f"{100 * np.mean([r['reserve_viol'] for r in x]):.0f} / "
                             f"{np.mean([r['mission_time_s'] for r in x]) / 60:.1f}")
            kms = []
            for c in sorted({r['tags']['c'] for r in d}):
                lv = sorted({r['tags']['soh'] for r in d if r['tags']['c'] == c}, reverse=True)
                fd = [np.mean([r['fd_cycle'] for r in d if r['tags']['c'] == c
                               and abs(r['tags']['soh'] - s) < 1e-6]) for s in lv]
                dist = 5.0
                kms.append(missions_to_eol(lv, fd) * dist)
            P(f'| {n} | ' + ' | '.join(cells) + ' | ' + ' / '.join(f'{k:.0f}' for k in kms) + ' |')

    if 'tight' in families:
        P('\n## Charge-limited missions (tight family): R / reserve viol % by SOH x k\n')
        keys = sorted({(r['tags']['soh'], r['tags']['k']) for r in rows if r['family'] == 'tight'},
                      key=lambda x: (-x[0], x[1]))
        P('| policy | ' + ' | '.join(f'SOH {h} k {k}' for h, k in keys) + ' |')
        P('|---' * (len(keys) + 1) + '|')
        for n in names:
            t = [r for r in by[n] if r['family'] == 'tight']
            P(f'| {n} | ' + ' | '.join(
                f"{np.mean([r['return'] for r in t if (r['tags']['soh'], r['tags']['k']) == hk]):.3f} / "
                f"{100 * np.mean([r['reserve_viol'] for r in t if (r['tags']['soh'], r['tags']['k']) == hk]):.0f}"
                for hk in keys) + ' |')

    P('\n## Long-term objective J = return - w * dSOH / (1 - SOH_EOL) (mean over all scenarios)\n')
    ws = (1000.0, 4000.0, 16000.0)
    P('| policy | return | Xu dSOH [ppm] | ' + ' | '.join(f'J (w={w:g})' for w in ws) + ' |')
    P('|---' * (len(ws) + 3) + '|')
    eol = 0.2
    for n in names:
        R = float(np.mean([r['return'] for r in by[n]]))
        D = float(np.mean([r['delta_soh_ppm'] for r in by[n]]))
        P(f'| {n} | {R:.4f} | {D:.3f} | ' + ' | '.join(f'{R - w * D * 1e-6 / eol:.4f}' for w in ws) + ' |')

    text = '\n'.join(lines)
    print(text)
    if out:
        with open(os.path.join(out, 'report.md'), 'w') as f:
            f.write(text + '\n')
    return text


def main():
    ap = argparse.ArgumentParser(description='Lag-RLS benchmark')
    ap.add_argument('--episodes', type=int, default=10, help='seeds per scenario')
    ap.add_argument('--out', default=os.path.join(EVAL_CONFIG['log_dir'], 'benchmark'))
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--families', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    ap.add_argument('--policies', nargs='+', choices=list(POLICIES),
                    default=[n for n, v in POLICIES.items() if v[2] != 'variant'])
    ap.add_argument('--ref', default='BA-EPD+SOH', help='reference for paired differences')
    ap.add_argument('--seed', type=int, default=None,
                    help=f"base seed (default test seed {EVAL_CONFIG['seed']}; validation: 2024)")
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--inputs', nargs='+', default=None,
                    help='report-only: merge raw.jsonl of several runs (same seeds)')
    args = ap.parse_args()
    if args.report_only:
        rows = []
        for d in (args.inputs or [args.out]):
            with open(os.path.join(d, 'raw.jsonl')) as f:
                rows += [json.loads(x) for x in f]
        os.makedirs(args.out, exist_ok=True)
    else:
        print(f'{len(build_suite(args.families))} scenarios x {len(args.policies)} policies x '
              f'{args.episodes} seeds, {args.workers} workers')
        rows = run(args.episodes, args.out, args.workers, args.families, args.policies, args.seed)
    report(rows, args.families, args.ref, args.out)


if __name__ == '__main__':
    main()
