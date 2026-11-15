"""Supporting results for the Lag-RLS paper (BMS observations + battery realism).

    python -m evaluation.paper_results soh-accuracy   # on-line SOH estimate vs true SOH
    python -m evaluation.paper_results robustness     # sensor noise / cell variation / mismatch sweeps
    python -m evaluation.paper_results trajectory     # time series of one charge-limited aged mission

Outputs (CSV + PNG + Markdown) go to --out (default <log_dir>/paper). SOH is in [0.8, 1]
(SOH_EOL = 0.8: the battery is retired below it).
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
from multiprocessing import Pool

import numpy as np

from . import EVAL_CONFIG
from methods import BatteryAwareEPD, BAEPDSohEstimate, LagRLSEPD
from .metrics import run_episode
from .scenarios import Scenario, build_suite, make_env, PLANTS
from envs.test_env import make_robot
from envs.robot_env import RobotParams, tight_soc_init

BMS = dict(EVAL_CONFIG, battery_obs='bms', battery_realism=True)
FULL = dict(EVAL_CONFIG, battery_obs='true', battery_realism=True)
SOHS = (1.0, 0.95, 0.9, 0.85, 0.8)


def _save_csv(path, rows):
    keys = list(rows[0])
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


# --- 1. SOH estimation accuracy ---------------------------------------------------

T_CHECK = (30, 60, 120, 300, 600, 1200)


def _soh_job(args):
    soh, c, soc_mode, seed = args
    rp = RobotParams()
    soc0 = 1.0 if soc_mode == 'full' else tight_soc_init({}, soh, c, 5000.0, 1.0)
    sc = Scenario('soh_acc', f'soh{soh}_c{c}_{soc_mode}', 5000.0, soh, soc0, c)
    env = make_env(BMS, sc)
    pol = LagRLSEPD(make_robot(BMS), BMS)
    obs, _ = env.reset(seed=BMS['seed'] + seed, options=sc.options())
    pol.reset()
    trace, t, done = {}, 0, False
    while not done:
        obs, _, te, tr, _ = env.step(pol.select_action(obs))
        t += 1
        done = te or tr
        trace[t] = pol.soh_hat
    t_end = t
    est = np.array([trace[k] for k in range(1, t_end + 1)])
    err = np.abs(est - soh)
    bad = np.nonzero(err > 0.02)[0]
    conv = float(bad[-1] + 2) if len(bad) else 1.0          # time after which |err| <= 0.02
    row = dict(soh=soh, c=c, soc0=soc_mode, seed=seed, t_end=t_end, err_end=float(est[-1] - soh),
               t_conv_s=conv if conv <= t_end else float('nan'),
               **{f'err_{k}s': float(trace[min(k, t_end)] - soh) for k in T_CHECK})
    row['trace'] = est[::10].round(4).tolist()
    return row


def soh_accuracy(out, seeds, workers):
    jobs = [(h, c, m, s) for h in SOHS for c in (1.0, 2.0, 3.0) for m in ('full', 'tight')
            for s in range(seeds)]
    with Pool(workers) as p:
        rows = p.map(_soh_job, jobs)
    _save_csv(os.path.join(out, 'soh_accuracy.csv'), [{k: v for k, v in r.items() if k != 'trace'} for r in rows])
    lines = ['## On-line SOH estimation (BMS + realism): |error| mean / P95, bias at the end, '
             'time until |error| stays <= 0.02\n',
             '| true SOH | ' + ' | '.join(f'|err| @{k}s' for k in T_CHECK) +
             ' | bias (end) | P95 |err| (end) | t_conv [s] (median) |', '|---' * (len(T_CHECK) + 4) + '|']
    for h in SOHS:
        r = [x for x in rows if x['soh'] == h]
        cells = [f"{np.mean([abs(x[f'err_{k}s']) for x in r]):.3f}" for k in T_CHECK]
        e = np.array([x['err_end'] for x in r])
        tc = np.array([x['t_conv_s'] for x in r])
        lines.append(f'| {h:.2f} | ' + ' | '.join(cells) + f' | {e.mean():+.3f} | '
                     f'{np.percentile(np.abs(e), 95):.3f} | {np.nanmedian(tc):.0f} |')
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(out, 'soh_accuracy.md'), 'w') as f:
        f.write(text + '\n')
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for h in SOHS:
        tr = [np.array(x['trace']) for x in rows if x['soh'] == h]
        n = min(len(t) for t in tr)
        a = np.stack([t[:n] for t in tr]) - h
        tt = np.arange(n) * 10
        ax.plot(tt, a.mean(0), label=f'SOH {h:.2f}')
        ax.fill_between(tt, np.percentile(a, 5, 0), np.percentile(a, 95, 0), alpha=0.15)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('SOH estimate - true')
    ax.set_xlim(0, 1200)
    ax.set_ylim(-0.1, 0.1)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'soh_accuracy.png'), dpi=200)


# --- 2. Robustness sweeps -----------------------------------------------------------

ROB_POLICIES = {
    'BA-EPD':              (BatteryAwareEPD, {}),
    'BA-EPD+SOH':          (BAEPDSohEstimate, {}),
    'Lag-RLS w/o SOH est': (LagRLSEPD, {'lag_soh': 'obs'}),
    'Lag-RLS':             (LagRLSEPD, {}),
    'Lag-RLS (full info)': (LagRLSEPD, {'lag_soh': 'obs', 'battery_obs': 'true'}),
}


def _charge_suite():
    """Charge-limited scenarios where the battery state matters: tight + degradation C = 2."""
    return [s for s in build_suite(['tight', 'degradation'])
            if s.family == 'tight' or s.tags['c'] == 2.0]


def _mismatch_suite(kind, x):
    out = []
    for soh in (1.0, 0.85):
        for k in (1.0, 1.1):
            if kind == 'payload':
                plant = {'motor_base_w': x * RobotParams().motor_base_w,
                         'motor_linear_w_per_mps': x * RobotParams().motor_linear_w_per_mps,
                         'motor_cubic_w_per_mps3': x * RobotParams().motor_cubic_w_per_mps3}
            else:
                plant = {'motor_cubic_w_per_mps3': x * RobotParams().motor_cubic_w_per_mps3}
            soc0 = tight_soc_init(plant, soh, 2.0, 5000.0, k)
            out.append(Scenario('mm', f'{kind}{x}_soh{soh}_k{k}', 5000.0, soh, soc0, 2.0, plant=plant,
                                tags={'soh': soh, 'k': k}))
    return out


SWEEPS = {
    'voltage_noise_mV': ([0.0, 15.0, 30.0, 60.0], lambda x: {'meas_voltage_noise_v': x / 1000.0}, 'charge'),
    'cell_var_scale':   ([0.0, 1.0, 2.0], lambda x: {'cell_var_scale': x}, 'charge'),
    'payload_factor':   ([1.0, 1.25, 1.5, 1.75], None, 'payload'),
    'terrain_v3_factor': ([1.0, 1.5, 2.0, 3.0], None, 'terrain'),
}


def _rob_job(args):
    sweep, x, k, sc, n = args
    _, f_over, kind = SWEEPS[sweep]
    extra = f_over(x) if f_over else {}
    res = []
    for name, (cls, over) in ROB_POLICIES.items():
        cfg = dict(BMS, **extra, **over)
        for i in range(n):
            m = run_episode(make_env(cfg, sc), cls(make_robot(cfg), cfg),
                            seed=cfg['seed'] + 1000 * k + i, options=sc.options())
            res.append(dict(sweep=sweep, x=x, policy=name, scenario=sc.name, seed=i,
                            ret=float(m['return']), reserve_viol=float(m['reserve_viol']),
                            completed=float(m['completed'])))
    return res


def robustness(out, seeds, workers):
    jobs = []
    for sweep, (xs, _, kind) in SWEEPS.items():
        for x in xs:
            suite = _charge_suite() if kind == 'charge' else _mismatch_suite(kind, x)
            jobs += [(sweep, x, k, sc, seeds) for k, sc in enumerate(suite)]
    with Pool(workers) as p:
        rows = [r for rs in p.map(_rob_job, jobs) for r in rs]
    _save_csv(os.path.join(out, 'robustness.csv'), rows)
    lines = []
    plt = _plt()
    fig, axes = plt.subplots(1, len(SWEEPS), figsize=(4 * len(SWEEPS), 3.4))
    for ax, (sweep, (xs, _, _)) in zip(axes, SWEEPS.items()):
        lines += [f'\n## {sweep}: return / reserve viol %\n',
                  '| policy | ' + ' | '.join(f'{x:g}' for x in xs) + ' |', '|---' * (len(xs) + 1) + '|']
        for name in ROB_POLICIES:
            m = [np.mean([r['ret'] for r in rows if r['sweep'] == sweep and r['x'] == x and r['policy'] == name])
                 for x in xs]
            v = [100 * np.mean([r['reserve_viol'] for r in rows
                                if r['sweep'] == sweep and r['x'] == x and r['policy'] == name]) for x in xs]
            lines.append(f'| {name} | ' + ' | '.join(f'{a:.3f} / {b:.0f}' for a, b in zip(m, v)) + ' |')
            ax.plot(xs, m, marker='o', label=name)
        ax.set_xlabel(sweep)
        ax.set_ylabel('return')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'robustness.png'), dpi=200)
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(out, 'robustness.md'), 'w') as f:
        f.write(text + '\n')


# --- 3. Trajectory example -------------------------------------------------------------

def trajectory(out, soh=0.8, c=2.0, seed=3):
    sc = [s for s in build_suite(['degradation']) if s.tags['c'] == c and abs(s.tags['soh'] - soh) < 1e-6][0]
    pols = {'Lag-RLS': (LagRLSEPD, BMS), 'Lag-RLS w/o SOH est': (LagRLSEPD, dict(BMS, lag_soh='obs')),
            'BA-EPD': (BatteryAwareEPD, BMS)}
    data = {}
    for name, (cls, cfg) in pols.items():
        env, pol = make_env(cfg, sc), cls(make_robot(cfg), cfg)
        obs, _ = env.reset(seed=cfg['seed'] + seed, options=sc.options())
        pol.reset()
        rows, done, ret = [], False, 0.0
        while not done:
            a = pol.select_action(obs)
            soc_bms = float(obs[2])
            obs, r, te, tr, info = env.step(a)
            ret += r
            done = te or tr
            rows.append(dict(t=info['t_s'], v=float(a[0]), f=float(a[1]), soc_true=info['soc'],
                             soc_bms=soc_bms, soh_hat=float(getattr(pol, 'soh_hat', np.nan)),
                             lam=float(getattr(pol, '_lam', np.nan)), d=info['distance_m']))
        data[name] = dict(rows=rows, ret=ret, soc_end=rows[-1]['soc_true'], t_end=rows[-1]['t'])
        _save_csv(os.path.join(out, f'trajectory_{name.replace(" ", "_").replace("/", "")}.csv'), rows)
    plt = _plt()
    fig, ax = plt.subplots(4, 1, figsize=(7, 8), sharex=True)
    for name, d in data.items():
        t = [r['t'] for r in d['rows']]
        ax[0].plot(t, [r['v'] for r in d['rows']], lw=0.8, label=f"{name} (R={d['ret']:.3f}, SOC_end={d['soc_end']:.3f})")
        ax[1].plot(t, [r['f'] for r in d['rows']], lw=0.8)
        ax[2].plot(t, [r['soc_true'] for r in d['rows']], lw=1.0)
        if name == 'Lag-RLS':
            ax[2].plot(t, [r['soc_bms'] for r in d['rows']], 'k--', lw=0.8, label='BMS SOC (Lag-RLS run)')
            ax[3].plot(t, [r['soh_hat'] for r in d['rows']], lw=1.0, label='SOH estimate')
    ax[2].axhline(RobotParams().soc_reserve, color='r', lw=0.6, ls=':', label='reserve')
    ax[3].axhline(soh, color='k', lw=0.6, ls=':', label='true SOH')
    ax[0].set_ylabel('speed [m/s]')
    ax[1].set_ylabel('CPU f [GHz]')
    ax[2].set_ylabel('SOC')
    ax[3].set_ylabel('SOH')
    ax[3].set_xlabel('time [s]')
    ax[0].legend(fontsize=7)
    ax[2].legend(fontsize=7)
    ax[3].legend(fontsize=7)
    fig.suptitle(f'Degradation mission: 5 km, C={c:g}, true SOH={soh}', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'trajectory.png'), dpi=200)
    summary = {n: {k: d[k] for k in ('ret', 'soc_end', 't_end')} for n, d in data.items()}
    print(json.dumps(summary, indent=1))
    with open(os.path.join(out, 'trajectory_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description='Supporting paper results')
    ap.add_argument('what', choices=['soh-accuracy', 'robustness', 'trajectory'])
    ap.add_argument('--out', default=os.path.join(EVAL_CONFIG['log_dir'], 'paper'))
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.what == 'soh-accuracy':
        soh_accuracy(args.out, args.seeds, args.workers)
    elif args.what == 'robustness':
        robustness(args.out, args.seeds, args.workers)
    else:
        trajectory(args.out)


if __name__ == '__main__':
    main()
