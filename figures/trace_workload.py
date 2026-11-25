"""Time traces under a switching workload: the complexity is drawn at random and held for
10 min (no step-to-step disturbance). Rows: speed, CPU frequency, energy/distance; the
background colour (50 % alpha) marks the complexity.

Scenario: 5 km, SOH 0.8, initial SOC such that the usable energy above the 5 % reserve equals
the energy-optimal mission energy (k = 1.0, time-weighted over the schedule),
BMS observations + battery realism, one seed.
'Static' = one fixed (speed, DVFS level), the best at SOC = 1, SOH = 1 and high complexity.

    python figures/trace_workload.py            # run (cached) + plot
    python figures/trace_workload.py --rerun    # ignore the cache
-> figures/pdf/trace_workload.pdf, cache figures/data/trace_workload.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'figures'))

import trace_complexity as tc  # noqa: E402  (methods, static config, helpers)
from evaluation import EVAL_CONFIG  # noqa: E402
from evaluation.metrics import run_episode  # noqa: E402
from envs.scenarios import Scenario, make_env  # noqa: E402
from envs.test_env import make_robot  # noqa: E402
from envs.robot_env import tight_soc_init, CPU_FREQ_GHZ  # noqa: E402

tc.JITTER = 0.0                       # no disturbance (env and controllers' allowance)
PERIOD_S = 300.0                      # workload held for 5 min, then switched
SCHED_SEED = 2                        # seed of the random schedule (first seed whose first 3 segments cover L, M, H)
T_EST_S = 24 * 60.0                   # expected mission time (for the SOC0 weighting)


def _schedule(n=6, seed=SCHED_SEED):
    """Random complexity per 10-min segment; every switch changes the level."""
    rng = np.random.default_rng(seed)
    out = [float(rng.choice([1.0, 2.0, 3.0]))]
    while len(out) < n:
        out.append(float(rng.choice([c for c in (1.0, 2.0, 3.0) if c != out[-1]])))
    return out


LEVELS = _schedule(n=8)
C_NAME = {1.0: 'Low', 2.0: 'Medium', 3.0: 'High'}
SOH, K, DIST, SEED = tc.SOH, tc.K, tc.DIST, tc.SEED
BMS = tc.BMS
METHODS = {k: v for k, v in tc.METHODS.items() if k != 'Lag-RLS'}   # DARO only
CACHE = ROOT / 'figures' / 'data' / 'trace_workload.pkl'
QOS_OK = 5.0                          # acceptable QoS violation rate [% of steps]
BAND_ALPHA = 0.3                      # complexity background opacity
RESERVE = 0.05                        # SOC reserve


def soc0():
    """Time-weighted SOC0 over the segments the mission is expected to see."""
    w = [max(0.0, min(PERIOD_S, T_EST_S - k * PERIOD_S)) for k in range(len(LEVELS))]
    return float(np.average([tight_soc_init({}, SOH, c, DIST, K) for c in LEVELS], weights=w))


def options():
    return {'complexity_schedule': (PERIOD_S, LEVELS), 'complexity_switch_period_s': 0.0}


def run_one(name):
    cls, over = METHODS[name]
    cfg = dict(EVAL_CONFIG, **over, test_complexity_jitter=0.0)
    sc = Scenario('workload', 'sched', DIST, SOH, soc0(), LEVELS[0])
    env = make_env(cfg, sc)
    if cls == 'static':
        pol = tc.FixedConfig(*tc.static_config(tc.STATIC_C))
    else:
        pol = cls(make_robot(cfg), cfg)
    m = run_episode(env, pol, seed=SEED, charge_c_rate=cfg['charge_c_rate'], options=options())
    tr = {k: np.array([s[k] for s in env.trajectory], dtype=float) for k in tc.KEYS}
    tr['qos_violation'] = np.array([s['qos_violation'] for s in env.trajectory], dtype=float)
    tr['throughput'] = np.array([min(s['throughput']) for s in env.trajectory], dtype=float)
    return dict(metrics=m, trace=tr, soc0=sc.soc0)


def complexity_at(t_s):
    k = (np.asarray(t_s) // PERIOD_S).astype(int)
    return np.asarray(LEVELS)[k % len(LEVELS)]


def plot(res):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 13, 'font.family': 'serif', 'mathtext.fontset': 'cm'})
    style = {
        'Static':     dict(color='#9a9a9a', ls=':'),
        'QoS-EPD':    dict(color='#8c6bb1', ls='-.'),
        'BA-EPD':     dict(color='#e08a2c', ls='--'),
        'BA-EPD+SOH': dict(color='#c8413b', ls='--'),
        'Lag-RLS':    dict(color='#2a5ea8', ls='-'),
        'DARO': dict(color='#1a8a4a', ls='-'),
    }
    band = {1.0: '#b8e6b0', 2.0: '#fff1a8', 3.0: '#f5aaa0'}     # Low / Medium / High, alpha 0.5
    rows = [('speed_mps', 'Speed (m/s)'), ('cpu_ghz', 'CPU freq. (GHz)'),
            ('e_per_m', 'Energy / distance (J/m)'), ('qos_rate', 'QoS violation (%)')]
    t_end = max(r['trace']['t_s'][-1] for r in res.values()) / 60.0
    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 12.5), sharex=True)

    for ax, (key, lab) in zip(axes, rows):
        for n in METHODS:
            tr = res[n]['trace']
            x = tr['t_s'] / 60.0
            if key == 'e_per_m':
                y = np.where(tr['distance_m'] >= tc.E_SKIP_M,
                             tr['total_energy_j'] / np.maximum(tr['distance_m'], 1.0), np.nan)
            elif key == 'qos_rate':     # cumulative share of steps below the QoS requirement
                y = 100.0 * np.cumsum(tr['qos_violation'] > 0) / np.arange(1, len(x) + 1)
            else:
                y = tr[key]
            ax.plot(x, y, lw=1.8, **style[n])
            m = res[n]['metrics']
            # mission end: dot = completed above the reserve; X = ended below the 5 % SOC reserve
            # (or failed)
            bad = (not m['completed']) or m['final_soc'] < RESERVE
            ax.plot(x[-1], y[-1], 'X' if bad else 'o', ms=11 if bad else 5, zorder=5,
                    color=style[n]['color'], mec='black' if bad else style[n]['color'], mew=0.8)
            if bad and key == 'e_per_m':
                txt = (f'{n}: failed' if not m['completed']
                       else f"{n}: SOC$_{{\\mathrm{{end}}}}$ = {100 * m['final_soc']:.1f}% < 5%")
                ax.annotate(txt, (x[-1], y[-1]), xytext=(-10, 0), textcoords='offset points',
                            ha='right', va='center', fontsize=11, color='#222222',
                            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8),
                            zorder=6)
        ax.set_ylabel(lab)
        # y range fitted to the data after the start-up / probing phase
        ys = np.concatenate([np.asarray(ln.get_ydata(), float)[
            (np.asarray(ln.get_xdata(), float) >= tc.Y_FIT_FROM_MIN)] for ln in ax.get_lines()
            if len(ln.get_xdata()) > 2])
        ys = ys[np.isfinite(ys)]
        pad = max(0.08 * (ys.max() - ys.min()), 0.05)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)
        if key == 'qos_rate':
            ax.set_ylim(-0.3, 1.6 * QOS_OK)
            ax.axhline(QOS_OK, color='#222222', ls='--', lw=1.2, zorder=1)
            ax.text(0.2, QOS_OK, f'acceptable ($\\leq${QOS_OK:g}%)', ha='left', va='bottom',
                    fontsize=11, color='#222222')
        if key == 'cpu_ghz':          # ticks at the DVFS levels in range
            lv = [f for f in CPU_FREQ_GHZ if ys.min() - pad <= f <= ys.max() + pad]
            ax.set_yticks(lv)
            ax.set_yticklabels([f'{f:.2f}' for f in lv])

    for ax in axes:                   # workload bands + end-of-mission lines
        for k in range(int(np.ceil(t_end * 60.0 / PERIOD_S))):
            c = LEVELS[k % len(LEVELS)]
            ax.axvspan(k * PERIOD_S / 60.0, (k + 1) * PERIOD_S / 60.0, color=band[c], alpha=BAND_ALPHA,
                       zorder=0, lw=0)
        for n in METHODS:
            ax.axvline(res[n]['trace']['t_s'][-1] / 60.0, color=style[n]['color'], ls=':',
                       lw=1.0, alpha=0.8, zorder=0.5)
        ax.grid(alpha=0.3, axis='y')
    axes[-1].set_xlim(0, t_end + 0.3)
    axes[-1].set_xlabel('Time (min)')
    from matplotlib.patches import Patch
    handles = [plt.Line2D([], [], lw=1.8, **style[n], label=n) for n in METHODS]
    bands = [Patch(color=band[c], alpha=BAND_ALPHA, label=f'{C_NAME[c]} complexity')
             for c in (1.0, 2.0, 3.0)]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.legend(handles=bands, loc='upper center', ncol=len(bands), frameon=False,
               bbox_to_anchor=(0.5, 0.968))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ROOT / 'figures' / 'pdf' / 'trace_workload.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(out)


def summary(res):
    print(f"{'method':<11} {'SOC0':>5} {'T[min]':>7} {'E[kJ]':>6} {'J/m':>5} {'SOC_end':>7} "
          f"{'done':>4} {'QoSv%':>6} {'R':>7}")
    for n in METHODS:
        r = res[n]
        m = r['metrics']
        print(f"{n:<11} {r['soc0']:5.3f} {m['mission_time_s'] / 60:7.2f} {m['energy_kj']:6.2f} "
              f"{1e3 * m['energy_kj'] / DIST:5.2f} {m['final_soc']:7.3f} {int(m['completed']):4d} "
              f"{m['qos_viol_pct']:6.2f} {m['return']:7.3f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rerun', action='store_true')
    a = ap.parse_args()
    if CACHE.exists() and not a.rerun:
        res = pickle.loads(CACHE.read_bytes())
    else:
        res = {n: run_one(n) for n in METHODS}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(res))
    summary(res)
    plot(res)
