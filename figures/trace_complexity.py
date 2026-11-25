"""Time traces of speed, CPU frequency and cumulative energy for all rule-based methods
at the three complexity levels (fixed workload, no switching).

Scenario per complexity (same as the 'tight' benchmark family): 5 km, SOH 0.8, initial SOC
such that the usable energy above the 5 % reserve equals the energy-optimal mission
energy (k = 1.0), BMS observations + battery realism, complexity jitter +-10 %, one seed.
Row 3: cumulative energy / cumulative distance [J/m].

    python figures/trace_complexity.py            # run (cached) + plot
    python figures/trace_complexity.py --rerun    # ignore the cache
-> figures/pdf/trace_complexity.pdf, cache figures/data/trace_complexity.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluation  # noqa: E402,F401  (path setup)
from evaluation import EVAL_CONFIG  # noqa: E402
from evaluation.metrics import run_episode  # noqa: E402
from envs.scenarios import Scenario, make_env  # noqa: E402
from envs.test_env import make_robot  # noqa: E402
from envs.robot_env import tight_soc_init, CPU_FREQ_GHZ  # noqa: E402
from methods import ASMFBaseline, QoSAwareEPD, BatteryAwareEPD, BAEPDSohEstimate, LagRLSEPD  # noqa: E402
from methods.degradation_aware import DegAwareLagRLS  # noqa: E402
from methods.oracle import LagrangianOracle, LAM_GRID  # noqa: E402

SOH, K, DIST, SEED = 0.8, 1.0, 5000.0, 42
E_SKIP_M = 100.0  # energy/distance shown from this distance on
JITTER = 0.1     # complexity disturbance +-10 % (plant and controllers' jitter allowance)
CS = {'Low': 1.0, 'Medium': 2.0, 'High': 3.0}
BMS = dict(battery_obs='bms', battery_realism=True)
METHODS = {   # name -> (class, config overrides)
    'Static': ('static', BMS),
    'QoS-EPD': (QoSAwareEPD, BMS),
    'BA-EPD': (BatteryAwareEPD, BMS),
    'BA-EPD+SOH': (BAEPDSohEstimate, BMS),
    'Lag-RLS': (LagRLSEPD, BMS),
    'DARO': (DegAwareLagRLS, dict(BMS, deg_weight=16000.0)),
}
KEYS = ('t_s', 'speed_mps', 'cpu_ghz', 'total_energy_j', 'distance_m', 'soc', 'power_w')
CACHE = ROOT / 'figures' / 'data' / 'trace_complexity.pkl'


class FixedConfig:
    """Constant speed and DVFS level for the whole mission."""

    def __init__(self, v, level):
        self.a = np.array([v, level], dtype=np.float32)

    def reset(self):
        pass

    def select_action(self, obs, deterministic=True):
        return self.a.copy()


STATIC_C = 3.0                    # the static config is tuned at high complexity (QoS-safe for all)
STATIC_SPEEDS = np.round(np.arange(1.0, 5.0 + 1e-9, 0.1), 1)
STATIC_SEEDS = (0, 1, 2)          # tuning seeds (differ from the test seed)
STATIC_CACHE = ROOT / 'figures' / 'data' / 'trace_static_config.pkl'


def _static_eval(args):
    c, v, lev = args
    cfg = dict(EVAL_CONFIG, **BMS, test_complexity_jitter=JITTER)
    sc = Scenario('static', f'c{c:g}', DIST, 1.0, 1.0, c)          # fresh, full battery
    rs = [run_episode(make_env(cfg, sc), FixedConfig(v, lev), seed=s,
                      charge_c_rate=cfg['charge_c_rate'], options=sc.options())['return']
          for s in STATIC_SEEDS]
    return c, v, lev, float(np.mean(rs))


def static_config(c):
    """Best fixed (speed, DVFS level) at SOC = 1, SOH = 1 for complexity c (grid search, cached)."""
    best = pickle.loads(STATIC_CACHE.read_bytes()) if STATIC_CACHE.exists() else {}
    key = (c, JITTER, DIST)
    if key not in best:
        from multiprocessing import Pool
        jobs = [(c, v, lev) for v in STATIC_SPEEDS for lev in range(len(CPU_FREQ_GHZ))]
        with Pool() as p:
            out = p.map(_static_eval, jobs)
        _, v, lev, r = max(out, key=lambda x: x[3])
        best[key] = (float(v), int(lev))
        print(f'static config c={c:g}: v={v:.1f} m/s, f={CPU_FREQ_GHZ[lev]:.2f} GHz (R={r:.3f})')
        STATIC_CACHE.parent.mkdir(parents=True, exist_ok=True)
        STATIC_CACHE.write_bytes(pickle.dumps(best))
    return best[key]


def run_one(name, c):
    cls, over = METHODS[name]
    cfg = dict(EVAL_CONFIG, **over, test_complexity_jitter=JITTER)
    sc = Scenario('trace', f'c{c:g}', DIST, SOH, tight_soc_init({}, SOH, c, DIST, K), c)
    if cls == 'static':      # one fixed (v, f) for all complexities, tuned at SOC = 1, SOH = 1, high c
        v, lev = static_config(STATIC_C)
        env = make_env(cfg, sc)
        m = run_episode(env, FixedConfig(v, lev), seed=SEED,
                        charge_c_rate=cfg['charge_c_rate'], options=sc.options())
    elif cls == 'oracle':      # best constant lam in hindsight on the same seed
        best = None
        for lam in LAM_GRID:
            env = make_env(cfg, sc)
            m = run_episode(env, LagrangianOracle(env, lam, cfg), seed=SEED,
                            charge_c_rate=cfg['charge_c_rate'], options=sc.options())
            if best is None or m['return'] > best[0]['return']:
                best = (m, env)
        m, env = best
    else:
        env = make_env(cfg, sc)
        m = run_episode(env, cls(make_robot(cfg), cfg), seed=SEED,
                        charge_c_rate=cfg['charge_c_rate'], options=sc.options())
    tr = {k: np.array([s[k] for s in env.trajectory], dtype=float) for k in KEYS}
    return dict(metrics=m, trace=tr, soc0=sc.soc0)


def run_all():
    return {(lvl, n): run_one(n, c) for lvl, c in CS.items() for n in METHODS}


Y_FIT_FROM_MIN = 3.0   # y range fitted to the data after the start-up / probing phase [min]
SMOOTH = {'speed_mps': 10, 'cpu_ghz': 30}   # moving-average window [steps = s]


def movavg(y, w):
    """Centred moving average with shrinking window at the edges (no zero padding)."""
    k = np.ones(w)
    return np.convolve(y, k, mode='same') / np.convolve(np.ones_like(y), k, mode='same')


def cruise(y):
    """Level the line sits on: the most frequent value after the start-up (10 % of the mission)."""
    y = y[len(y) // 10:]
    v, n = np.unique(np.round(y, 3), return_counts=True)
    if n.max() >= 0.3 * len(y):          # piecewise constant (DVFS level, fixed speed)
        return float(v[np.argmax(n)])
    return float(np.round(np.median(y), 1))   # noisy speed: median


def level_ticks(vals, gap):
    """Sorted distinct levels, dropping one of two levels closer than gap (keeps the higher)."""
    out = []
    for v in sorted(set(vals)):
        if out and v - out[-1] < gap:
            out[-1] = v
        else:
            out.append(v)
    return out


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
    rows = [('speed_mps', 'Speed (m/s)', 1.0), ('cpu_ghz', 'CPU freq. (GHz)', 1.0),
            ('e_per_m', 'Energy / distance (J/m)', 1.0)]
    fig, axes = plt.subplots(len(rows), len(CS), figsize=(18, 10), sharex='col')
    for j, lvl in enumerate(CS):
        for i, (key, lab, scale) in enumerate(rows):
            ax = axes[i, j]
            for n in METHODS:
                tr = res[(lvl, n)]['trace']
                st = {'lw': 1.6, **style[n]}
                x = tr['t_s'] / 60.0
                if key == 'e_per_m':        # skip the start-up (first 100 m: acceleration from 0)
                    y = np.where(tr['distance_m'] >= E_SKIP_M,
                                 tr['total_energy_j'] / np.maximum(tr['distance_m'], 1.0), np.nan)
                else:
                    y = tr[key] * scale
                if key in SMOOTH:            # moving average (jitter / probing are 1 s events)
                    y = movavg(y, SMOOTH[key])
                ax.plot(x, y, **st)
                done = bool(res[(lvl, n)]['metrics']['completed'])
                # mission end: dot if completed, cross if failed (SOC cutoff / timeout)
                ax.plot(x[-1], y[-1], 'o' if done else 'X', ms=5 if done else 10,
                        color=style[n]['color'], mec='black' if not done else style[n]['color'])
                ax.axvline(x[-1], color=style[n]['color'], ls=':', lw=1.0, alpha=0.8, zorder=0)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(f'{lvl} complexity')
            if i == len(rows) - 1:
                ax.set_xlabel('Time (min)')
            ax.tick_params(labelleft=True)
        # y ticks at the levels the lines actually sit on (cruise speed, DVFS level)
        for i, key, gap, fmt in ((0, 'speed_mps', 0.08, '{:.1f}'), (1, 'cpu_ghz', 0.1, '{:.2f}')):
            lv = level_ticks([cruise(res[(lvl, n)]['trace'][key]) for n in METHODS], gap)
            axes[i, j].set_yticks(lv)
            axes[i, j].set_yticklabels([fmt.format(x) for x in lv])
    for ax in axes.flat:      # y range per panel, fitted to the lines after the start-up
        ys = []
        for ln in ax.get_lines():
            x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
            if len(x) > 2:                           # skip end markers and axvlines
                ys.append(y[(x >= Y_FIT_FROM_MIN) & np.isfinite(y)])
        ys = np.concatenate(ys)
        lo, hi = ys.min(), ys.max()
        pad = max(0.12 * (hi - lo), 0.05)
        ax.set_ylim(lo - pad, hi + pad)
    handles = [plt.Line2D([], [], lw=2, **{k: v for k, v in style[n].items() if k != 'lw'}, label=n)
               for n in METHODS]
    fig.legend(handles=handles, loc='upper center', ncol=len(METHODS), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = ROOT / 'figures' / 'pdf' / 'trace_complexity.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(out)


def summary(res):
    print(f"{'complexity':<8} {'method':<11} {'SOC0':>5} {'T[min]':>7} {'E[kJ]':>6} "
          f"{'SOC_end':>7} {'done':>4} {'QoSv%':>6} {'R':>7}")
    for (lvl, n), r in res.items():
        m = r['metrics']
        print(f"{lvl:<8} {n:<11} {r['soc0']:5.3f} {m['mission_time_s'] / 60:7.2f} "
              f"{m['energy_kj']:6.2f} {m['final_soc']:7.3f} {int(m['completed']):4d} "
              f"{m['qos_viol_pct']:6.2f} {m['return']:7.3f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rerun', action='store_true')
    a = ap.parse_args()
    if CACHE.exists() and not a.rerun:
        res = pickle.loads(CACHE.read_bytes())
    else:
        res = run_all()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(res))
    summary(res)
    plot(res)
