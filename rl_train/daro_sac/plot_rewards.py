"""
plot_rewards.py
---------------
Reward curves from training logs.

  plot_training_curve(log_dir)   -> <log_dir>/reward_curve.png
      (a) training episode return vs environment steps, moving average per
          mission distance (returns scale with distance, so they are not mixed)
      (b) periodic DARO evaluation return on the fixed test grid (if logged)

  plot_seed_comparison(log_dirs, out_path)
      mean +/- std across seeds of the moving-average return (per km, so all
      distances share one axis), interpolated onto a common step grid.

CLI:
    python plot_rewards.py logs/seed_42
    python plot_rewards.py logs/seed_0 logs/seed_1 logs/seed_2 --out logs/reward_seeds.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-4 (validated: CVD + normal-vision separation pass; two
# slots are < 3:1 vs surface, so lines carry direct end labels + a legend).
# Categorical slots in fixed order (validated; adjacent-pair CVD/normal-vision pass).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
          '#e87ba4', '#008300', '#6250d6', '#e34948']
TEXT, TEXT_2, GRID, SURFACE = '#1f1f1e', '#5f5e58', '#e4e3df', '#fcfcfb'


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT_2)
    ax.yaxis.label.set_color(TEXT_2)



def target_distance(e: dict) -> float:
    """Mission (target) distance of a logged episode [m].

    Newer logs store it as 'target_distance_m' ('distance_m' is the distance
    actually travelled); older logs only have 'distance_m'.
    """
    return float(e.get('target_distance_m', e.get('distance_m', 1000.0)))

def moving_average(x: Sequence[float], window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    w = max(1, min(window, len(x)))
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    out[:w - 1] = c[1:w] / np.arange(1, w)  # expanding mean at the start
    return out


def _load(log_dir: str):
    with open(os.path.join(log_dir, 'train_log.json')) as f:
        train = json.load(f)
    evals = []
    p = os.path.join(log_dir, 'eval_log.jsonl')
    if os.path.exists(p):
        with open(p) as f:
            evals = [json.loads(line) for line in f if line.strip()]
    return train, evals


def _label_end(ax, x, y, text, color):
    ax.plot([x], [y], 'o', ms=4, color=color, zorder=4)
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords='offset points',
                va='center', fontsize=8.5, color=TEXT)


def plot_training_curve(log_dir: str, window: int = 50,
                        out_path: Optional[str] = None) -> Optional[str]:
    train, evals = _load(log_dir)
    if not train:
        return None
    out_path = out_path or os.path.join(log_dir, 'reward_curve.png')
    has_eval = any('global_step' in e for e in evals)
    fig, axes = plt.subplots(1, 2 if has_eval else 1, figsize=(12 if has_eval else 7.5, 4.2),
                             squeeze=False, facecolor=SURFACE)
    ax = axes[0, 0]
    _style(ax)
    dists = sorted({target_distance(e) for e in train})
    x_max = max(e['global_step'] for e in train)
    for i, d in enumerate(dists):
        rows = [e for e in train if target_distance(e) == d]
        x = np.array([e['global_step'] for e in rows])
        y = np.array([e['return'] for e in rows])
        color = SERIES[i % len(SERIES)]
        label = f"{d:.0f} m" if d > 0 else 'all'
        ax.scatter(x, y, s=9, color=color, alpha=0.3, linewidths=0, zorder=2)
        ma = moving_average(y, window)
        ax.plot(x, ma, color=color, linewidth=2, label=label, zorder=3)
        _label_end(ax, x[-1], ma[-1], label, color)
    ax.set_xlim(0, x_max * 1.08)
    ax.set_xlabel('Environment steps')
    ax.set_ylabel('Episode return')
    ax.set_title(f'Training return (moving average, {window} episodes per distance)',
                 loc='left', fontsize=10.5, color=TEXT)
    ax.legend(title='Mission distance', fontsize=8.5, title_fontsize=8.5,
              frameon=False, loc='lower right', labelcolor=TEXT)

    if has_eval:
        ax2 = axes[0, 1]
        _style(ax2)
        ev = [e for e in evals if 'global_step' in e]
        x = np.array([e['global_step'] for e in ev])
        y = np.array([e['results']['DARO']['return'] for e in ev])
        ax2.plot(x, y, color=SERIES[0], linewidth=2, marker='o', ms=4, zorder=3)
        _label_end(ax2, x[-1], y[-1], f"{y[-1]:.2f}", SERIES[0])
        ax2.set_xlim(0, x_max * 1.08)
        ax2.set_xlabel('Environment steps')
        ax2.set_ylabel('Mean test return (500 m grid)')
        ax2.set_title('Periodic evaluation (deterministic policy, fixed test seeds)',
                      loc='left', fontsize=10.5, color=TEXT)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def _spread_labels(ys, min_gap):
    """Nudge label y-positions apart (keeps order) so end labels don't collide."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    shift = (np.mean(out) - np.mean(ys))           # re-centre the block
    return out - shift


def plot_seed_comparison(log_dirs: List[str], out_path: str,
                         window: int = 50, n_grid: int = 300,
                         zoom_from: float = 0.3) -> Optional[str]:
    """One PNG with every seed's curve (per-km return, moving average) + mean.

    Left: full run. Right: zoom on the last (1 - zoom_from) of training,
    where seeds converge and overlap on the full-range axis.
    """
    curves = []
    for d in log_dirs:
        train, _ = _load(d)
        if not train:
            continue
        x = np.array([e['global_step'] for e in train], dtype=float)
        per_km = np.array([e['return'] / max(target_distance(e) / 1000.0, 1e-9)
                           for e in train])
        name = os.path.basename(os.path.normpath(d)).replace('_', ' ')
        curves.append((name, x, moving_average(per_km, window)))
    if len(curves) < 2:
        return None
    x_end = min(c[1][-1] for c in curves)
    grid = np.linspace(max(c[1][0] for c in curves), x_end, n_grid)
    ys = np.stack([np.interp(grid, x, y) for _, x, y in curves])
    mean = ys.mean(0)
    many = len(curves) > len(SERIES)               # no 9th hue: fall back to grey

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE,
                             gridspec_kw={'width_ratios': [1.25, 1]})
    z0 = zoom_from * x_end
    zmask = grid >= z0
    for ax, mask, title in ((axes[0], slice(None), 'Full run'),
                            (axes[1], zmask, f'Zoom: steps \u2265 {z0:,.0f}')):
        _style(ax)
        for k, (name, _, _) in enumerate(curves):
            ax.plot(grid[mask], ys[k][mask], linewidth=1.6, alpha=0.9,
                    color=TEXT_2 if many else SERIES[k], label=name, zorder=3)
        ax.plot(grid[mask], mean[mask], color=TEXT, linewidth=2.2, linestyle='--',
                label=f'mean of {len(curves)} seeds', zorder=4)
        ax.set_xlabel('Environment steps')
        ax.set_title(title, loc='left', fontsize=10, color=TEXT)
    axes[0].axvspan(z0, x_end, color=GRID, alpha=0.35, linewidth=0, zorder=1)
    axes[0].set_ylabel('Episode return per km')
    axes[0].set_xlim(0, x_end * 1.02)
    # Direct end labels on the zoom panel (spread to avoid overlaps).
    ax = axes[1]
    zy = ys[:, zmask]
    lo, hi = float(zy.min()), float(zy.max())
    pad = 0.08 * max(hi - lo, 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(z0, x_end * 1.14)
    if not many:
        finals = ys[:, -1]
        pos = _spread_labels(finals, 0.06 * (hi - lo + 2 * pad))
        for k, (name, _, _) in enumerate(curves):
            ax.plot([grid[-1]], [finals[k]], 'o', ms=4, color=SERIES[k], zorder=5)
            ax.annotate(f'{name}  {finals[k]:.2f}', (grid[-1], finals[k]),
                        xytext=(grid[-1] + 0.015 * x_end, pos[k]), textcoords='data',
                        va='center', fontsize=8.5, color=TEXT)
    axes[0].legend(fontsize=8.5, frameon=False, loc='lower right', labelcolor=TEXT)
    fig.suptitle(f'Training return per seed (moving average, {window} episodes)',
                 x=0.01, ha='left', fontsize=11, color=TEXT)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Plot DARO reward curves')
    ap.add_argument('log_dirs', nargs='+', help='One or more logs/seed_<seed> directories')
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--out', type=str, default=None, help='Seed-comparison output path')
    args = ap.parse_args()
    for d in args.log_dirs:
        p = plot_training_curve(d, args.window)
        print(f'Reward curve -> {p}')
    if len(args.log_dirs) > 1:
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.log_dirs[0])),
                                       'reward_curve_seeds.png')
        print(f'Seed comparison -> {plot_seed_comparison(args.log_dirs, out, args.window)}')


if __name__ == '__main__':
    main()
