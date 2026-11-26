"""Paper-ready figures (vector PDF), LaTeX tables (booktabs) and a compiled results PDF.

Reads the outputs of evaluation.benchmark and evaluation.paper_results in --src
(default <log_dir>/paper) and writes

    <out>/figures/*.pdf, *.png     one file per figure, sized for a two-column IEEE layout
    <out>/tables/*.tex             booktabs tables (\\usepackage{booktabs})
    <out>/lag_rls_results.pdf      everything above with captions and key findings

    python -m evaluation.paper_pdf [--src logs/paper] [--out logs/paper/pdf]

Requires matplotlib, pandas and reportlab.
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
import re
from datetime import date

import numpy as np
import pandas as pd

from . import EVAL_CONFIG
from .lifetime import missions_to_eol

# ---------------------------------------------------------------------------------
# Names, order, colours (colour follows the method in every figure)
# ---------------------------------------------------------------------------------

FAMILIES = ['tight', 'switch', 'mismatch', 'combined', 'degradation']
FAM_LABEL = {'nominal': 'Nominal', 'tight': 'Tight', 'switch': 'Switch', 'mismatch': 'Mismatch',
             'combined': 'Combined', 'degradation': 'Ageing', 'all': 'All'}
BASELINES = ['AS-MF', 'QoS-EPD', 'BA-EPD', 'BA-EPD+SOH']
PROPOSED = 'Lag-RLS'
REFS = ['Lag-RLS (full info)', 'Oracle-obs', 'Oracle']
ABLATIONS = ['Lag-RLS w/o SOH est', 'Lag-RLS w/o lambda', 'Lag-RLS tier select', 'Lag-RLS w/o RLS',
             'Lag-RLS w/o probe', 'Lag-RLS w/o eta', 'Lag-RLS w/o CD']
DA = ['DA-Lag-RLS (w=1000)', 'DA-Lag-RLS (w=4000)', 'DA-Lag-RLS (w=16000)']
ABL_LABEL = {'Lag-RLS w/o SOH est': 'w/o SOH estimation (SOH = 1)',
             'Lag-RLS w/o lambda': 'w/o Lagrangian price (λ = 0)',
             'Lag-RLS tier select': 'tier λ selection (not cost-based)',
             'Lag-RLS w/o RLS': 'w/o RLS power-model adaptation',
             'Lag-RLS w/o probe': 'w/o speed probing',
             'Lag-RLS w/o eta': 'w/o online loss factor η',
             'Lag-RLS w/o probe+eta': 'w/o probing and w/o η',
             'Lag-RLS w/o CD': 'w/o coordinate descent (grid only)'}

STYLE = {  # method -> (colour, marker, linestyle); validated categorical palette + greys for references
    'Lag-RLS':             ('#2a78d6', 'o', '-'),
    'BA-EPD+SOH':          ('#eb6834', 's', '-'),
    'Lag-RLS w/o SOH est': ('#1baf7a', '^', '--'),
    'BA-EPD':              ('#6250d6', 'v', '-'),
    'QoS-EPD':             ('#eda100', 'D', '-'),
    'AS-MF':               ('#e87ba4', 'P', '-'),
    'Lag-RLS (full info)': ('#52514e', 'x', ':'),
    'Oracle-obs':          ('#8a8986', '+', ':'),
    'Oracle':              ('#0b0b0b', '*', ':'),
}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
SOH_RAMP = ['#b7d3f2', '#86b4ea', '#4f90dd', '#2a6dbd', '#16457f']  # sequential blue, SOH 1.0 -> 0.8


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.4,
        'lines.markersize': 4, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    })
    return plt


COL1, COL2 = 3.5, 7.16  # IEEE column / page width [in]


# ---------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------

def load_raw(path):
    from envs.scenarios import canonical_family
    rows = [json.loads(x) for x in open(path)]
    for r in rows:
        r['family'] = canonical_family(r['family'])
    rows = [r for r in rows if r['family'] in FAMILIES]
    df = pd.DataFrame(rows)
    tags = pd.DataFrame(list(df.pop('tags'))).add_prefix('tag_')
    return pd.concat([df, tags], axis=1)


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype=float)
    r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(df, pol, ref, mask=None):
    d = df if mask is None else df[mask]
    a = d[d.policy == pol].set_index(['scenario', 'seed_i'])['return']
    b = d[d.policy == ref].set_index(['scenario', 'seed_i'])['return']
    diff = (a - b.reindex(a.index)).dropna().values
    lo, hi = boot_ci(diff)
    return float(diff.mean()), lo, hi


def km_to_eol(d):
    """km to SOH 0.8 per C-rate, extrapolated from the degradation family (5 km missions)."""
    out = []
    for c in sorted(d.tag_c.unique()):
        dc = d[d.tag_c == c]
        lv = sorted(dc.tag_soh.unique(), reverse=True)
        fd = [dc[np.isclose(dc.tag_soh, s)].fd_cycle.mean() for s in lv]
        out.append(missions_to_eol(lv, fd) * 5.0)
    return out


# ---------------------------------------------------------------------------------
# Tables: one object -> LaTeX and reportlab
# ---------------------------------------------------------------------------------

class Table:
    def __init__(self, key, caption, header, rows, align=None, bold=None, groups=None, note=None):
        self.key, self.caption, self.header, self.rows = key, caption, header, rows
        self.align = align or ('l' + 'r' * (len(header) - 1))
        self.bold = bold or set()            # {(row, col)}
        self.groups = groups or {}           # row index -> \midrule before it
        self.note = note

    @staticmethod
    def _tex(s):
        s = str(s)
        for a, b in (('%', r'\%'), ('&', r'\&'), ('_', r'\_'), ('#', r'\#')):
            s = s.replace(a, b)
        s = re.sub(r'\blambda\b', r'$\\lambda$', s)
        s = re.sub(r'\beta\b', r'$\\eta$', s)
        for a, b in (('±', r'$\pm$'), ('Δ', r'$\Delta$'), ('≤', r'$\leq$'), ('−', '$-$'), ('→', r'$\to$'),
                     ('*', r'$^{*}$'), ('×', r'$\times$'), ('³', r'$^3$'), ('·', r'$\cdot$'), ('λ', r'$\lambda$'), ('η', r'$\eta$')):
            s = s.replace(a, b)
        return s

    def latex(self):
        L = [r'\begin{table*}[t]' if len(self.header) > 7 else r'\begin{table}[t]', r'\centering',
             r'\caption{' + self._tex(self.caption) + '}', r'\label{tab:' + self.key + '}',
             r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{' + self.align + '}', r'\toprule',
             ' & '.join(self._tex(h) for h in self.header) + r' \\', r'\midrule']
        for i, row in enumerate(self.rows):
            if i in self.groups and i > 0:
                L.append(r'\midrule')
            cells = [(r'\textbf{' + self._tex(c) + '}') if (i, j) in self.bold else self._tex(c)
                     for j, c in enumerate(row)]
            L.append(' & '.join(cells) + r' \\')
        L += [r'\bottomrule', r'\end{tabular}']
        if self.note:
            L.append(r'\par\vspace{2pt}\parbox{\linewidth}{\scriptsize ' + self._tex(self.note) + '}')
        L.append(r'\end{table*}' if len(self.header) > 7 else r'\end{table}')
        return '\n'.join(L) + '\n'


def f3(x):
    return f'{x:.3f}'.replace('-', '−')


def fsd(x):
    return f'{x:+.3f}'.replace('-', '−')


def bold_best(rows, cols, candidates, higher=True):
    """(row, col) of the best entry per column among candidate rows."""
    out = set()
    for j in cols:
        vals = [(float(str(rows[i][j]).replace('−', '-').split()[0].rstrip('*')), i) for i in candidates]
        best = max(vals) if higher else min(vals)
        out |= {(i, j) for v, i in vals if abs(v - best[0]) < 1e-9}
    return out


# ---------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------

def _save(fig, out, name):
    fig.savefig(os.path.join(out, name + '.pdf'))
    fig.savefig(os.path.join(out, name + '.png'), dpi=300)


# --- Motivation (plant model, no controller) --------------------------------------

MOT_C, MOT_L, MOT_LOSS = 2.0, 5000.0, 1.025   # C = 2 workload, 5 km; loss factor fitted to the simulator
MOT_SOC0 = 0.1924                              # initial SOC of the ageing-family C = 2 missions


class _Plant:
    """Steady-speed mission energy and usable battery energy from the simulator's own models."""

    def __init__(self):
        from envs.robot_env import (RobotParams, ECMParams, motor_power, non_motor_power,
                               throughput_freq_required, _voc, quantize_freq)
        self.rp = rp = RobotParams()
        self.ecm, self.voc = ECMParams.default(rp), _voc
        self.v = np.linspace(1.0, rp.v_max, 321)
        fr = throughput_freq_required(self.v, max(rp.qos_ref), 1.2 * MOT_C)
        self.ok = fr <= rp.f_max
        f = quantize_freq(fr)
        self.p_load = motor_power(self.v, 0.0, rp) + non_motor_power(self.v, f, MOT_C, rp)

    def demand(self, soh):
        """Cell-side mission energy [J] vs cruise speed (OCV(SOH), R0(SOH))."""
        soc = 0.12
        r0 = self.ecm.aged(soc, soh)[0]
        u = self.voc(soc, soh)
        i = 2 * self.p_load / (u + np.sqrt(u * u - 4 * r0 * self.p_load))
        return MOT_LOSS * u * i * MOT_L / self.v

    def budget(self, soc0, soh, floor=None):
        """Energy [J] between soc0 and the SOC reserve (capacity and OCV shape scale with SOH)."""
        floor = self.rp.soc_reserve if floor is None else floor
        if soc0 <= floor:
            return 0.0
        s = np.linspace(floor, soc0, 120)
        return float(np.trapezoid([self.voc(x, soh) for x in s], s) * self.rp.Q_n * soh)

    def cost(self, soh):
        """Per-mission energy + time cost of the reward (lower is better)."""
        rp = self.rp
        return rp.w_energy * self.demand(soh) / MOT_L + rp.w_time * rp.v_ref_mps / self.v

    def plan(self, soc0, soh):
        """Best constant speed that ends above the reserve, for a planner that believes (soc0, soh)."""
        e, b = self.demand(soh), self.budget(soc0, soh)
        feas = self.ok & (e <= b)
        if not feas.any():
            return np.nan
        return float(self.v[np.argmin(np.where(feas, self.cost(soh), np.inf))])

    def v_free(self, soh=1.0):
        return float(self.v[np.argmin(np.where(self.ok, self.cost(soh), np.inf))])


def fig_motivation_budget(src, out, plt):
    """(a) the same BMS SOC holds less energy on an aged pack; (b) the BMS SOC drifts above the true SOC."""
    pl = _Plant()
    d = pd.read_csv(os.path.join(src, 'trajectory_BA-EPD.csv'))
    v_un = float(d.v.median())                       # speed the SOH-unaware BA-EPD actually drives
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.35), gridspec_kw={'width_ratios': [1.1, 1]})
    ax = axes[0]
    v, ok = pl.v, pl.ok
    dem = pl.demand(0.8) / 1e3
    ax.plot(v[ok], dem[ok], color=INK, lw=1.6, label='energy needed (5 km mission)')
    ramp = {1.0: SOH_RAMP[1], 0.9: SOH_RAMP[2], 0.8: SOH_RAMP[4]}
    for soh, c in ramp.items():
        b = pl.budget(MOT_SOC0, soh) / 1e3
        ax.axhline(b, color=c, lw=1.2, ls='--', label='energy available above reserve' if soh == 1.0 else None)
        ax.text(5.07, b, f'SOH {soh:.1f}', color=c, fontsize=6.5, va='center', clip_on=False)
        feas = ok & (dem <= b)
        if feas.any() and v[feas].max() < pl.rp.v_max - 1e-6:
            vm = v[feas].max()
            ax.plot([vm], [b], marker='o', color=c, ms=4, zorder=5)
            ax.text(vm + 0.06, b - 0.25, f'{vm:.1f} m/s', color=c, fontsize=6.3, va='top')
    b8 = pl.budget(MOT_SOC0, 0.8) / 1e3
    e_un = float(np.interp(v_un, v, dem))
    ax.annotate('', xy=(v_un, b8), xytext=(v_un, e_un), arrowprops=dict(arrowstyle='<->', color='#e34948', lw=0.9))
    ax.text(v_un + 0.07, (b8 + e_un) / 2, f'shortfall {e_un - b8:.1f} kJ', color='#e34948', fontsize=6.5, va='center')
    ax.plot([v_un], [e_un], marker='X', color=STYLE['BA-EPD+SOH'][0], ms=6.5, zorder=6)
    ax.annotate(f'SOH-unaware choice ({v_un:.1f} m/s)', (v_un, e_un), xytext=(2.45, 30.2), fontsize=6.5,
                color=INK2, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
    ax.set_xlim(1.5, 5.0)
    ax.set_ylim(25, 39)
    ax.set_xlabel('cruise speed [m/s]')
    ax.set_ylabel('energy [kJ]')
    ax.set_title(f'(a) same BMS reading (SOC {MOT_SOC0:.2f}), ageing pack', loc='left')
    ax.legend(loc='upper right', fontsize=6.5)

    ax = axes[1]
    t = d.t / 60
    ax.plot(t, d.soc_bms, color=STYLE['BA-EPD+SOH'][0], lw=1.4, ls='--', label='SOC reported by the BMS')
    ax.plot(t, d.soc_true, color=INK, lw=1.4, label='true SOC')
    ax.axhline(0.05, color=INK2, lw=0.8, ls=':')
    ax.text(0.3, 0.052, 'SOC reserve', fontsize=6.5, color=INK2, va='bottom')
    te = t.iloc[-1]
    ax.text(te + 0.6, d.soc_bms.iloc[-1], f'{d.soc_bms.iloc[-1]:.3f}\nreserve looks intact', fontsize=6.5,
            color=INK2, va='bottom')
    ax.text(te + 0.6, d.soc_true.iloc[-1] - 0.004, f'{d.soc_true.iloc[-1]:.3f}\nreserve violated', fontsize=6.5,
            color=INK2, va='top')
    ax.set_xlim(0, te * 1.45)
    ax.set_ylim(0.02, 0.2)
    ax.set_xlabel('time [min]')
    ax.set_ylabel('SOC')
    ax.set_title('(b) SOH-unaware mission (BA-EPD) on an SOH 0.8 pack', loc='left')
    ax.legend(loc='upper right', fontsize=6.5)
    fig.tight_layout(w_pad=2.2)
    _save(fig, out, 'fig_motivation_budget')
    plt.close(fig)
    return dict(v_un=v_un, shortfall=e_un - b8, b1=pl.budget(MOT_SOC0, 1.0) / 1e3, b8=b8,
                bms_end=float(d.soc_bms.iloc[-1]), true_end=float(d.soc_true.iloc[-1]))


def fig_motivation_life(df, out, plt):
    """Over the service life, the budget error of an SOH-unaware planner vs what speed/DVFS can trade."""
    pl = _Plant()
    d = df[(df.policy == 'Lag-RLS') & (df.family == 'degradation') & np.isclose(df.tag_c, MOT_C)]
    g = d.groupby('tag_soh').fd_cycle.mean()
    F = np.linspace(0.0, -np.log(0.8), 201)
    fd = np.interp(np.exp(-F), g.index.values, g.values)
    km = 5.0 * np.concatenate([[0.0], np.cumsum(0.5 * (1 / fd[1:] + 1 / fd[:-1]) * np.diff(F))]) / 1e3
    soh = np.exp(-F)
    dem = pl.demand(1.0)
    e_ref = float(np.interp(pl.v_free(1.0), pl.v, dem))
    b1 = pl.budget(MOT_SOC0, 1.0)
    over = np.array([100 * (b1 - pl.budget(MOT_SOC0, h)) / e_ref for h in soh])
    e_min = dem[pl.ok].min()
    t35 = 100 * (e_ref - e_min) / e_ref
    t39 = 100 * (float(np.interp(3.9, pl.v, dem)) - e_min) / e_ref
    fig, ax = plt.subplots(figsize=(COL1, 2.45))
    ax.axhspan(0, t39, color='#86b4ea', alpha=0.35, lw=0)
    ax.axhspan(0, t35, color='#4f90dd', alpha=0.35, lw=0)
    ax.text(km[-1] - 0.6, t39 - 0.35, f'energy the speed/DVFS decision\ncan trade: {t39:.1f} % (3.0 to 3.9 m/s)',
            fontsize=6.3, color='#16457f', va='top', ha='right')
    ax.text(km[-1] - 0.6, t35 - 0.25, f'{t35:.1f} % (3.0 to 3.5 m/s)', fontsize=6.3, color='#16457f', va='top',
            ha='right')
    c = STYLE['BA-EPD+SOH'][0]
    ax.plot(km, over, color=c, lw=1.8)
    ax.text(km[-1] - 0.6, over[-1] - 4.5, f'{over[-1]:.0f} % at EOL', color=INK, fontsize=6.5, ha='right', va='top')
    ax.text(21, 12.5, 'budget over-estimate\nwhen SOH is ignored', color=INK, fontsize=7, va='center')
    k = int(np.argmax(over >= t39))
    ax.plot([km[k]], [over[k]], marker='o', color=c, ms=4.5, zorder=5)
    ax.annotate(f'after {km[k]:.1f}' + r'$\times10^3$' + f' km (SOH {soh[k]:.2f})\nthe error exceeds what\n'
                'the decision can compensate', (km[k], over[k]), xytext=(0.8, 17.5),
                fontsize=6.3, color=INK2, va='center', arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
    ax.axvline(km[-1], color=INK2, lw=0.8, ls=':')
    ax.text(km[-1] - 0.4, 29.2, 'EOL', fontsize=6.5, color=INK2, ha='right', va='top')
    ax.set_xlim(0, km[-1] * 1.03)
    ax.set_ylim(0, 30)
    ax.set_xlabel(r'distance driven [$10^3$ km] (repeated 5 km missions, Xu ageing)')
    ax.set_ylabel('% of mission energy')
    top = ax.secondary_xaxis('top', functions=(lambda x: np.interp(x, km, soh), lambda y: np.interp(-y, -soh, km)))
    top.set_xticks([1.0, 0.95, 0.9, 0.85, 0.8])
    top.set_xlabel('SOH', fontsize=7)
    top.tick_params(labelsize=6.5, colors=INK2)
    _save(fig, out, 'fig_motivation_life')
    plt.close(fig)
    return dict(km=float(km[k]), soh=float(soh[k]), over_eol=float(over[-1]), trade39=t39, trade35=t35)


def fig_family(df, out, plt):
    """Paired return difference vs BA-EPD+SOH per scenario family (95 % CI)."""
    pols = ['QoS-EPD', 'BA-EPD', 'Lag-RLS', 'Oracle-obs', 'Oracle']
    fams = FAMILIES + ['all']
    fig, ax = plt.subplots(figsize=(COL2, 2.3))
    w = 0.15
    for k, p in enumerate(pols):
        c, m, _ = STYLE[p]
        xs, ys, lo, hi = [], [], [], []
        for j, f in enumerate(fams):
            mask = None if f == 'all' else (df.family == f)
            mu, a, b = paired(df, p, 'BA-EPD+SOH', mask)
            xs.append(j + (k - 2) * w)
            ys.append(mu)
            lo.append(mu - a)
            hi.append(b - mu)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=m, color=c, ms=4.5, capsize=1.5, elinewidth=1, label=p,
                    mfc=c if p != 'Oracle-obs' else 'none')
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(len(FAMILIES) - 0.5, color=GRID, lw=0.8)
    ax.set_xticks(range(len(fams)), [FAM_LABEL[f] for f in fams])
    ax.set_ylabel('Δ return vs BA-EPD+SOH\n(higher is better)')
    ax.grid(axis='x', visible=False)
    ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.18))
    _save(fig, out, 'fig_family_gain')
    plt.close(fig)


def fig_soh_value(df, out, plt):
    """Ageing family: return and reserve violation vs true SOH."""
    pols = ['BA-EPD', 'BA-EPD+SOH', 'Lag-RLS w/o SOH est', 'Lag-RLS', 'Lag-RLS (full info)']
    d = df[df.family == 'degradation']
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3))
    for p in pols:
        c, m, ls = STYLE[p]
        g = d[d.policy == p].groupby('tag_soh')
        axes[0].plot(g['return'].mean().index, g['return'].mean().values, color=c, marker=m, ls=ls, label=p)
        axes[1].plot(g['reserve_viol'].mean().index, 100 * g['reserve_viol'].mean().values,
                     color=c, marker=m, ls=ls, label=p)
    for ax in axes:
        ax.set_xlabel('true SOH (unknown to the controller)')
        ax.set_xlim(1.005, 0.795)
    axes[0].set_ylabel('return (higher is better)')
    axes[1].set_ylabel('missions ending below SOC reserve [%]')
    axes[0].set_title('(a) mission return', loc='left')
    axes[1].set_title('(b) SOC-reserve violations', loc='left')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    _save(fig, out, 'fig_soh_value')
    plt.close(fig)


def fig_ablation(df, val, out, plt):
    """Paired return difference of each ablation vs full Lag-RLS (all scenarios)."""
    abl = ABLATIONS + (['Lag-RLS w/o probe+eta'] if val is not None else [])
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    ys = np.arange(len(abl))[::-1]
    for y, a in zip(ys, abl):
        for src, dy, col, lab in ((df, 0.17, STYLE['Lag-RLS'][0], 'test seeds'),
                                  (val, -0.17, '#8a8986', 'validation seeds')):
            if src is None or a not in set(src.policy):
                continue
            mu, lo, hi = paired(src, a, 'Lag-RLS')
            ax.errorbar(mu, y + dy, xerr=[[mu - lo], [hi - mu]], fmt='o', color=col, ms=3.5,
                        capsize=1.5, elinewidth=1, label=lab)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ys, [ABL_LABEL[a] for a in abl])
    ax.set_xlabel('Δ return vs full Lag-RLS (< 0: component helps)')
    ax.grid(axis='y', visible=False)
    h, l = ax.get_legend_handles_labels()
    u = dict(zip(l, h))
    ax.legend(u.values(), u.keys(), ncol=2, loc='lower center', bbox_to_anchor=(0.3, 1.0))
    _save(fig, out, 'fig_ablation')
    plt.close(fig)


def fig_soh_est(soh, out, plt):
    """On-line SOH estimation error vs time, per true SOH."""
    ts = [30, 60, 120, 300, 600, 1200]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2))
    for c, h in zip(SOH_RAMP, sorted(soh.soh.unique(), reverse=True)):
        d = soh[np.isclose(soh.soh, h)]
        e = np.stack([d[f'err_{t}s'].values for t in ts])
        axes[0].plot(ts, np.abs(e).mean(1), color=c, marker='o', label=f'SOH {h:.2f}')
        axes[1].plot(ts, e.mean(1), color=c, marker='o')
        axes[1].fill_between(ts, np.percentile(e, 10, 1), np.percentile(e, 90, 1), color=c, alpha=0.12, lw=0)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks(ts, [str(t) for t in ts])
        ax.minorticks_off()
        ax.set_xlabel('time since mission start [s]')
    axes[0].set_ylabel('mean |SOH estimate − true SOH|')
    axes[1].set_ylabel('SOH estimate − true SOH')
    axes[1].axhline(0, color=INK2, lw=0.8)
    axes[0].set_title('(a) absolute error', loc='left')
    axes[1].set_title('(b) bias, mean and 10–90 % band', loc='left')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    _save(fig, out, 'fig_soh_estimation')
    plt.close(fig)


def fig_robustness(rob, out, plt):
    sweeps = [('voltage_noise_mV', 'voltage-sensor noise σ [mV]', None),
              ('cell_var_scale', 'cell-to-cell variation (× nominal)', None),
              ('payload_factor', 'payload (× motor power model)', -2.2),
              ('terrain_v3_factor', 'terrain (× cubic drag term)', -2.2)]
    pols = ['BA-EPD', 'BA-EPD+SOH', 'Lag-RLS w/o SOH est', 'Lag-RLS', 'Lag-RLS (full info)']
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.0))
    for ax, (sw, lab, ymin) in zip(axes, sweeps):
        d = rob[rob.sweep == sw]
        for p in pols:
            c, m, ls = STYLE[p]
            g = d[d.policy == p].groupby('x')['ret'].mean()
            ax.plot(g.index, g.values, color=c, marker=m, ls=ls, label=p)
            if ymin is not None:
                for x, y in g.items():
                    if y < ymin:
                        ax.plot([x], [ymin + 0.05], marker='v', color=c, ms=4, clip_on=False)
                        ax.annotate(f'{y:.1f}'.replace('-', '−'), (x, ymin + 0.05), textcoords='offset points',
                                    xytext=(-3, 4), fontsize=6, color=c, ha='right')
        if ymin is not None:
            ax.set_ylim(ymin, None)
        ax.set_xlabel(lab)
    axes[0].set_ylabel('return')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    _save(fig, out, 'fig_robustness')
    plt.close(fig)


def fig_trajectory(src, out, plt):
    pols = [('Lag-RLS', 'Lag-RLS'), ('Lag-RLS w/o SOH est', 'Lag-RLS_wo_SOH_est'), ('BA-EPD', 'BA-EPD')]
    data = {p: pd.read_csv(os.path.join(src, f'trajectory_{f}.csv')) for p, f in pols}
    fig, axes = plt.subplots(4, 1, figsize=(COL1, 4.6), sharex=True)
    for p, _ in pols:
        c, _, ls = STYLE[p]
        d = data[p]
        t = d.t / 60
        axes[0].plot(t, d.v, color=c, ls=ls, lw=1.1, label=p)
        axes[1].plot(t, d.f, color=c, ls=ls, lw=1.1)
        axes[2].plot(t, d.soc_true, color=c, ls=ls, lw=1.1)
        if p == 'Lag-RLS':
            axes[3].plot(t, d.soh_hat, color=c, lw=1.1, label='Lag-RLS estimate')
        if p == 'Lag-RLS w/o SOH est':
            axes[3].plot(t, d.soh_hat, color=c, ls=ls, lw=1.1, label='assumed (w/o SOH est)')
    axes[2].axhline(0.05, color=INK2, lw=0.8, ls='--')
    axes[2].text(0.2, 0.053, 'SOC reserve', fontsize=6, color=INK2, va='bottom')
    axes[3].axhline(0.8, color=INK, lw=0.8, ls=':', label='true SOH')
    axes[0].set_ylabel('speed [m/s]')
    axes[1].set_ylabel('CPU freq. [GHz]')
    axes[2].set_ylabel('true SOC')
    axes[3].set_ylabel('SOH')
    axes[3].set_xlabel('time [min]')
    axes[3].set_ylim(0.74, 1.03)
    axes[0].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.4))
    axes[3].legend(ncol=1, loc='center right', fontsize=6)
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.4)
    _save(fig, out, 'fig_trajectory')
    plt.close(fig)


# ---------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------

def tab_main(df):
    pols = BASELINES + [PROPOSED] + REFS
    fams = FAMILIES + ['all']
    rows = []
    for p in pols:
        d = df[df.policy == p]
        cells = [d[d.family == f]['return'].mean() if f != 'all' else d['return'].mean() for f in fams]
        bal = np.mean(cells[:-1])
        rows.append([p] + [f3(x) for x in cells] + [f3(bal)])
    cand = list(range(len(BASELINES) + 1))
    t = Table('main_return', 'Mean return per scenario family (higher is better). BMS observations '
              '(SOH unknown) with battery realism; test seed 42, 10 episodes per scenario, 168 scenarios. '
              'Bold: best implementable method (references excluded).',
              ['Method'] + [FAM_LABEL[f] for f in fams] + ['Family avg.'], rows,
              bold=bold_best(rows, range(1, len(fams) + 2), cand), groups={len(BASELINES): 1, len(BASELINES) + 1: 1},
              note='"All" weights episodes (54 % come from Combined); "Family avg." weights the six families '
                   'equally. Lag-RLS (full info) observes true SOC/SOH; Oracle-obs knows the plant and picks the '
                   'best lambda in hindsight; Oracle additionally observes the workload.')
    return t


def tab_paired(df):
    pols = ['AS-MF', 'QoS-EPD', 'BA-EPD', PROPOSED] + REFS
    fams = FAMILIES + ['all']
    rows = []
    for p in pols:
        cells = []
        for f in fams:
            mu, lo, hi = paired(df, p, 'BA-EPD+SOH', None if f == 'all' else df.family == f)
            cells.append(fsd(mu) + ('*' if lo > 0 or hi < 0 else ''))
        rows.append([p] + cells)
    mu, lo, hi = paired(df, PROPOSED, 'BA-EPD+SOH')
    return Table('paired', 'Paired return difference against the strongest baseline BA-EPD+SOH '
                 '(same scenario and seed). * = 95 % bootstrap CI excludes 0.',
                 ['Method'] + [FAM_LABEL[f] for f in fams], rows, groups={3: 1, 4: 1},
                 note=f'Lag-RLS vs BA-EPD+SOH over all scenarios: {fsd(mu)} [{fsd(lo)}, {fsd(hi)}].')


def tab_metrics(df):
    pols = BASELINES + [PROPOSED] + REFS
    rows = []
    for p in pols:
        d = df[df.policy == p]
        dt = d.decision_us.mean()
        rows.append([p, f'{100 * d.completed.mean():.1f}', f'{d.mission_time_s.mean() / 60:.2f}',
                     f'{d.energy_kj.mean():.2f}', f'{d.qos_viol_pct.mean():.2f}',
                     f'{100 * d.reserve_viol.mean():.1f}', f'{d.delta_soh_ppm.mean():.2f}',
                     '–' if np.isnan(dt) else ('<0.001' if dt < 1 else (f'{dt / 1000:.2f}' if dt >= 100
                                                                       else f'{dt / 1000:.3f}'))])
    cand = list(range(len(BASELINES) + 1))
    b = bold_best(rows, [1], cand) | bold_best(rows, [3, 4, 5, 6], cand, higher=False)
    return Table('metrics', 'Short- and long-term metrics, mean over all scenarios and episodes.',
                 ['Method', 'Done [%]', 'Time [min]', 'Energy [kJ]', 'QoS viol. [%]', 'Reserve viol. [%]',
                  'ΔSOH [ppm/mission]', 'Decision [ms]'], rows, bold=b,
                 groups={len(BASELINES): 1, len(BASELINES) + 1: 1},
                 note='Reserve viol.: share of missions ending below SOC 0.05. ΔSOH: Xu et al. cycle-ageing '
                      'model per mission. Decision time: Python, one core, per control step (1 s).')


def tab_ageing(df):
    pols = BASELINES + [PROPOSED, 'Lag-RLS w/o SOH est'] + REFS
    d = df[df.family == 'degradation']
    sohs = [1.0, 0.9, 0.85, 0.8]
    rows = []
    for p in pols:
        dp = d[d.policy == p]
        cells = []
        for s in sohs:
            x = dp[np.isclose(dp.tag_soh, s)]
            cells += [f3(x['return'].mean()), f'{100 * x.reserve_viol.mean():.0f}']
        km = km_to_eol(dp)
        rows.append([p] + cells + [f'{np.mean(km) / 1000:.2f}'])
    cand = list(range(len(BASELINES) + 2))
    b = bold_best(rows, [1, 3, 5, 7], cand) | bold_best(rows, [2, 4, 6, 8], cand, higher=False)
    hdr = ['Method']
    for s in sohs:
        hdr += [f'R @{s:.2f}', f'Res.% @{s:.2f}']
    return Table('ageing', 'Ageing family: same 5 km mission and the same initial charge on a pack of '
                 'decreasing (unknown) SOH. R = return, Res.% = missions ending below the SOC reserve.',
                 hdr + ['km to EOL [10³ km]'], rows, bold=b,
                 groups={len(BASELINES): 1, len(BASELINES) + 2: 1},
                 note='km to EOL: distance until SOH 0.8 under repeated missions (Xu model, mean over C = 1, 2, 3).')


def tab_ablation(df, val):
    abl = ABLATIONS + (['Lag-RLS w/o probe+eta'] if val is not None else [])
    rows = []
    for a in abl:
        row = [ABL_LABEL[a]]
        for src in (df, val):
            if src is None or a not in set(src.policy):
                row += ['–', '–']
                continue
            mu, lo, hi = paired(src, a, 'Lag-RLS')
            row += [fsd(mu) + ('*' if lo > 0 or hi < 0 else ''), f'{100 * src[src.policy == a].reserve_viol.mean():.1f}']
        rows.append(row)
    ref = ['Full Lag-RLS (reference)', '0', f'{100 * df[df.policy == "Lag-RLS"].reserve_viol.mean():.1f}']
    ref += (['0', f'{100 * val[val.policy == "Lag-RLS"].reserve_viol.mean():.1f}'] if val is not None else ['–', '–'])
    return Table('ablation', 'Ablation: paired return difference against the full Lag-RLS (all scenarios; '
                 '< 0 means the removed component helps). * = 95 % CI excludes 0.',
                 ['Variant', 'Δ return (test)', 'Res.% (test)', 'Δ return (val.)', 'Res.% (val.)'],
                 [ref] + rows, groups={1: 1},
                 note='Test: base seed 42; validation: base seed 2024 (different episodes); 168 scenarios x 10 episodes each.')


def tab_soh_est(soh):
    ts = [30, 60, 120, 300, 600, 1200]
    rows = []
    for h in sorted(soh.soh.unique(), reverse=True):
        d = soh[np.isclose(soh.soh, h)]
        e = d.err_end.values
        rows.append([f'{h:.2f}'] + [f"{d[f'err_{t}s'].abs().mean():.3f}" for t in ts]
                    + [fsd(e.mean()), f'{np.percentile(np.abs(e), 95):.3f}', f'{np.nanmedian(d.t_conv_s):.0f}'])
    return Table('soh_est', 'On-line SOH estimation from BMS voltage/current (true SOH unknown). '
                 'Mean absolute error at time t, bias and 95th percentile of |error| at mission end, '
                 'median time after which |error| stays ≤ 0.02.',
                 ['True SOH'] + [f'|e| {t}s' for t in ts] + ['Bias (end)', 'P95 |e|', 't_conv [s]'], rows,
                 note='5 SOH levels × C = 1, 2, 3 × full/charge-limited start; BMS + battery realism. '
                      'The estimate is biased low, i.e. conservative.')


def tab_robustness(rob):
    sweeps = [('voltage_noise_mV', 'Voltage noise [mV]'), ('cell_var_scale', 'Cell variation [×]'),
              ('payload_factor', 'Payload [×]'), ('terrain_v3_factor', 'Terrain v³ [×]')]
    pols = ['BA-EPD', 'BA-EPD+SOH', 'Lag-RLS w/o SOH est', 'Lag-RLS', 'Lag-RLS (full info)']
    rows, groups = [], {}
    for sw, lab in sweeps:
        d = rob[rob.sweep == sw]
        xs = sorted(d.x.unique())
        groups[len(rows)] = 1
        rows.append([lab] + [f'{x:g}' for x in xs] + [''] * (4 - len(xs)))
        for p in pols:
            dp = d[d.policy == p]
            rows.append(['   ' + p] + [f"{f3(dp[dp.x == x].ret.mean())} / {100 * dp[dp.x == x].reserve_viol.mean():.0f}"
                                      for x in xs] + [''] * (4 - len(xs)))
    return Table('robustness', 'Robustness sweeps: return / reserve violation [%].',
                 ['Sweep / method', 'x1', 'x2', 'x3', 'x4'], rows, groups=groups,
                 note='Voltage noise and cell variation on charge-limited scenarios (tight + ageing C = 2); '
                      'payload/terrain: plant differs from the controller model (SOH 1.0/0.85, k = 1.0/1.1).')


def tab_da(df):
    pols = [PROPOSED] + DA + ['BA-EPD+SOH']
    ws = (1000.0, 4000.0, 16000.0)
    rows = []
    for p in pols:
        d = df[df.policy == p]
        R, D = d['return'].mean(), d.delta_soh_ppm.mean()
        km = np.mean(km_to_eol(d[d.family == 'degradation']))
        rows.append([p, f'{R:.4f}'.replace('-', '−'), f'{D:.2f}', f'{km / 1000:.2f}']
                    + [f'{R - w * D * 1e-6 / 0.2:.4f}'.replace('-', '−') for w in ws])
    return Table('degradation_aware', 'Adding the predicted Xu ageing to the objective (DA-Lag-RLS, '
                 'J = return − w·ΔSOH/(1 − SOH_EOL)). Mean over all scenarios.',
                 ['Method', 'Return', 'ΔSOH [ppm]', 'km to EOL [10³]', 'J (w=1000)', 'J (w=4000)', 'J (w=16000)'],
                 rows, groups={len(pols) - 1: 1},
                 note='Even w = 16000 reduces ΔSOH by only 0.3 %: in the Xu model, fade is driven by depth of '
                      'discharge (mission energy) and the SOC window, not by C-rate, and the pack runs at ~0.15 C. '
                      'Speed/DVFS control therefore barely changes long-term fade.')


# ---------------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------------

def build_pdf(path, figs, tabs, text, fig_dir):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                    Table as RLTable, TableStyle)
    import matplotlib
    fd = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
    pdfmetrics.registerFont(TTFont('Stix', os.path.join(fd, 'STIXGeneral.ttf')))
    pdfmetrics.registerFont(TTFont('Stix-B', os.path.join(fd, 'STIXGeneralBol.ttf')))
    pdfmetrics.registerFont(TTFont('Stix-I', os.path.join(fd, 'STIXGeneralItalic.ttf')))
    pdfmetrics.registerFontFamily('Stix', normal='Stix', bold='Stix-B', italic='Stix-I', boldItalic='Stix-B')

    body = ParagraphStyle('b', fontName='Stix', fontSize=9.5, leading=12.5, alignment=TA_JUSTIFY, spaceAfter=5)
    h1 = ParagraphStyle('h1', fontName='Stix-B', fontSize=13, leading=16, spaceBefore=8, spaceAfter=5, keepWithNext=1)
    title = ParagraphStyle('t', fontName='Stix-B', fontSize=17, leading=21, spaceAfter=4)
    cap = ParagraphStyle('c', fontName='Stix', fontSize=8.5, leading=10.5, alignment=TA_JUSTIFY, spaceAfter=3)
    cell = ParagraphStyle('cell', fontName='Stix', fontSize=7.3, leading=8.6)
    cellb = ParagraphStyle('cellb', parent=cell, fontName='Stix-B')
    small = ParagraphStyle('s', fontName='Stix', fontSize=7.5, leading=9.5, textColor=colors.HexColor(INK2))
    W = A4[0] - 36 * mm

    def esc(s):
        return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def table(t, n):
        data = [[Paragraph('<b>' + esc(h) + '</b>', cell) for h in t.header]]
        for i, r in enumerate(t.rows):
            data.append([Paragraph(esc(c), cellb if (i, j) in t.bold else cell) for j, c in enumerate(r)])
        first = W * (0.30 if len(t.header) <= 6 else 0.20)
        cw = [first] + [(W - first) / (len(t.header) - 1)] * (len(t.header) - 1)
        st = [('LINEABOVE', (0, 0), (-1, 0), 0.9, colors.black), ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.black),
              ('LINEBELOW', (0, -1), (-1, -1), 0.9, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
              ('TOPPADDING', (0, 0), (-1, -1), 1.2), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.2),
              ('LEFTPADDING', (0, 0), (-1, -1), 2.5), ('RIGHTPADDING', (0, 0), (-1, -1), 2.5)]
        for i in t.groups:
            if i > 0:
                st.append(('LINEABOVE', (0, i + 1), (-1, i + 1), 0.4, colors.HexColor(INK2)))
        tb = RLTable(data, colWidths=cw, repeatRows=1)
        tb.setStyle(TableStyle(st))
        parts = [Paragraph(f'<b>Table {n}.</b> ' + esc(t.caption), cap), tb]
        if t.note:
            parts.append(Paragraph(esc(t.note), small))
        parts.append(Paragraph(f'LaTeX: tables/{t.key}.tex', small))
        return KeepTogether(parts + [Spacer(1, 8)])

    def figure(name, caption, n, width):
        from PIL import Image as PI
        p = os.path.join(fig_dir, name + '.png')
        w, h = PI.open(p).size
        wd = min(W, width)
        return KeepTogether([Image(p, width=wd, height=wd * h / w),
                             Paragraph(f'<b>Fig. {n}.</b> ' + esc(caption), cap),
                             Paragraph(f'Vector: figures/{name}.pdf', small), Spacer(1, 8)])

    story = [Paragraph('Lag-RLS results: battery-state-aware speed and DVFS co-control', title),
             Paragraph(f'Generated {date.today().isoformat()} by evaluation/paper_pdf.py from '
                       'logs/paper (benchmark, paper_results).', small), Spacer(1, 6)]
    tn, fn = [1], [1]

    def add(kind, *a):
        if kind == 'h':
            story.append(Paragraph(a[0], h1))
        elif kind == 'p':
            story.append(Paragraph(a[0], body))
        elif kind == 't':
            story.append(table(tabs[a[0]], tn[0]))
            tn[0] += 1
        elif kind == 'f':
            story.append(figure(a[0], figs[a[0]], fn[0], W * (a[1] if len(a) > 1 else 1.0)))
            fn[0] += 1
        elif kind == 'br':
            story.append(PageBreak())

    for item in text:
        add(*item)
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title='Lag-RLS results',
                            author='DARO evaluation')

    def footer(c, d):
        c.setFont('Stix', 8)
        c.setFillColor(colors.HexColor(INK2))
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, str(d.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# ---------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='paper figures, tables and results PDF')
    ap.add_argument('--src', default=os.path.join(EVAL_CONFIG['log_dir'], 'paper'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    src = args.src
    out = args.out or os.path.join(src, 'pdf')
    fig_dir, tab_dir = os.path.join(out, 'figures'), os.path.join(out, 'tables')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    df = load_raw(os.path.join(src, 'benchmark', 'raw.jsonl'))
    vp = os.path.join(src, 'val_probe_eta', 'raw.jsonl')
    val = load_raw(vp) if os.path.exists(vp) else None
    soh = pd.read_csv(os.path.join(src, 'soh_accuracy.csv'))
    rob = pd.read_csv(os.path.join(src, 'robustness.csv'))

    plt = _mpl()
    mb = fig_motivation_budget(src, fig_dir, plt)
    ml = fig_motivation_life(df, fig_dir, plt)
    fig_family(df, fig_dir, plt)
    fig_soh_value(df, fig_dir, plt)
    fig_ablation(df, val, fig_dir, plt)
    fig_soh_est(soh, fig_dir, plt)
    fig_robustness(rob, fig_dir, plt)
    fig_trajectory(src, fig_dir, plt)

    tabs = {t.key: t for t in (tab_main(df), tab_paired(df), tab_metrics(df), tab_ageing(df),
                               tab_ablation(df, val), tab_soh_est(soh), tab_robustness(rob), tab_da(df))}
    for k, t in tabs.items():
        with open(os.path.join(tab_dir, k + '.tex'), 'w') as f:
            f.write(t.latex())

    # numbers quoted in the text
    g = lambda p: df[df.policy == p]
    R = {p: g(p)['return'].mean() for p in set(df.policy)}
    RV = {p: 100 * g(p).reserve_viol.mean() for p in set(df.policy)}
    mu, lo, hi = paired(df, 'Lag-RLS', 'BA-EPD+SOH')
    deg8 = df[(df.family == 'degradation') & np.isclose(df.tag_soh, 0.8)]
    d8 = {p: (deg8[deg8.policy == p]['return'].mean(), 100 * deg8[deg8.policy == p].reserve_viol.mean())
          for p in set(df.policy)}
    s = lambda x: f'{x:.3f}'.replace('-', '−')
    val_txt = ''
    if val is not None:
        vv = {a: paired(val, a, 'Lag-RLS') for a in ['Lag-RLS w/o probe', 'Lag-RLS w/o eta', 'Lag-RLS w/o probe+eta']}
        tt = {a: paired(df, a, 'Lag-RLS') for a in ['Lag-RLS w/o probe', 'Lag-RLS w/o eta']}
        val_txt = (f'On the validation seeds, removing probing changes the return by {fsd(vv["Lag-RLS w/o probe"][0])} '
                   f'[{fsd(vv["Lag-RLS w/o probe"][1])}, {fsd(vv["Lag-RLS w/o probe"][2])}] '
                   f'(test: {fsd(tt["Lag-RLS w/o probe"][0])}), removing eta by {fsd(vv["Lag-RLS w/o eta"][0])} '
                   f'[{fsd(vv["Lag-RLS w/o eta"][1])}, {fsd(vv["Lag-RLS w/o eta"][2])}] '
                   f'(test: {fsd(tt["Lag-RLS w/o eta"][0])}), and removing both by '
                   f'{fsd(vv["Lag-RLS w/o probe+eta"][0])} [{fsd(vv["Lag-RLS w/o probe+eta"][1])}, '
                   f'{fsd(vv["Lag-RLS w/o probe+eta"][2])}].')

    figs = {
        'fig_motivation_budget': f'Motivation 1 - the long-term state is hidden behind the BMS reading. (a) Energy above '
                                 f'the SOC reserve from the same BMS reading (SOC {MOT_SOC0:.2f}) for true SOH 1.0/0.9/0.8 '
                                 f'(capacity, OCV shape and R0 all scale with SOH) against the energy a 5 km mission needs '
                                 f'at each cruise speed (C = {MOT_C:g}). The maximum feasible speed falls from above 5 to '
                                 f'3.3 m/s; an SOH-unaware planner believes {mb["b1"]:.1f} kJ are available and drives '
                                 f'{mb["v_un"]:.1f} m/s, {mb["shortfall"]:.1f} kJ more than an SOH 0.8 pack holds. '
                                 f'(b) The simulated mission: the BMS counts charge against the rated capacity, so it '
                                 f'reports {mb["bms_end"]:.3f} at the end while the true SOC is {mb["true_end"]:.3f}. '
                                 f'Steady-speed model of the simulator (within 3 % of simulated mission energy).',
        'fig_motivation_life': f'Motivation 2 - why the long-term state must be tracked online. Over the service life '
                               f'(repeated 5 km missions, Xu ageing, C = {MOT_C:g}), for a charge-limited mission (BMS SOC '
                               f'{MOT_SOC0:.2f} at the start, as in Fig. 1), the energy budget of a planner that '
                               f'ignores SOH is over-estimated by up to {ml["over_eol"]:.0f} % of the mission energy at end '
                               f'of life. The whole range the speed/DVFS decision can trade is only {ml["trade39"]:.1f} % '
                               f'(3.0 to 3.9 m/s); after {ml["km"]:.1f}' + '×10³ km (SOH ' + f'{ml["soh"]:.2f}) the '
                               f'error is larger than anything a late correction can recover.',
        'fig_family_gain': 'Paired return difference against BA-EPD+SOH per scenario family (mean and 95 % '
                           'bootstrap CI; same scenario and seed). Lag-RLS gains most where the charge is '
                           'limited (Tight, Combined) and matches the baselines in Nominal, where the battery '
                           'constraint is inactive.',
        'fig_soh_value': 'Value of long-term battery-state awareness. Same mission and initial charge on a pack '
                         'of decreasing SOH (unknown, estimated online). Without the SOH estimate the controller '
                         'over-estimates the usable energy and ends below the SOC reserve as the pack ages.',
        'fig_ablation': 'Ablation of Lag-RLS components: paired return difference against the full method '
                        '(95 % CI), test seeds and, for the probing/eta components, validation seeds.',
        'fig_soh_estimation': 'On-line SOH estimation from BMS voltage and current. Error decreases with time; '
                              'the estimate is biased low (conservative) and is least accurate near end of '
                              'life (SOH 0.8).',
        'fig_robustness': 'Robustness: voltage-sensor noise, cell-to-cell variation (both on charge-limited '
                          'missions) and plant mismatch in payload and terrain. Off-scale values are marked '
                          'with a triangle and labelled.',
        'fig_trajectory': 'Example mission on an aged pack (true SOH 0.8, C = 2, charge-limited). Lag-RLS '
                          'identifies the reduced capacity within the first minutes, raises the energy price '
                          'and slows down, ending at the SOC reserve; the SOH-unaware variants drive faster and '
                          'end below it.',
    }
    text = [
        ('h', '0  Motivation'),
        ('p', 'Speed and DVFS decide how much battery energy a mission uses, so the controller needs the energy budget. '
              'The short-term state (SOC) says how much charge is left; the long-term state (SOH: capacity, OCV shape, '
              'internal resistance) says how much energy that charge is worth. A BMS reports SOC against the rated '
              'capacity and does not report SOH, so the same reading over-states the energy of an aged pack (Fig. 1) '
              'and the BMS does not notice the resulting reserve violation. Over the service life this error quickly '
              'exceeds the entire energy range the speed/DVFS decision can trade (Fig. 2). Lag-RLS therefore prices '
              'the short-term budget with a Lagrange multiplier and estimates the long-term state online.'),
        ('f', 'fig_motivation_budget'), ('f', 'fig_motivation_life', 0.62),
        ('h', '1  Setting and summary'),
        ('p', 'All methods observe the battery only through a BMS (SOC from a coulomb counter on the rated '
              'capacity; SOH, capacity, internal resistance and OCV shape unknown) and run with battery realism '
              '(sensor noise, BMS gain error, per-episode cell variation). Test suite: 159 scenarios in five '
              'families (charge-limited "tight", workload switching, plant mismatch, tight + mismatch, '
              'ageing with SOH 0.8-1.0), 10 episodes each, test seed 42. SOH 0.8 is end of life.'),
        ('p', f'<b>Main result.</b> Lag-RLS attains a mean return of {s(R["Lag-RLS"])} versus {s(R["BA-EPD+SOH"])} '
              f'for the strongest baseline BA-EPD+SOH (paired difference {fsd(mu)}, 95 % CI [{fsd(lo)}, {fsd(hi)}]) '
              f'and reduces SOC-reserve violations from {RV["BA-EPD+SOH"]:.1f} % to {RV["Lag-RLS"]:.1f} %. It is '
              f'better than the hindsight-lambda oracle on the controller model (Oracle-obs, {s(R["Oracle-obs"])}) '
              f'and {R["Oracle"] - R["Lag-RLS"]:.3f} below the full Oracle ({s(R["Oracle"])}).'),
        ('p', f'<b>Long-term state.</b> On an aged pack (SOH 0.8) Lag-RLS keeps the return at '
              f'{s(d8["Lag-RLS"][0])} with {d8["Lag-RLS"][1]:.0f} % reserve violations; without the SOH estimate the '
              f'return drops to {s(d8["Lag-RLS w/o SOH est"][0])} with {d8["Lag-RLS w/o SOH est"][1]:.0f} % '
              f'violations, close to the full-information bound {s(d8["Lag-RLS (full info)"][0])} / '
              f'{d8["Lag-RLS (full info)"][1]:.0f} %.'),
        ('t', 'main_return'), ('t', 'paired'), ('f', 'fig_family_gain'), ('t', 'metrics'),
        ('h', '2  Short-term and long-term battery state'),
        ('p', 'The ageing family keeps the mission and the initial charge fixed while the (unknown) SOH decreases. '
              'This isolates the effect of the long-term battery state on the decision.'),
        ('f', 'fig_soh_value'), ('t', 'ageing'), ('f', 'fig_trajectory', 0.62),
        ('h', '3  Ablation'),
        ('p', 'Each row removes one component. Cost-based lambda selection, the Lagrangian price, the SOH estimate '
              'and the coordinate-descent refinement all contribute. RLS power-model adaptation is essential under '
              'plant mismatch (Combined). ' + val_txt + ' Probing is therefore '
              'dispensable in this scenario suite (it helps only marginally under mismatch), while eta leaves the '
              'return unchanged but slightly lowers reserve violations.'),
        ('f', 'fig_ablation', 0.62), ('t', 'ablation'),
        ('h', '4  On-line SOH estimation'),
        ('f', 'fig_soh_estimation'), ('t', 'soh_est'),
        ('h', '5  Robustness'),
        ('p', 'Sensor noise up to 60 mV has no measurable effect. Doubling cell-to-cell variation degrades the SOH '
              'estimate until it is no better than assuming SOH = 1 (limitation; joint estimation of R0 and OCV '
              'offset is a possible remedy). Under payload or terrain mismatch Lag-RLS stays close to the '
              'full-information variant, while the BA-EPD baselines collapse.'),
        ('f', 'fig_robustness'), ('t', 'robustness'),
        ('h', '6  Ageing in the objective (negative result)'),
        ('p', 'Lag-RLS reports Xu-model fade and km to end of life, but speed/DVFS control is not a lever on '
              'long-term fade: adding the predicted fade to the objective (DA-Lag-RLS) changes ΔSOH by at most '
              '0.3 %. Claim battery-state awareness, not ageing mitigation; charging and SOC-window decisions over '
              'mission sequences are the lever that changes fade (future work).'),
        ('t', 'degradation_aware'),
    ]
    build_pdf(os.path.join(out, 'lag_rls_results.pdf'), figs, tabs, text, fig_dir)
    print(f'figures -> {fig_dir}\ntables  -> {tab_dir}\nPDF     -> {os.path.join(out, "lag_rls_results.pdf")}')


if __name__ == '__main__':
    main()
