"""Experiment figures from the benchmark (logs/paper/<run>/raw.jsonl, 10% complexity jitter).

    python figures/exp_results.py [--run benchmark_td3]

Main comparison: AS-MF, QoS-EPD, BA-EPD, BA-EPD+SOH, Lag-RLS, DA-Lag-RLS. Lag-RLS variants
(incl. full info) are in the ablation figures; the Oracle is not shown. If logs/paper/benchmark_sensing exists
(same seeds), its extra-sensor variants are merged into the ablation.

-> figures/pdf/exp_overall.pdf       cost (-return) per scenario family, 95% CI
   figures/pdf/exp_tradeoff.pdf      (a) time vs energy, (b) QoS vs reserve violation
   figures/pdf/exp_aging.pdf         degradation family: (a) cost, (b) reserve violation vs SOH
   figures/pdf/exp_longterm.pdf      (a) dSOH vs mission time (w_D sweep), (b) km to EOL
   figures/pdf/exp_ablation.pdf      paired change in cost vs Lag-RLS per removed component
   figures/pdf/exp_ablation_soh.pdf  battery-state information under aging (degradation family)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.lifetime import missions_to_eol  # noqa: E402

RUN = sys.argv[sys.argv.index('--run') + 1] if '--run' in sys.argv else 'benchmark_td3'

RAW = ROOT / 'logs' / 'paper' / RUN / 'raw.jsonl'
RAW_SENSING = ROOT / 'logs' / 'paper' / 'benchmark_sensing' / 'raw.jsonl'
OUT = ROOT / 'figures' / 'pdf'
FS = 15
plt.rcParams.update({'font.size': FS, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.titlesize': FS, 'legend.fontsize': FS - 2})

DA = 'DA-Lag-RLS (w=16000)'
FAMILIES = [('nominal', 'Nominal'), ('tight', 'Tight'), ('switch', 'Switch'),
            ('mismatch', 'Mismatch'), ('tight_mismatch', 'Tight-\nmismatch'),
            ('degradation', 'Degra-\ndation'), ('all', 'All')]
# same colours as the trace figures
STYLE = {
    'AS-MF':               dict(color='#9a9a9a', ls=':', marker='X', label='AS-MF'),
    'QoS-EPD':             dict(color='#8c6bb1', ls='-.', marker='v', label='QoS-EPD'),
    'BA-EPD':              dict(color='#e08a2c', ls='--', marker='s', label='BA-EPD'),
    'BA-EPD+SOH':          dict(color='#c8413b', ls='--', marker='D', label='BA-EPD+SOH'),
    'Lag-RLS w/o SOH est': dict(color='#7fa6d6', ls=':', marker='x', label='Lag-RLS w/o SOH est.'),
    'Lag-RLS':             dict(color='#2a5ea8', ls='-', marker='o', label='Lag-RLS'),
    DA:                    dict(color='#1a8a4a', ls='-', marker='^', label='DA-Lag-RLS'),
    'Lag-RLS (full info)': dict(color='#555555', ls=(0, (1, 1)), marker='P', label='Lag-RLS (full info)'),
    'Oracle':              dict(color='#222222', ls=(0, (5, 2)), marker='*', label='Oracle'),
}
MAIN = ['AS-MF', 'QoS-EPD', 'BA-EPD', 'BA-EPD+SOH', DA]


def load():
    rows = [json.loads(line) for line in RAW.read_text().splitlines()]
    if RAW_SENSING.exists():
        rows += [json.loads(line) for line in RAW_SENSING.read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r['policy'], []).append(r)
    return by


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def sel(rs, fam):
    return [r for r in rs if fam == 'all' or r['family'] == fam]


def mean(rs, k):
    return float(np.nanmean([r[k] for r in rs]))


def save(fig, name):
    p = OUT / f'{name}.pdf'
    fig.savefig(p, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(p)


# ------------------------------------------------------------------------------------------
def fig_overall(by):
    """Cost = -return per family; bars above Y_MAX (AS-MF) are clipped and labelled."""
    Y_MAX = 1.7
    fig, ax = plt.subplots(figsize=(12, 4))
    nf, nm = len(FAMILIES), len(MAIN)
    w = 0.8 / nm
    for j, n in enumerate(MAIN):
        st = STYLE[n]
        y, lo, hi = [], [], []
        for fam, _ in FAMILIES:
            x = [-r['return'] for r in sel(by[n], fam)]
            a, b = boot_ci(x)
            y.append(np.mean(x)); lo.append(np.mean(x) - a); hi.append(b - np.mean(x))
        xs = np.arange(nf) + (j - (nm - 1) / 2) * w
        ref = n == 'Oracle'
        ax.bar(xs, y, w * 0.92, color='white' if ref else st['color'], ec=st['color'],
               hatch='///' if ref else None, lw=1.2, label=st['label'], zorder=2)
        y = np.array(y)
        clip = y > Y_MAX
        ax.errorbar(xs[~clip], y[~clip], yerr=[np.array(lo)[~clip], np.array(hi)[~clip]],
                    fmt='none', ecolor='#333333', lw=1, capsize=2, zorder=3)
        for x0, y0 in zip(xs[clip], y[clip]):
            ax.text(x0, Y_MAX * 0.98, f'{y0:.2f}', rotation=90, ha='center', va='top',
                    fontsize=FS - 3, color='white', zorder=4)
    ax.axvline(nf - 1.5, color='#888888', lw=0.8, ls=':')
    ax.set_xticks(np.arange(nf))
    ax.set_xticklabels([lab for _, lab in FAMILIES])
    ax.set_ylabel(r'Cost $-R$ (lower is better)')
    ax.set_ylim(0, Y_MAX)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.legend(ncol=nm, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=True,
              handlelength=1.5, columnspacing=1.0)
    save(fig, 'exp_overall')


# ------------------------------------------------------------------------------------------
def fig_tradeoff(by):
    names = MAIN
    sty = STYLE
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    for n in names:
        rs, st = by[n], sty[n]
        kw = dict(color=st['color'], marker=st['marker'], ms=11,
                  mec='black', mew=0.6, ls='none', label=st['label'], zorder=3)
        a.plot(mean(rs, 'mission_time_s') / 60, mean(rs, 'energy_kj'), **kw)
        b.plot(mean(rs, 'qos_viol_pct'), 100 * mean(rs, 'reserve_viol'), **kw)
    a.set_xlabel('Mission time (min)')
    a.set_ylabel('Energy (kJ)')
    a.set_title('(a) Mission time vs. energy')
    b.set_xlabel('QoS violation (% of steps)')
    b.set_ylabel('Reserve violation (% of missions)')
    b.set_title('(b) QoS vs. reserve violation')
    for ax in (a, b):
        ax.grid(alpha=0.3)
        ax.annotate('better', xy=(0.04, 0.06), xytext=(0.2, 0.3), xycoords='axes fraction',
                    textcoords='axes fraction', fontsize=FS - 2, ha='center', va='center',
                    arrowprops=dict(arrowstyle='->', color='#555555'), color='#555555')
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, ncol=len(names), loc='lower center', bbox_to_anchor=(0.5, 0.97),
               frameon=True, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    save(fig, 'exp_tradeoff')


# ------------------------------------------------------------------------------------------
MAIN_AGING = ['QoS-EPD', 'BA-EPD', 'BA-EPD+SOH', DA]   # AS-MF: cost 1.1-2.9, off the scale
ABL_AGING = ['Lag-RLS w/o SOH est', 'Lag-RLS', DA, 'Lag-RLS (full info)']


def fig_aging(by, names=MAIN_AGING, name='exp_aging', ncol=4):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    for n in names:
        d = [r for r in by[n] if r['family'] == 'degradation']
        sohs = sorted({r['tags']['soh'] for r in d}, reverse=True)
        at = lambda s: [r for r in d if abs(r['tags']['soh'] - s) < 1e-6]
        cost = [-mean(at(s), 'return') for s in sohs]
        res = [100 * mean(at(s), 'reserve_viol') for s in sohs]
        st = STYLE[n]
        kw = dict(color=st['color'], ls=st['ls'], marker=st['marker'], lw=1.8, ms=7,
                  label=st['label'])
        a.plot(100 * np.array(sohs), cost, **kw)
        b.plot(100 * np.array(sohs), res, **kw)
    a.set_ylabel(r'Cost $-R$')
    a.set_title('(a) Cost with SOH')
    b.set_ylabel('Reserve violation (%)')
    b.set_title('(b) Reserve violation with SOH')
    for ax in (a, b):
        ax.set_xlabel('SOH (%)')
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, ncol=ncol, loc='lower center', bbox_to_anchor=(0.5, 0.97), frameon=True,
               handlelength=2.2, columnspacing=1.0)
    fig.tight_layout()
    save(fig, name)


# ------------------------------------------------------------------------------------------
def km_to_eol(rs):
    d = [r for r in rs if r['family'] == 'degradation']
    out = []
    for c in (1.0, 2.0, 3.0):
        lv = sorted({r['tags']['soh'] for r in d if r['tags']['c'] == c}, reverse=True)
        fd = [np.mean([r['fd_cycle'] for r in d if r['tags']['c'] == c
                       and abs(r['tags']['soh'] - s) < 1e-6]) for s in lv]
        out.append(missions_to_eol(lv, fd) * 5.0)
    return np.array(out)


def fig_longterm(by):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    # (a) w_D sweep: Lag-RLS = w_D 0
    sweep = [('Lag-RLS', 0), ('DA-Lag-RLS (w=1000)', 1000), ('DA-Lag-RLS (w=4000)', 4000), (DA, 16000)]
    t = [mean(by[n], 'mission_time_s') / 60 for n, _ in sweep]
    s = [mean(by[n], 'delta_soh_ppm') for n, _ in sweep]
    a.plot(t, s, '-', color=STYLE[DA]['color'], lw=1.8, zorder=2)
    offs = {0: (-20, 6), 1000: (8, 6), 4000: (8, 4), 16000: (8, 4)}
    for (n, w), x, y in zip(sweep, t, s):
        a.plot(x, y, 'o', color=STYLE[DA]['color'], mec='black', mew=0.6, ms=9, zorder=3)
        a.annotate(f'{w}', (x, y), xytext=offs[w], textcoords='offset points', fontsize=FS - 3)
    for n in ['QoS-EPD', 'BA-EPD', 'BA-EPD+SOH']:   # AS-MF (29.9 min, 32.7 ppm) off the scale
        st = STYLE[n]
        a.plot(mean(by[n], 'mission_time_s') / 60, mean(by[n], 'delta_soh_ppm'), st['marker'],
               color=st['color'], mec='black', mew=0.6, ms=10,
               label=st['label'], zorder=3)
    a.plot([], [], 'o-', color=STYLE[DA]['color'], mec='black', mew=0.6, label=r'DA-Lag-RLS, $w_D$ = 0 ... 16000')
    a.set_xlabel('Mission time (min)')
    a.set_ylabel(r'$\Delta$SOH per mission (ppm)')
    a.set_title(r'(a) Capacity loss vs. mission time')
    a.grid(alpha=0.3)
    a.legend(loc='upper right', frameon=True, fontsize=FS - 3, handletextpad=0.3)
    # (b) km to EOL, relative to BA-EPD
    names = ['AS-MF', 'QoS-EPD', 'BA-EPD+SOH', DA]
    base = km_to_eol(by['BA-EPD'])
    w = 0.8 / len(names)
    for j, n in enumerate(names):
        st = STYLE[n]
        y = 100 * (km_to_eol(by[n]) / base - 1)
        xs = np.arange(3) + (j - (len(names) - 1) / 2) * w
        b.bar(xs, y, w * 0.92, color=st['color'], ec=st['color'], lw=1.2, label=st['label'],
              zorder=2)
    b.axhline(0, color='#333333', lw=0.8)
    b.set_xticks(range(3))
    b.set_xticklabels(['Low', 'Medium', 'High'])
    b.set_xlabel('Complexity')
    b.set_ylabel('km to EOL vs. BA-EPD (%)')
    b.set_title('(b) Distance to end of life')
    b.grid(axis='y', alpha=0.3, zorder=0)
    b.set_ylim(None, 9.5)
    b.legend(loc='upper center', frameon=True, fontsize=FS - 3, ncol=2, columnspacing=0.8,
             handlelength=1.2, handletextpad=0.3)
    fig.tight_layout()
    save(fig, 'exp_longterm')


# ------------------------------------------------------------------------------------------
def fig_ablation(by):
    """Top: components removed from Lag-RLS (sorted). Bottom: additional / true information."""
    comp = [('Lag-RLS w/o CD', 'w/o change detection'), ('Lag-RLS w/o RLS', r'w/o RLS'),
            ('Lag-RLS w/o lambda', r'w/o $\lambda$ (BA-EPD modes)'),
            ('Lag-RLS w/o QoS cost', 'w/o expected QoS cost'),
            ('Lag-RLS tier select', r'tier selection of $\lambda$'),
            ('Lag-RLS w/o SOH est', 'w/o SOH estimation'), ('Lag-RLS w/o probe', 'w/o probing'),
            ('Lag-RLS w/o eta', r'w/o loss factor $\eta$')]
    other = [('Lag-RLS split power', 'separate power sensing'),
             ('Lag-RLS event count', 'measured event count'),
             ('Lag-RLS split+events', 'both extra sensors'),
             ('Lag-RLS (full info)', 'true SOC / SOH')]
    key = lambda r: (r['scenario'], r['seed_i'])
    ref = {key(r): r['return'] for r in by['Lag-RLS']}

    def vals(lst):
        out = []
        for n, lab in lst:
            if n not in by:
                continue
            d = [-(r['return'] - ref[key(r)]) for r in by[n]]      # increase in cost
            lo, hi = boot_ci(d)
            out.append((lab, np.mean(d), lo, hi))
        return out

    ref_da = {key(r): r['return'] for r in by[DA]}
    d0 = [-(r['return'] - ref_da[key(r)]) for r in by['Lag-RLS']]
    deg = [(r'w/o degradation cost ($w_D$=0)', np.mean(d0), *boot_ci(d0))]
    top = sorted(vals(comp), key=lambda v: -v[1])
    bot = vals(other)
    rows = deg + top + bot
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 0.9))
    y = -np.arange(len(rows), dtype=float)
    y[1:] -= 0.6                                              # gaps between the groups
    y[1 + len(top):] -= 0.6
    m = np.array([v[1] for v in rows])
    col = [STYLE[DA]['color']] + [STYLE['Lag-RLS']['color']] * len(top) + ['#9a9a9a'] * len(bot)
    ax.barh(y, m, 0.6, color=col, zorder=2)
    ax.errorbar(m, y, xerr=[m - [v[2] for v in rows], [v[3] for v in rows] - m], fmt='none',
                ecolor='#333333', lw=1, capsize=3, zorder=3)
    ax.axvline(0, color='#333333', lw=0.8)
    ax.axhline(y[0] - 0.8, color='#888888', lw=0.8, ls=':')
    ax.axhline(y[len(top)] - 0.8, color='#888888', lw=0.8, ls=':')
    ax.set_yticks(y)
    ax.set_yticklabels([v[0] for v in rows])
    ax.set_xlabel('Increase in cost')
    ax.grid(axis='x', alpha=0.3, zorder=0)
    save(fig, 'exp_ablation')


if __name__ == '__main__':
    by = load()
    fig_overall(by)
    fig_tradeoff(by)
    fig_aging(by, ncol=4)
    fig_aging(by, ABL_AGING, 'exp_ablation_soh', ncol=4)
    fig_longterm(by)
    fig_ablation(by)
