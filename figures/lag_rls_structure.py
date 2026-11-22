"""Block diagram of DA-Lag-RLS (methods/lag_rls.py, methods/degradation_aware.py)
-> figures/pdf/lag_rls_structure.pdf.

Layout: rover (left); real-time loop every 1 s (estimator + decision-making, top right);
Lagrangian planner every 10 s (bottom right)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.size': 12, 'mathtext.fontset': 'cm', 'font.family': 'serif'})

EDGE = '#222222'
C_ROVER = '#eeeeee'

fig, ax = plt.subplots(figsize=(11.5, 5.2))
ax.set_xlim(0.2, 9.6)
ax.set_ylim(-0.1, 4.3)
ax.axis('off')
R = fig.canvas.get_renderer()


def rbox(x, y, w, h, fc='white', lw=1.2, r=0.1, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.0,rounding_size={r}',
                                fc=fc, ec=EDGE, lw=lw, zorder=z))


def sbox(x, y, w, h, text, fs=12, z=4):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc='white', ec=EDGE, lw=1.0, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, zorder=z + 1)


def title(x, y, text, italic=None, fs=13.5):
    """Bold title centred at x (top at y); optional bold-italic suffix, e.g. '(every 1 s)'."""
    parts = [(text, 'normal')] + ([(italic, 'italic')] if italic else [])
    objs = [ax.text(0, y, t, ha='left', va='top', fontsize=fs, weight='bold', style=s, zorder=6)
            for t, s in parts]
    inv = ax.transData.inverted()
    widths = []
    for o in objs:
        bb = o.get_window_extent(R)
        widths.append(inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0])
    x0 = x - sum(widths) / 2
    for o, w in zip(objs, widths):
        o.set_x(x0)
        x0 += w


def arrow(pts, label=None, lpos=None, ha='center', va='bottom', fs=13):
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=EDGE, lw=1.4, zorder=3, solid_capstyle='butt')
    ax.annotate('', xy=pts[-1], xytext=pts[-2], zorder=3,
                arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.4, mutation_scale=15,
                                shrinkA=0, shrinkB=0))
    if label:
        ax.text(*lpos, label, ha=ha, va=va, fontsize=fs, zorder=6)


# --- rover ------------------------------------------------------------------------------
rbox(0.3, 0.22, 2.6, 3.88, C_ROVER, r=0.15)
title(1.6, 4.02, 'Rover')
mods = [('Motor', r'speed $v$'),
        ('CPU', r'frequency $f$'),
        ('Battery', r'$V, I, T$, SOC$^{\mathrm{cc}}$'),
        ('Applications', r'throughput $q$, req. $q_{\mathrm{ref}}$')]
for k, (t, b) in enumerate(mods):
    y = 2.86 - k * 0.84
    ax.add_patch(plt.Rectangle((0.42, y), 2.36, 0.72, fc='white', ec=EDGE, lw=1.0, zorder=3))
    title(1.6, y + 0.65, t, fs=12.5)
    ax.text(1.6, y + 0.2, b, ha='center', va='center', fontsize=11.5, zorder=6)

# --- real-time loop (every 1 s) ------------------------------------------------------------
rbox(3.6, 2.05, 5.4, 2.05, r=0.12)
title(6.3, 4.02, 'Real-Time Loop ', '(every 1 s)')
rbox(3.8, 2.25, 2.45, 1.4, r=0.1, z=3)
title(5.025, 3.58, 'Estimator', fs=12.5)
sbox(3.92, 2.83, 1.08, 0.4, 'SOH / SOC', fs=11.5)
sbox(3.92, 2.36, 1.08, 0.4, 'Loss factor', fs=11.5)
sbox(5.06, 2.83, 1.08, 0.4, 'Power', fs=11.5)
sbox(5.06, 2.36, 1.08, 0.4, 'Complexity', fs=11.5)
rbox(6.9, 2.25, 1.9, 1.4, r=0.1, z=3)
title(7.85, 3.58, 'Decision-Making', fs=12.5)
sbox(7.02, 2.83, 1.66, 0.4, r'Speed cmd. $v^\ast$')
sbox(7.02, 2.36, 1.66, 0.4, r'DVFS cmd. $f^\ast$')

# --- Lagrangian planner (every 10 s) -------------------------------------------------------
rbox(3.6, 0.55, 5.0, 1.05, r=0.12)
title(6.1, 1.52, 'Lagr. Planner ', '(every 10 s)')
sbox(3.75, 0.7, 1.95, 0.42, 'Safe energy budget')
sbox(6.0, 0.7, 2.45, 0.42, 'Lagrangian optimization')
arrow([(5.7, 0.91), (6.0, 0.91)])

# --- signals -------------------------------------------------------------------------------
arrow([(2.9, 3.0), (3.6, 3.0)], 'obs.', (3.25, 3.05))
arrow([(6.25, 2.95), (6.9, 2.95)], r'$\hat P, \hat c$', (6.575, 3.0))
arrow([(4.8, 2.25), (4.8, 1.6)], r'$\widehat{\mathrm{SOC}}, \widehat{\mathrm{SOH}}, \hat\eta, \hat P, \hat c$',
      (4.9, 1.8), ha='left', va='center')
arrow([(7.85, 1.6), (7.85, 2.25)], r'$\lambda^\ast$', (7.95, 1.8), ha='left', va='center')
arrow([(8.8, 2.95), (9.35, 2.95), (9.35, 0.05), (1.6, 0.05), (1.6, 0.22)],
      r'$(v^\ast, f^\ast)$', (5.5, 0.1))

out = Path(__file__).resolve().parent / 'pdf' / 'lag_rls_structure.pdf'
fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
print(out)
