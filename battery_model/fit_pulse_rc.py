"""Time-domain identification and validation of the slow polarisation RC branches.

The EIS-fitted 2RC (parameter_fitting.py) only covers 5 kHz .. 0.2 Hz, i.e. time
constants of milliseconds; slow (diffusion / concentration) polarisation with time
constants of tens to hundreds of seconds lies outside that band. This script
identifies it from the 1C partial-discharge pulses of the LiPo dataset and validates
it on a held-out cell. It does not change the simulator.

Per EIS cycle c of a cell (dataset layout LiPO_<cell>/EIS_Charge_discharge):
  0_c_1_stddischarge.csv  full 1C discharge      -> capacity Q_c, discharge curve
  0_c_2_stdcharge.csv     1C CC(-CV) recharge    -> charge curve
  1_c_k_pardisch.csv      k-th 360 s 1C pulse (~10 % SOC each, rest + EIS before it)
  pseudo-OCV(SOC) = mean of the 1C charge and discharge curves, per cycle
  SOH = Q_c / Q_(first cycle)

Model (cell level), per pulse with the OCV drift taken from the pseudo-OCV:
  V(t) = OCV(SOC(t)) - I R_i - I R_1 (1 - e^{-t/tau_1}) - I R_2 (1 - e^{-t/tau_2})
  R_i = resistance seen at the 1 s sampling (ohmic + the ms EIS branches + offset of
        the pseudo-OCV).
Steps
  1. tau_1, tau_2: one global pair, grid search on the training pulses (R's by NNLS).
  2. R_i, R_1, R_2 per pulse by NNLS (linear for fixed taus).
  3. pooled (train cells), robust: log R = poly2(SOC) + log(1 + k (1 - SOH)).
  4. validation (held-out cell): simulate every pulse from its pseudo-OCV with
       EIS 2RC as in the simulator (pooled R0, R1, R2; R0 x (1 + 7.5 (1 - SOH)))
       time-domain 2RC (steps 1-3)
     voltage RMSE: absolute, and 'shape' (initial offset removed).

    python battery_model/fit_pulse_rc.py   -> battery_model/pulse_rc_fit.json
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, nnls

HERE = Path(__file__).resolve().parent
DATA = HERE / 'LiPo_dataset'
EIS_FIT = HERE / 'lipo_fit_parameters.json'
OUT = HERE / 'pulse_rc_fit.json'
TRAIN, VAL = (1, 2), (3,)
SOC_MIN = 0.15                 # drop pulses that start below this (cut-off knee)
TAU1_GRID = np.array([3, 5, 7, 10, 12, 15, 20, 25, 30, 40], float)
TAU2_GRID = np.array([50, 60, 75, 90, 110, 130, 160, 200, 250, 300, 400, 600], float)
K_R_SIM = 7.5                  # simulator's R0 ageing factor (ECMParams.resistance_aging_factor)
PACK_R_SCALE = 0.8082          # Ns / Np of the 4S 5 Ah pack (ECMParams.r_scale)
warnings.filterwarnings('ignore')


def load(path):
    return np.loadtxt(path).T          # t [s], I [A] (discharge < 0), V [V]


def ah(t, i):
    return np.concatenate([[0.0], np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t))]) / 3600.0


def cycles(cell):
    d = DATA / f'LiPO_{cell}' / 'EIS_Charge_discharge'
    return sorted({int(m.group(1)) for f in d.glob('1_*_*_pardisch.csv')
                   if (m := re.match(r'1_(\d+)_\d+_pardisch', f.name))})


def cycle_data(cell, c):
    """Capacity, pseudo-OCV(SOC) and the pulses of one EIS cycle."""
    d = DATA / f'LiPO_{cell}' / 'EIS_Charge_discharge'
    td, idis, vd = load(d / f'0_{c}_1_stddischarge.csv')
    q = -ah(td, idis)[-1]
    soc_d = 1.0 - (-ah(td, idis)) / q
    tc, ic, vc = load(d / f'0_{c}_2_stdcharge.csv')
    cc = ic > 0.95 * np.median(ic[ic > 0.5])            # constant-current part
    soc_c = (ah(tc, ic) / q)[cc]
    grid = np.linspace(0.0, 1.0, 201)
    v_dis = np.interp(grid, soc_d[::-1], vd[::-1])
    ok = (grid >= soc_c.min()) & (grid <= soc_c.max())
    half_gap = 0.5 * (np.interp(grid, soc_c, vc[cc]) - v_dis)
    gap = np.where(ok, half_gap, np.interp(grid, grid[ok], half_gap[ok]))   # hold the edge gap
    cyc = dict(q=q, grid=grid, ocv=v_dis + gap, pulses=[])
    soc0 = 1.0
    for k in range(20):
        f = d / f'1_{c}_{k}_pardisch.csv'
        if not f.exists():
            break
        t, i, v = load(f)
        t = t - t[0]
        cyc['pulses'].append(dict(k=k, t=t, i=-i, v=v, soc0=soc0))
        soc0 -= (-ah(t, i)[-1]) / q
    return cyc


def collect(cells):
    rows = []
    for cell in cells:
        q0 = None
        for c in cycles(cell):
            try:
                cyc = cycle_data(cell, c)
            except (OSError, ValueError):
                continue
            q0 = q0 or cyc['q']
            for p in cyc['pulses']:
                if p['soc0'] < SOC_MIN:
                    continue
                soc = p['soc0'] - ah(p['t'], p['i']) / cyc['q']
                rows.append(dict(cell=cell, cycle=c, soh=cyc['q'] / q0, soc=float(np.mean(soc)),
                                 p=p, cyc=cyc, ocv=np.interp(soc, cyc['grid'], cyc['ocv'])))
    return rows


def regressors(p, tau1, tau2):
    t, i = p['t'], p['i']
    return np.column_stack([i, i * (1 - np.exp(-t / tau1)), i * (1 - np.exp(-t / tau2))])


def fit_rs(r, tau1, tau2):
    """(R_i, R_1, R_2) >= 0 by NNLS for fixed taus; returns (R, sse)."""
    a = regressors(r['p'], tau1, tau2)
    y = r['ocv'] - r['p']['v']
    x, _ = nnls(a, y)
    return x, float(np.sum((a @ x - y) ** 2))


def fit_soc_soh(soc, soh, y):
    """Robust: log y = poly2(SOC) + log(1 + k (1 - SOH)); returns (coeffs, k, log-rmse)."""
    def res(x):
        return np.polyval(x[:3], soc) + np.log1p(x[3] * (1.0 - soh)) - np.log(y)
    sol = least_squares(res, x0=[0, 0, float(np.log(np.median(y))), 1.0], loss='soft_l1', f_scale=0.1,
                        bounds=([-np.inf] * 3 + [-0.9], [np.inf] * 3 + [20.0]))
    return sol.x[:3].tolist(), float(sol.x[3]), float(np.median(np.abs(sol.fun)))


def simulate(p, cyc, soh, r_inst, branches=()):
    """Terminal voltage of one pulse from the pseudo-OCV, an instantaneous resistance and RC branches."""
    t, I = p['t'], p['i']
    soc = p['soc0'] - ah(t, I) / cyc['q']
    v = np.interp(soc, cyc['grid'], cyc['ocv']) - I * r_inst(soc, soh)
    for rfun, tau in branches:
        r = rfun(soc, soh)
        vp = np.zeros_like(t)
        for n in range(1, len(t)):           # exact RC update with the sampled current
            a = math.exp(-(t[n] - t[n - 1]) / tau)
            vp[n] = a * vp[n - 1] + (1.0 - a) * r[n] * I[n]
        v = v - vp
    return v


def main():
    eis = json.loads(EIS_FIT.read_text())['ecm']
    pe = {k: eis[k]['coefficients'] for k in ('R0', 'R1', 'R2')}

    def r_eis_sim(soc, soh):          # the simulator's EIS 2RC (ms branches = instantaneous at 1 s)
        r0 = np.exp(np.polyval(pe['R0'], soc)) * (1.0 + K_R_SIM * (1.0 - soh))
        return r0 + np.exp(np.polyval(pe['R1'], soc)) + np.exp(np.polyval(pe['R2'], soc))

    rows = collect(TRAIN + VAL)
    tr = [r for r in rows if r['cell'] in TRAIN]
    va = [r for r in rows if r['cell'] in VAL]
    print(f'pulses: {len(tr)} train (cells {TRAIN}), {len(va)} validation (cells {VAL}), SOC >= {SOC_MIN}')

    # 1. global taus
    sse = {(t1, t2): sum(fit_rs(r, t1, t2)[1] for r in tr)
           for t1 in TAU1_GRID for t2 in TAU2_GRID if t2 > 2 * t1}
    tau1, tau2 = min(sse, key=sse.get)
    n_pts = sum(len(r['p']['t']) for r in tr)
    print(f'global taus: tau1 = {tau1:g} s, tau2 = {tau2:g} s (train rmse '
          f'{1e3 * math.sqrt(sse[tau1, tau2] / n_pts):.2f} mV; 1RC best '
          f"{1e3 * math.sqrt(min(sum(fit_rs(r, t, 1e9)[1] for r in tr) for t in TAU1_GRID) / n_pts):.2f} mV)")

    # 2. per-pulse resistances
    R = np.array([fit_rs(r, tau1, tau2)[0] for r in tr])
    soc = np.array([r['soc'] for r in tr])
    soh = np.array([r['soh'] for r in tr])
    for j, n in enumerate(('R_i', 'R_1', 'R_2')):
        print(f'  {n}: median {1e3 * np.median(R[:, j]):.1f} mOhm [p10 {1e3 * np.percentile(R[:, j], 10):.1f}, '
              f'p90 {1e3 * np.percentile(R[:, j], 90):.1f}]')

    # 3. pooled SOC / SOH model
    model = {'tau1': tau1, 'tau2': tau2}
    for j, n in enumerate(('R_i', 'R_1', 'R_2')):
        y = np.maximum(R[:, j], 1e-4)
        c, k, e = fit_soc_soh(soc, soh, y)
        model[n] = dict(coefficients=c, k_soh=k, median_abs_log_err=e)
    print('SOH factors (R x (1 + k (1 - SOH))): ' + ', '.join(f"k({n}) = {model[n]['k_soh']:.2f}"
                                                            for n in ('R_i', 'R_1', 'R_2'))
          + f'   train SOH {soh.min():.3f}-{soh.max():.3f}')

    def ev(n, s, h=1.0):
        return np.exp(np.polyval(model[n]['coefficients'], s)) * (1.0 + model[n]['k_soh'] * (1.0 - h))

    print(f"{'SOC':>5} | {'cell: R_i':>9} {'R_1':>6} {'C_1':>6} {'R_2':>6} {'C_2':>6} | "
          f"{'pack: R_i':>9} {'R_1':>6} {'C_1':>6} {'R_2':>6} {'C_2':>6} | EIS-sim R")
    for s in (1.0, 0.75, 0.5, 0.25):
        ri, r1, r2 = (ev(n, s) for n in ('R_i', 'R_1', 'R_2'))
        k = PACK_R_SCALE
        print(f'{s:5.2f} | {1e3 * ri:7.1f}mO {1e3 * r1:5.1f}m {tau1 / r1:5.0f}F {1e3 * r2:5.1f}m {tau2 / r2:5.0f}F | '
              f'{1e3 * ri * k:7.1f}mO {1e3 * r1 * k:5.1f}m {tau1 / (r1 * k):5.0f}F {1e3 * r2 * k:5.1f}m '
              f'{tau2 / (r2 * k):5.0f}F | {1e3 * r_eis_sim(s, 1.0):.1f} mO')

    # 4. validation on the held-out cell
    td = (lambda s, h: ev('R_i', s, h), [(lambda s, h: ev('R_1', s, h), tau1), (lambda s, h: ev('R_2', s, h), tau2)])
    res = {'eis': [], 'td': []}
    for r in va:
        for name, (ri, br) in {'eis': (r_eis_sim, ()), 'td': td}.items():
            e = simulate(r['p'], r['cyc'], r['soh'], ri, br) - r['p']['v']
            res[name].append(dict(soh=r['soh'], abs=1e3 * float(np.sqrt(np.mean(e ** 2))),
                                  shape=1e3 * float(np.sqrt(np.mean((e - e[0]) ** 2))),
                                  drift=1e3 * float(e[-1] - e[0])))
    print(f'\nvalidation on cell {VAL} ({len(va)} pulses, SOH {min(r["soh"] for r in va):.3f}-'
          f'{max(r["soh"] for r in va):.3f}): voltage error [mV], median')
    print(f"{'model':<16}{'RMSE abs':>10}{'RMSE shape':>12}{'end-start':>11}")
    out = {}
    for name, lab in (('eis', 'EIS 2RC (sim)'), ('td', 'time-domain 2RC')):
        x = res[name]
        ab, sh, dr = (np.array([d[k] for d in x]) for k in ('abs', 'shape', 'drift'))
        print(f'{lab:<16}{np.median(ab):10.1f}{np.median(sh):12.1f}{np.median(dr):11.1f}')
        out[name] = dict(rmse_abs_median_mv=float(np.median(ab)), rmse_shape_median_mv=float(np.median(sh)),
                         end_minus_start_median_mv=float(np.median(dr)))
    OUT.write_text(json.dumps(dict(
        train_cells=TRAIN, val_cells=VAL, n_train_pulses=len(tr), n_val_pulses=len(va),
        units='cell level; R [ohm] = exp(poly2(SOC)) * (1 + k_soh (1 - SOH)); tau [s]; '
              f'pack: R x {PACK_R_SCALE}, C / {PACK_R_SCALE}',
        model=model, validation=out), indent=1))
    print(f'\n-> {OUT.relative_to(HERE.parent)}')


if __name__ == '__main__':
    main()
