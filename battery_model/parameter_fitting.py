#!/usr/bin/env python3
"""Audited LiPo EIS / capacity fitting.

ECM: Fits SOC polynomials to R0, R1, R2, and approximated 2RC capacitances.
The LiPo dataset's Q1/Q2 are CONSTANT PHASE ELEMENT coefficients, not Farads.
This script uses the Hsu-Mansfeld parallel R-CPE equivalent capacitance:
  C_equiv = Q**(1/alpha) * R**((1-alpha)/alpha).
This maps a CPE to a local equivalent RC circuit, NOT a dynamically validated 2RC.

Aging: Fits only an EFFECTIVE constant-condition decay rate to capacity vs cycle,
  SOH(N) = exp(-k_eff * N),
using battery-disjoint train/validation splits. Each training cell gets equal
weight by default (cells, not check-up points, are the independent samples).
Xu cycle/SOC/T stress coefficients are NOT identifiable from this dataset: every
cell was cycled with one protocol (CC-CV to 4.2 V, 3 A to cutoff, room
temperature, no temperature log).

For battery_model.py, k_eff is translated into the single Xu scale k_cyc:
the Xu stress SHAPE (k_d1..k_d3, k_sigma, sigma_ref, k_T, k_t) is kept at the
literature values and k_cyc is chosen so that one stress cycle at the measured
protocol (median DoD from Capacity_stress/Capacity_std, median cycle duration
from the stress time series) gives fd = k_eff. This is a TRANSFER assumption,
not a calibration of the stress shape; it is recorded in 'degradation_provenance'.

Example:
 python parameter_fitting.py --train-datasets 1,2 --val-datasets 3 --plot

Relative paths (dataset root, outputs) resolve against this script's folder,
so the script works from any working directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
import numpy as np

from battery_model import (_DEFAULT_AGING_PARAMS_SOURCE, _xu_calendar_fd,
                           _xu_fd_halfcycle)

SCRIPT_DIR = Path(__file__).resolve().parent
CELL_NOMINAL_VOLTAGE = 3.7   # LiPo single cell
ASSUMED_TEST_TEMP_K = 298.15 # dataset has no temperature log


def resolve_path(p: str) -> Path:
    """Absolute paths unchanged; relative paths resolve against SCRIPT_DIR."""
    path = Path(p).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path


def parse_ids(s):
    return sorted(set(int(v.strip()) for v in s.split(',') if v.strip()))


def cpe_to_c_parallel(r: float, q: float, alpha: float) -> float:
    if not (r > 0.0 and q > 0.0 and 0.0 < alpha <= 1.0):
        raise ValueError(f'Invalid R||CPE combination: R={r}, Q={q}, alpha={alpha}')
    return math.exp((math.log(q) + (1.0 - alpha)*math.log(r))/alpha)


def ecm_rows(root, ids, skipped=None):
    rows = []
    for i in ids:
        files = sorted((root/f'LiPO_{i}_fit').glob('model_fit_cycle_*.csv'))
        if not files:
            raise FileNotFoundError(f'No LiPO_{i}_fit/model_fit_cycle_*.csv under {root}')
        for fn in files:
            cyc = int(re.search(r'cycle_(\d+)\.csv$', fn.name).group(1))
            with fn.open(newline='') as fp:
                for row in csv.DictReader(fp):
                    soc=float(row['SOC']); r0=float(row['RO'])
                    r1=float(row['R1']); r2=float(row['R2'])
                    q1=float(row['Q1']); a1=float(row['a1'])
                    q2=float(row['Q2']); a2=float(row['a2'])
                    if not 0 <= soc <= 1 or r0 <= 0:
                        if skipped is not None: skipped[i] = skipped.get(i, 0) + 1
                        continue
                    try:
                        c1=cpe_to_c_parallel(r1, q1, a1)
                        c2=cpe_to_c_parallel(r2, q2, a2)
                    except ValueError:
                        if skipped is not None: skipped[i] = skipped.get(i, 0) + 1
                        continue
                    rows.append(dict(dataset=i, cycle=cyc, SOC=soc, R0=r0,
                                     R1=r1, R2=r2, C1=c1, C2=c2,
                                     Q1=q1, Q2=q2, alpha1=a1, alpha2=a2))
    if not rows:
        raise ValueError('No valid ECM data')
    return rows


def capacity_traces(root, ids):
    traces={}
    for i in ids:
        fn=root/f'LiPO_{i}'/'Capacity'/'Capacity_std.csv'
        if not fn.exists():
            raise FileNotFoundError(fn)
        x=np.loadtxt(fn, dtype=float)
        if x.ndim != 2 or x.shape[1] < 2 or len(x) < 2:
            raise ValueError(f'Invalid capacity file: {fn}')
        x=x[np.argsort(x[:,0])]
        if np.any(np.diff(x[:,0]) <= 0) or np.any(x[:,1] <= 0):
            raise ValueError(f'Non-monotone cycles or non-positive capacity in {fn}')
        cycles=x[:,0]-x[0,0]
        soh=x[:,1]/x[0,1]
        traces[i]=(cycles,soh)
    return traces


def fit_effective_fade(traces, cell_weighting='equal'):
    """Least-squares k for fd = -ln(SOH) = k*N through the origin.

    cell_weighting='equal': every cell contributes equally (each cell's squared
    errors are averaged over its own points) so cells with more check-ups do
    not dominate. 'points': plain pooled least squares over all points.
    """
    if cell_weighting not in ('equal', 'points'):
        raise ValueError(f'Unknown cell_weighting: {cell_weighting!r}')
    numerator=denominator=0.0
    for n,soh in traces.values():
        fd=-np.log(np.maximum(soh, 1e-12))
        w = 1.0/len(n) if cell_weighting == 'equal' else 1.0
        numerator+=w*float(np.dot(n,fd))
        denominator+=w*float(np.dot(n,n))
    if denominator <= 0:
        raise ValueError('Cannot fit cycle fade without cycle variation')
    return max(numerator/denominator,0.0)


def _last_time_s(fn: Path) -> float:
    last = None
    with fn.open() as fh:
        for line in fh:
            if line.strip():
                last = line
    return float(last.split()[0]) if last else float('nan')


def stress_protocol(root, ids):
    """Measured stress-cycle DoD and duration for the given cells.

    DoD_n = stress-discharge Ah (Capacity_stress.csv row n) / standard capacity
    interpolated at cycle n (Capacity_std.csv). Duration_n = charge + discharge
    time from 2_n_stresscharge/2_n_stressdischarge.csv.
    """
    dods, durs = [], []
    for i in ids:
        cdir = root/f'LiPO_{i}'
        std = np.loadtxt(cdir/'Capacity'/'Capacity_std.csv', dtype=float)
        std = std[np.argsort(std[:,0])]
        stress = np.loadtxt(cdir/'Capacity'/'Capacity_stress.csv', dtype=float, ndmin=1)
        if stress.ndim > 1:
            stress = stress[:,0]
        n = np.arange(stress.size, dtype=float)
        dods.append(np.clip(stress/np.interp(n, std[:,0], std[:,1]), 1e-3, 1.0))
        eis = next(cdir.glob('EIS_Charge_*'), None)
        if eis is not None:
            for k in range(stress.size):
                fd, fc = eis/f'2_{k}_stressdischarge.csv', eis/f'2_{k}_stresscharge.csv'
                if fd.exists() and fc.exists():
                    durs.append(_last_time_s(fd) + _last_time_s(fc))
    dod = np.concatenate(dods)
    durs = np.array([d for d in durs if np.isfinite(d)])
    if not durs.size:
        raise FileNotFoundError('No stress charge/discharge time series found')
    return {'dod_median': float(np.median(dod)), 'dod_p10': float(np.percentile(dod, 10)),
            'dod_p90': float(np.percentile(dod, 90)),
            'cycle_duration_s_median': float(np.median(durs)), 'n_cycles': int(dod.size)}


def k_eff_to_xu_k_cyc(k_eff, protocol):
    """k_cyc such that one protocol stress cycle (discharge + charge half-cycle
    + calendar time) produces fd = k_eff under the literature Xu shape."""
    ap = dict(_DEFAULT_AGING_PARAMS_SOURCE)
    ap['k_cyc'] = 1.0
    dod = protocol['dod_median']
    avg_soc = 1.0 - dod/2.0
    cyc_unit = 2.0*_xu_fd_halfcycle(dod, avg_soc, ASSUMED_TEST_TEMP_K, **ap)
    cal = _xu_calendar_fd(1.0, 1.0 - dod, protocol['cycle_duration_s_median'],
                          ASSUMED_TEST_TEMP_K, **ap)
    if cyc_unit <= 0:
        raise ValueError('Xu cycle stress is zero at the protocol DoD')
    k_cyc = max((k_eff - cal)/cyc_unit, 0.0)
    ap['k_cyc'] = k_cyc
    return ap, {'cycle_fd_per_cycle_at_k_cyc_1': cyc_unit,
                'calendar_fd_per_cycle': cal,
                'calendar_share_of_k_eff': cal/k_eff if k_eff > 0 else None}


def metrics(pred, ref):
    diff=np.asarray(pred)-np.asarray(ref)
    return {'n':len(diff), 'mae':float(np.mean(np.abs(diff))),
            'rmse':float(np.sqrt(np.mean(diff**2)))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-root',default='LiPo_dataset')
    p.add_argument('--train-datasets',default='1,2')
    p.add_argument('--val-datasets',default='3')
    p.add_argument('--degree', type=int,default=3)
    p.add_argument('--json-out', default='lipo_fit_parameters.json',
                   help='Read by battery_model.py (default name)')
    p.add_argument('--metrics-out',default='lipo_validation_metrics.json')
    p.add_argument('--cell-weighting',choices=('equal','points'),default='equal',
                   help='equal: each training cell weighted equally (default); points: pooled over check-up points')
    p.add_argument('--plot',action='store_true')
    args=p.parse_args()
    root=resolve_path(args.dataset_root)
    train=parse_ids(args.train_datasets); val=parse_ids(args.val_datasets)
    if not train or not val or set(train)&set(val):
        raise ValueError('Train and validation datasets must be nonempty and disjoint')
    skipped={}
    tr=ecm_rows(root,train,skipped); va=ecm_rows(root,val,skipped)
    ecm={}; ecm_metrics={}
    for key in ('R0','R1','R2','C1','C2'):
        x=np.array([r['SOC'] for r in tr]); y=np.array([r[key] for r in tr])
        xv=np.array([r['SOC'] for r in va]); yv=np.array([r[key] for r in va])
        # Fit log(parameter) so resistance and equivalent capacitance stay positive.
        c=np.polyfit(x,np.log(y),args.degree)
        pv=np.exp(np.polyval(c,xv))
        ecm[key]={
            'degree':args.degree, 'coefficients':c.tolist(),
            'reference_soc_0_5':float(np.exp(np.polyval(c,.5))),
            'coefficients_are_for':'log(parameter); predict using exp(polyval(coefficients, SOC))',
            'units':'ohm' if key[0]=='R' else 'farad (local R-CPE equivalent)'
        }
        ecm_metrics[key]=metrics(pv,yv)
        ecm_metrics[key]['mape_pct']=float(np.mean(np.abs(pv-yv)/yv)*100)
        ecm_metrics[key]['non_positive_predictions']=int(np.sum(pv<=0))

    c_train=capacity_traces(root,train); c_val=capacity_traces(root,val)
    k=fit_effective_fade(c_train, args.cell_weighting)
    k_points=fit_effective_fade(c_train, 'points')
    individual={str(i):fit_effective_fade({i:t}) for i,t in c_train.items()}
    k_lo, k_hi = min(individual.values()), max(individual.values())
    val_soh_metrics={}
    for i,(n,s) in c_val.items():
        pred=np.exp(-k*n)
        m=metrics(pred,s)
        m['mae_percentage_points']=100*m['mae']
        m['rmse_percentage_points']=100*m['rmse']
        m['observed_final_soh']=float(s[-1])
        m['predicted_final_soh']=float(pred[-1])
        # Band from the slowest/fastest single training cell: cell-to-cell spread.
        m['predicted_final_soh_range_from_train_cells']=[float(np.exp(-k_hi*n[-1])), float(np.exp(-k_lo*n[-1]))]
        m['last_measured_cycle']=float(n[-1])
        sst=float(np.sum((s-np.mean(s))**2))
        m['r2']=float(1-np.sum((pred-s)**2)/sst) if sst>0 else None
        val_soh_metrics[str(i)]=m

    # Cell nominal capacity: median initial standard capacity of training cells.
    cell_capacity_ah=float(np.median([
        np.loadtxt(root/f'LiPO_{i}'/'Capacity'/'Capacity_std.csv', dtype=float)[0,1]
        for i in train]))
    protocol=stress_protocol(root, train)
    degradation, mapping=k_eff_to_xu_k_cyc(k, protocol)

    report={
        'train_dataset_ids':train,'val_dataset_ids':val,
        'cell':{'nominal_voltage_v':CELL_NOMINAL_VOLTAGE,'capacity_ah':cell_capacity_ah,
                'capacity_source':'median first Capacity_std value of training cells'},
        'ecm':ecm,
        'ecm_fit_method':'CPE Q/alpha transformed to effective C using Hsu-Mansfeld; pooled SOC-only log-polynomial cubic; not time-domain validated',
        'ecm_metrics':ecm_metrics,
        'ecm_rows_skipped_by_dataset':{str(i):c for i,c in skipped.items()},
        'effective_cycle_fade':{'model':'SOH=exp(-k_eff*N)',
                                'k_eff_per_cycle':k,
                                'cell_weighting':args.cell_weighting,
                                'k_eff_pooled_points':k_points,
                                'individual_train_cell_rates':individual,
                                'validation_soh':val_soh_metrics,
                                'warning':'Only an effective rate at the dataset protocols. Does not identify Xu SoC/DoD/T stress factors.'},
        'degradation':degradation,
        'degradation_provenance':{
            'fitted':'k_cyc (from k_eff only)',
            'fixed_literature':sorted(k_ for k_ in degradation if k_ != 'k_cyc'),
            'method':'k_cyc = (k_eff - calendar_fd_per_cycle) / (2 * Xu half-cycle fd at protocol DoD, k_cyc=1)',
            'protocol':protocol,
            'assumed_temperature_k':ASSUMED_TEST_TEMP_K,
            **mapping,
            'warning':'Stress shape is Xu (LMO) literature, transferred to LiPo; extrapolation to DoD far from the protocol median is unvalidated.'},
        'warnings':[
            'ECM is for source single cell, not a 4S pack; CPE-to-C is a local approximation.',
            'SOC-only fit pools multiple ageing stages and does not learn an independent SOH dependence.',
            'Do not mix dimensional ECM errors into one overall MAE.',
            'No rainflow charge/discharge stress histories or temperature variation are identified.',
            "The 'degradation' field is k_eff mapped through the literature Xu shape (see degradation_provenance), not a Xu calibration.",
            'EIS RC time constants are milliseconds (high-frequency arcs); they do not capture slow diffusion dynamics.'
        ]}
    json_out=resolve_path(args.json_out); metrics_out=resolve_path(args.metrics_out)
    json_out.write_text(json.dumps(report,indent=2))
    metrics_out.write_text(json.dumps({'ecm_metrics':ecm_metrics, 'soh_metrics':val_soh_metrics},indent=2))

    print('Training',train,'Validation',val)
    print('2RC EIS parameter validation (local CPE equivalents):')
    for key,m in ecm_metrics.items():
        print(f'  {key}: MAE={m["mae"]:.6g}, RMSE={m["rmse"]:.6g}, MAPE={m["mape_pct"]:.2f}%, predicted_nonpositive={m["non_positive_predictions"]}')
    if skipped:
        print('Skipped invalid ECM rows:', {f'LiPO_{i}':c for i,c in skipped.items()})
    print(f'Effective fade rate k_eff={k:.9g} /cycle ({args.cell_weighting} cell weighting; pooled-points={k_points:.9g})')
    print('Per training cell:', individual)
    print('Held-out SOH validation:')
    for i,m in val_soh_metrics.items():
        r2='n/a' if m['r2'] is None else f'{m["r2"]:.3f}'
        lo,hi=m['predicted_final_soh_range_from_train_cells']
        print(f'  LiPO_{i}: MAE={m["mae_percentage_points"]:.3f} pp, RMSE={m["rmse_percentage_points"]:.3f} pp, R2={r2}; '
              f'final SOH obs={m["observed_final_soh"]:.4f}, pred={m["predicted_final_soh"]:.4f} '
              f'(train-cell range {lo:.4f}..{hi:.4f})')
    print(f'Stress protocol (train): DoD median={protocol["dod_median"]:.3f} '
          f'(p10..p90 {protocol["dod_p10"]:.3f}..{protocol["dod_p90"]:.3f}), '
          f'cycle duration median={protocol["cycle_duration_s_median"]:.0f} s')
    print(f'battery_model mapping: k_cyc={degradation["k_cyc"]:.4g} '
          f'(calendar share of k_eff {mapping["calendar_share_of_k_eff"]:.2%}); '
          f'cell capacity {cell_capacity_ah:.4f} Ah')
    print('WARNING: this is not identification of the Xu stress coefficients.')
    print('Saved:', json_out, metrics_out)

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,5))
        for i,(n,soh) in c_val.items():
            ax.plot(n,soh,'o-',label=f'LiPO_{i}: measured SOH')
            ax.plot(n,np.exp(-k*n),'--',label=f'LiPO_{i}: train-derived prediction')
        ax.set(xlabel='Cycle index (relative to first record)',ylabel='Normalized SOH',title='Held-out LiPo SOH validation')
        ax.grid(alpha=.3);ax.legend();fig.tight_layout()
        fig_path=SCRIPT_DIR/'lipo_soh_validation_audited.png'
        fig.savefig(fig_path,dpi=170)
        print('Saved figure:', fig_path)
        plt.close(fig)

if __name__=='__main__':
    main()
