"""Fit StreamingSoH damage parameters to observed SOH data.

Usage examples:
  python3 fit_soh.py --rainflow data/rainflow_counts.csv --soh 0.95
  python3 fit_soh.py --rainflow data/rainflow_counts.csv --capacity-file ecm_data/LiPO_1/Capacity/Capacity_stress.csv

The rainflow CSV must have columns: range, mean, count
The script fits k_sigma, k_T, k_delta1, k_delta2, k_delta3 to match observed SOH.
"""

import argparse
import math
from pathlib import Path
import csv
import importlib.util
import sys

# load StreamingSoH

def load_module_from_path(name, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

soh_mod = load_module_from_path("soh_estimate", Path(__file__).parent / "soh_estimate.py")
StreamingSoH = soh_mod.StreamingSoH


def read_rainflow(csv_path: Path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({'range': float(r.get('range', r.get('DoD', 0.0))),
                         'mean': float(r.get('mean', r.get('mSoC', 0.5))),
                         'count': float(r.get('count', 1.0)),
                         'temp': float(r.get('temp', 25.0)) if 'temp' in r else 25.0})
    return rows


def f_d_from_params(rows, params, T_ref=25.0):
    # params: k_sigma, k_T, k_delta1, k_delta2, k_delta3, sigma_ref
    k_sigma, k_T, k_delta1, k_delta2, k_delta3, sigma_ref = params
    f_d = 0.0
    for row in rows:
        r = max(row['range'], 1e-12)
        m = row['mean']
        cnt = row['count']
        T_i = row.get('temp', T_ref) or T_ref
        S_sigma = math.exp(k_sigma * (m - sigma_ref))
        S_T = math.exp(k_T * (T_i - T_ref) * (T_ref / T_i))
        S_delta = 1.0 / (k_delta1 * (r ** k_delta2) * (T_ref / T_i) + k_delta3)
        f_d += cnt * 1.0 * S_delta * S_sigma * S_T
    return f_d


def try_import_scipy():
    try:
        from scipy.optimize import least_squares

        return least_squares
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rainflow', required=True)
    p.add_argument('--soh', type=float, help='Observed SOH (0..1)')
    p.add_argument('--capacity-file', type=str, help='File with capacity values per cycle (last value used as SOH)')
    p.add_argument('--sigma-ref', type=float, default=0.5)
    p.add_argument('--max-iters', type=int, default=200)
    args = p.parse_args()

    rf_path = Path(args.rainflow)
    if not rf_path.exists():
        print('Rainflow CSV not found:', rf_path)
        return

    rows = read_rainflow(rf_path)

    if args.soh is None and args.capacity_file is None:
        print('Provide either --soh or --capacity-file')
        return

    observed_soh = None
    if args.soh is not None:
        observed_soh = float(args.soh)
    else:
        cap_path = Path(args.capacity_file)
        if not cap_path.exists():
            print('Capacity file not found:', cap_path)
            return
        # read last numeric value
        vals = [float(x.strip()) for x in open(cap_path) if x.strip()]
        observed_soh = vals[-1]

    print('Observed SOH:', observed_soh)

    # initial guess from current code defaults
    init = [1.04, 6.93e-2, 1.40e5, -5.01e-1, -1.23e5, args.sigma_ref]

    def obj(x):
        f_d = f_d_from_params(rows, x)
        pred_soh = math.exp(-f_d)
        return pred_soh - observed_soh

    least_squares = try_import_scipy()
    if least_squares is not None:
        print('Fitting with scipy.least_squares...')
        def resid(x):
            return [obj(x)]
        lb = [-5.0, -5.0, 1.0, -5.0, -1e9, 0.0]
        ub = [5.0, 5.0, 1e12, 5.0, 1e12, 1.0]
        res = least_squares(lambda x: resid(x), init, bounds=(lb, ub), xtol=1e-12)
        print('success:', res.success)
        print('message:', res.message)
        x = res.x
    else:
        print('Scipy not available: running simple random-search')
        import random
        rng = random.Random(0)
        best = None
        best_err = float('inf')
        max_iters = args.max_iters
        for i in range(max_iters):
            cand = [
                rng.uniform(0.1, 2.0),
                rng.uniform(1e-3, 1.0),
                10 ** rng.uniform(3, 8),
                rng.uniform(-2.0, 0.5),
                10 ** rng.uniform(2, 8),
                rng.uniform(0.0, 1.0),
            ]
            pred = math.exp(-f_d_from_params(rows, cand))
            err = abs(pred - observed_soh)
            if err < best_err:
                best_err = err
                best = cand
        x = best

        # Local hill-climb refinement around best candidate
        if x is not None:
            step_scale = [0.2, 0.1, 0.5, 0.1, 0.5, 0.05]
            for scale in [1.0, 0.5, 0.2, 0.1]:
                improved = True
                iter_local = 0
                while improved and iter_local < 2000:
                    improved = False
                    iter_local += 1
                    for j in range(len(x)):
                        cand = list(x)
                        # multiplicative or additive perturbation depending on magnitude
                        if j in (2, 4):
                            # log-scale params (C1, C2)
                            factor = 10 ** (rng.uniform(-0.1, 0.1) * scale)
                            cand[j] = max(1e-6, cand[j] * factor)
                        else:
                            cand[j] = cand[j] + rng.uniform(-step_scale[j], step_scale[j]) * scale
                            # clamp reasonable ranges
                            if j == 5:
                                cand[j] = max(0.0, min(1.0, cand[j]))

                        pred = math.exp(-f_d_from_params(rows, cand))
                        err = abs(pred - observed_soh)
                        if err + 1e-12 < best_err:
                            best_err = err
                            x = cand
                            improved = True
            # end refinement

    print('Fitted params: k_sigma, k_T, k_delta1, k_delta2, k_delta3, sigma_ref')
    print(x)
    f_d = f_d_from_params(rows, x)
    print('Predicted SOH:', math.exp(-f_d))
    print('Observed SOH:', observed_soh)


if __name__ == '__main__':
    main()
