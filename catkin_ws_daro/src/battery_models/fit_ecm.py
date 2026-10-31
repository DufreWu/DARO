"""Fit a 2-RC ECM to time/current/voltage data.

Usage:
    python3 fit_ecm.py --file path/to/data.csv

The script expects files with either:
- two columns: x, voltage (fits OCV polynomial)
- three columns: time(s), current(A), voltage(V) (fits dynamic ECM)

If `scipy` is available, it uses `least_squares`; otherwise falls back to simple random search.
"""

import argparse
import math
from pathlib import Path
import numpy as np
import importlib.util
import sys

# load TwoRCECM from rc_ecm.py
def load_module_from_path(name, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

rc_mod = load_module_from_path("rc_ecm", Path(__file__).parent / "rc_ecm.py")
TwoRCECM = rc_mod.TwoRCECM


def try_import_scipy():
    try:
        from scipy.optimize import least_squares

        return least_squares
    except Exception:
        return None


def parse_data(path: Path):
    # try to read whitespace- or comma-separated
    data = np.loadtxt(str(path))
    if data.ndim == 1:
        data = data.reshape(-1, data.shape[0])
    return data


def fit_ocv(x, v):
    # fit polynomial degree 3
    coefs = np.polyfit(x, v, 3)
    return coefs


def simulate_ecm_params(params, t, I, soc0=1.0):
    R0, R1, C1, R2, C2 = params
    # enforce small positive values
    R0 = max(1e-12, R0)
    R1 = max(1e-12, R1)
    C1 = max(1e-6, C1)
    R2 = max(1e-12, R2)
    C2 = max(1e-6, C2)

    model = TwoRCECM(R0=R0, R1=R1, C1=C1, R2=R2, C2=C2)
    model.soc = soc0
    dt = np.diff(t)
    # if uniform, ok; otherwise per-step dt
    V = []
    for i, Ii in enumerate(I):
        dti = dt[i] if i < len(dt) else dt[-1]
        model.step(Ii, dti)
        V.append(model.voltage(Ii))
    return np.array(V)


def residuals_ecm(x, t, I, V_meas, soc0=1.0):
    V_sim = simulate_ecm_params(x, t, I, soc0)
    return V_sim - V_meas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--soc0", type=float, default=1.0)
    args = p.parse_args()

    path = Path(args.file)
    if not path.exists():
        print("File not found:", path)
        return

    data = parse_data(path)
    ncols = data.shape[1]
    print(f"Loaded {data.shape[0]} rows, {ncols} columns from {path}")

    if ncols == 2:
        x = data[:, 0]
        v = data[:, 1]
        coefs = fit_ocv(x, v)
        print("OCV polynomial coefs (deg3):", coefs)
        # compute fit error
        v_fit = np.polyval(coefs, x)
        rmse = math.sqrt(np.mean((v - v_fit) ** 2))
        print(f"OCV fit RMSE: {rmse:.6e} V")
        return

    if ncols >= 3:
        t = data[:, 0]
        I = data[:, 1]
        V = data[:, 2]

        least_squares = try_import_scipy()

        x0 = np.array([0.01, 0.02, 2000.0, 0.03, 5000.0])
        bounds = (np.array([1e-6, 1e-6, 1e-3, 1e-6, 1e-3]), np.array([1.0, 1.0, 1e6, 1.0, 1e8]))

        if least_squares is not None:
            print("Fitting with scipy.least_squares...")
            res = least_squares(lambda x: residuals_ecm(x, t, I, V, soc0=args.soc0), x0, bounds=bounds, xtol=1e-8)
            print("success:", res.success)
            print("message:", res.message)
            params = res.x
        else:
            print("Scipy not available: running simple random-search for 200 iterations")
            best = None
            best_err = float("inf")
            rng = np.random.default_rng(0)
            for i in range(200):
                cand = np.array([
                    10 ** rng.uniform(-4, -0),
                    10 ** rng.uniform(-4, -0),
                    10 ** rng.uniform(2, 6),
                    10 ** rng.uniform(-4, -0),
                    10 ** rng.uniform(2, 8),
                ])
                r = residuals_ecm(cand, t, I, V, soc0=args.soc0)
                err = np.sqrt(np.mean(r ** 2))
                if err < best_err:
                    best_err = err
                    best = cand
            params = best

        print("Fitted params: R0, R1, C1, R2, C2")
        print(params)
        # compute RMSE on dataset
        r = residuals_ecm(params, t, I, V, soc0=args.soc0)
        rmse = math.sqrt(np.mean(r ** 2))
        print(f"Voltage RMSE: {rmse:.6e} V")

        # simple validation: split in two
        split = len(t) // 2
        r_val = residuals_ecm(params, t[split:], I[split:], V[split:], soc0=args.soc0)
        rmse_val = math.sqrt(np.mean(r_val ** 2))
        print(f"Validation RMSE (second half): {rmse_val:.6e} V")


if __name__ == "__main__":
    main()
