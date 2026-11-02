#!/usr/bin/env python3
"""Fit a 2RC ECM and Xu-style degradation parameters from the LiPo_dataset.

The script reads:
  - LiPo_dataset/LiPO_*_fit/model_fit_*.csv
  - LiPo_dataset/LiPO_*/Capacity/Capacity_std.csv

and produces:
  - SOC-dependent ECM fit coefficients for R0, R1, R2, C1, C2
  - a data-driven Xu degradation parameter estimate based on cycle capacity fade

The fit is approximate because the dataset contains static fitted ECM points and
capacity fade traces, but not the full EIS/HPPC time series needed for a full
parameter identification at every SOC/SOH/T condition.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # optional but preferred for constrained fitting
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None


DATASET_ROOT = Path(__file__).resolve().parent / 'LiPo_dataset'


def _dataset_id_from_fit_dir(path: Path) -> Optional[int]:
    """Infer the LiPo dataset id from a path like LiPO_1_fit."""
    name = path.name
    if not name.endswith('_fit'):
        return None
    stem = name[:-4]
    if not stem.startswith('LiPO_'):
        return None
    suffix = stem.split('_')[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _parse_dataset_ids(raw: str) -> List[int]:
    """Parse a comma-separated dataset selector like '1,2,3,4' or '5'."""
    values: List[int] = []
    for item in str(raw).split(','):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def _load_fit_catalog(dataset_ids: Optional[Sequence[int]] = None) -> List[Dict[str, float]]:
    """Load LiPo fit rows from selected dataset ids, or all datasets by default."""
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f'LiPo_dataset not found at {DATASET_ROOT}')

    selected = set(int(v) for v in (dataset_ids or []))
    rows: List[Dict[str, float]] = []
    for fit_dir in sorted(DATASET_ROOT.glob('LiPO_*_fit')):
        dataset_id = _dataset_id_from_fit_dir(fit_dir)
        if dataset_ids is not None and dataset_id not in selected:
            continue
        for csv_path in sorted(fit_dir.glob('model_fit_*.csv')):
            with csv_path.open(newline='') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        soc = float(row['SOC'])
                    except (TypeError, ValueError):
                        continue
                    if not 0.0 <= soc <= 1.0:
                        continue
                    rows.append({
                        'SOC': soc,
                        'R0': float(row['RO']),
                        'R1': float(row['R1']),
                        'R2': float(row['R2']),
                        'C1': float(row['Q1']),
                        'C2': float(row['Q2']),
                    })
    if not rows:
        if dataset_ids is not None:
            raise ValueError(f'No model_fit_*.csv rows found for datasets {list(dataset_ids)} under {DATASET_ROOT}')
        raise ValueError(f'No model_fit_*.csv rows found under {DATASET_ROOT}')
    return rows


def _fit_poly_coeffs(x: np.ndarray, y: np.ndarray, degree: int = 3) -> np.ndarray:
    """Fit a polynomial y ≈ p(x) and return the coefficients in descending order."""
    if len(x) < 2:
        raise ValueError('Not enough samples to fit a polynomial.')
    degree = min(int(degree), max(1, len(x) - 1))
    coeffs = np.polyfit(x, y, degree)
    return coeffs


def fit_ecm_parameters(dataset_ids: Optional[Sequence[int]] = None) -> Dict[str, Dict[str, float]]:
    """Fit SOC-dependent ECM parameter curves from a selected LiPo subset."""
    rows = _load_fit_catalog(dataset_ids)
    soc = np.array([r['SOC'] for r in rows], dtype=float)

    param_names = ['R0', 'R1', 'R2', 'C1', 'C2']
    fits: Dict[str, Dict[str, float]] = {}
    for key in param_names:
        values = np.array([r[key] for r in rows], dtype=float)
        coeffs = _fit_poly_coeffs(soc, values, degree=3)
        fits[key] = {
            'degree': 3,
            'coefficients': [float(v) for v in coeffs],
            'reference_soc_0_5': float(np.polyval(coeffs, 0.5)),
            'reference_soc_1_0': float(np.polyval(coeffs, 1.0)),
        }

    return fits


def evaluate_ecm_fit(
    train_dataset_ids: Optional[Sequence[int]] = None,
    val_dataset_ids: Optional[Sequence[int]] = None,
) -> Dict[str, float | Dict[str, float]]:
    """Evaluate the fitted polynomial model on a held-out validation set.

    Returns both overall MAE/MSE and per-parameter metrics.
    """
    if val_dataset_ids is None:
        val_dataset_ids = [5]
    if train_dataset_ids is None:
        train_dataset_ids = [1, 2, 3, 4]

    train_fit = fit_ecm_parameters(train_dataset_ids)
    val_rows = _load_fit_catalog(val_dataset_ids)

    param_names = ['R0', 'R1', 'R2', 'C1', 'C2']
    all_errors: List[float] = []
    by_param: Dict[str, Dict[str, List[float]]] = {
        name: {'mae': [], 'mse': []} for name in param_names
    }

    for row in val_rows:
        soc = float(row['SOC'])
        for name in param_names:
            coeffs = np.array(train_fit[name]['coefficients'], dtype=float)
            pred = float(np.polyval(coeffs, soc))
            actual = float(row[name])
            err = pred - actual
            all_errors.append(err)
            by_param[name]['mae'].append(abs(err))
            by_param[name]['mse'].append(err ** 2)

    metrics: Dict[str, float | Dict[str, float]] = {
        'mae': float(np.mean(np.abs(all_errors))) if all_errors else 0.0,
        'mse': float(np.mean(np.square(all_errors))) if all_errors else 0.0,
        'mae_by_param': {},
        'mse_by_param': {},
    }
    for name in param_names:
        mae = float(np.mean(by_param[name]['mae'])) if by_param[name]['mae'] else 0.0
        mse = float(np.mean(by_param[name]['mse'])) if by_param[name]['mse'] else 0.0
        metrics['mae_by_param'][name] = mae
        metrics['mse_by_param'][name] = mse

    return metrics


def _parse_capacity_curve(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a Capacity_std.csv file of the form: cycle capacity."""
    cycles: List[float] = []
    caps: List[float] = []
    for line in path.read_text(errors='ignore').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            cyc = float(parts[0])
            cap = float(parts[1])
        except ValueError:
            continue
        cycles.append(cyc)
        caps.append(cap)
    if not cycles:
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(cycles)
    return np.array(cycles, dtype=float)[order], np.array(caps, dtype=float)[order]


def _capacity_fade_data() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return all capacity traces in the dataset."""
    traces: List[Tuple[np.ndarray, np.ndarray]] = []
    if not DATASET_ROOT.exists():
        return traces
    for capacity_file in sorted(DATASET_ROOT.glob('LiPO_*/Capacity/Capacity_std.csv')):
        cyc, cap = _parse_capacity_curve(capacity_file)
        if cyc.size:
            traces.append((cyc, cap))
    return traces


def _xu_fd_model(params: Dict[str, float], cycle_count: np.ndarray) -> np.ndarray:
    """Evaluate the Xu et al. half-cycle degradation model for a constant cycle state."""
    k_sigma = params['k_sigma']
    k_T = params['k_T']
    k_d1 = params['k_d1']
    k_d2 = params['k_d2']
    k_d3 = params['k_d3']
    sigma_ref = params['sigma_ref']
    T_ref_k = params['T_ref_k']

    # This dataset does not expose full DOD/avgSOC/temperature traces for each cycle,
    # so the fit is performed using an effective representative cycle condition.
    avg_soc = 0.5
    dod = 0.6
    temp_k = 298.15
    s_sigma = np.exp(k_sigma * (dod - sigma_ref))
    s_delta = 1.0 / (k_d1 * (avg_soc ** k_d2) + k_d3)
    s_t = np.exp(k_T * (temp_k - T_ref_k) * T_ref_k / temp_k)
    per_cycle_fd = 0.5 * s_sigma * s_delta * s_t
    return cycle_count * per_cycle_fd


def fit_degradation_parameters() -> Dict[str, float]:
    """Fit effective Xu degradation coefficients from measured capacity fade."""
    traces = _capacity_fade_data()
    if not traces:
        raise FileNotFoundError(f'No Capacity_std.csv files found under {DATASET_ROOT}')

    fd_all: List[np.ndarray] = []
    cycle_all: List[np.ndarray] = []
    for cyc, cap in traces:
        if cyc.size == 0:
            continue
        initial = float(cap[0])
        soh = np.clip(cap / max(initial, 1e-9), 1e-9, 1.0)
        fd = -np.log(soh)
        fd_all.append(fd)
        cycle_all.append(cyc)

    if not fd_all:
        raise ValueError('No valid capacity data found for degradation fitting.')

    n_points = max(3, min(30, min(len(x) for x in fd_all)))
    cycles = np.array(sorted(np.unique(np.concatenate([c[:n_points] for c in cycle_all]))), dtype=float)
    fd_median = np.median(np.stack([fd[:n_points] for fd in fd_all]), axis=0)

    prior = {
        # DoD stress model
        'k_d1': 1.40e5,
        'k_d2': -5.01e-1,
        'k_d3': -1.23e5,

        # SoC stress model
        'k_sigma': 1.04,
        'sigma_ref': 0.50,

        # Temperature stress model
        'k_T': 6.93e-2,
        'T_ref_k': 298.15,  # 25°C in Kelvin
    }
    if minimize is None:
        return prior

    def objective(params_vec: np.ndarray) -> float:
        params = {
            'k_sigma': float(params_vec[0]),
            'k_T': float(params_vec[1]),
            'k_d1': float(params_vec[2]),
            'k_d2': float(params_vec[3]),
            'k_d3': float(params_vec[4]),
            'sigma_ref': float(params_vec[5]),
            'T_ref_k': 298.15,
        }
        pred = _xu_fd_model(params, cycles)
        return float(np.mean((pred - fd_median) ** 2))

    x0 = np.array([
        prior['k_sigma'],
        prior['k_T'],
        prior['k_d1'],
        prior['k_d2'],
        prior['k_d3'],
        prior['sigma_ref'],
    ], dtype=float)

    bounds = [
        (0.1, 5.0),
        (1e-4, 0.5),
        (1e3, 1e7),
        (-2.0, 0.0),
        (-1e7, -1e3),
        (0.05, 0.95),
    ]

    try:
        result = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')
        if result.success:
            opt = result.x
            return {
                'k_sigma': float(opt[0]),
                'k_T': float(opt[1]),
                'k_d1': float(opt[2]),
                'k_d2': float(opt[3]),
                'k_d3': float(opt[4]),
                'sigma_ref': float(opt[5]),
                'T_ref_k': 298.15,
            }
    except Exception:
        pass

    return prior


def main() -> None:
    parser = argparse.ArgumentParser(description='Fit 2RC ECM and degradation parameters from the LiPo_dataset.')
    parser.add_argument('--train-datasets', type=str, default='1,2,3,4', help='Comma-separated LiPo dataset ids used for fitting, e.g. 1,2,3,4')
    parser.add_argument('--val-datasets', type=str, default='5', help='Comma-separated LiPo dataset ids used for validation, e.g. 5')
    parser.add_argument('--json-out', type=str, default='lipo_fit_parameters.json', help='Save results as JSON')
    parser.add_argument('--metrics-out', type=str, default='lipo_validation_metrics.json', help='Save validation metrics as JSON')
    args = parser.parse_args()

    train_ids = _parse_dataset_ids(args.train_datasets)
    val_ids = _parse_dataset_ids(args.val_datasets)

    ecm = fit_ecm_parameters(train_ids)
    aging = fit_degradation_parameters()
    metrics = evaluate_ecm_fit(train_ids, val_ids)

    out = {
        'train_dataset_ids': train_ids,
        'val_dataset_ids': val_ids,
        'ecm': ecm,
        'degradation': aging,
        'validation_metrics': metrics,
    }

    out_path = Path(__file__).resolve().parent / args.json_out
    out_path.write_text(json.dumps(out, indent=2))

    metrics_path = Path(__file__).resolve().parent / args.metrics_out
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print('Train datasets:')
    print(f'  {train_ids}')
    print('Validation datasets:')
    print(f'  {val_ids}')
    print('\nFitted 2RC ECM parameters:')
    for name, info in ecm.items():
        print(f'  {name}: coeffs={info["coefficients"]}, ref@SOC=0.5={info["reference_soc_0_5"]:.6g}')

    print('\nFitted Xu-style degradation parameters:')
    for key, value in aging.items():
        print(f'  {key}: {value}')

    print('\nValidation metrics on held-out datasets:')
    print(f'  Overall MAE: {metrics["mae"]:.6g}')
    print(f'  Overall MSE: {metrics["mse"]:.6g}')
    for name in ('R0', 'R1', 'R2', 'C1', 'C2'):
        print(f'  {name}: MAE={metrics["mae_by_param"][name]:.6g}, MSE={metrics["mse_by_param"][name]:.6g}')

    print(f'\nSaved JSON: {out_path}')
    print(f'Saved metrics: {metrics_path}')


if __name__ == '__main__':
    main()
