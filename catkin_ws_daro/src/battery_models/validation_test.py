"""Validation script for SOC and SOH estimation accuracy.

This script:
- Simulates a battery using `TwoRCECM`.
- Estimates SOC using coulomb counting.
- Computes SOH estimate using `StreamingSoH` and compares it to a
  damage baseline computed from `data/rainflow_counts.csv`.

Metrics reported: RMSE and MAE for SOC; absolute and relative error for SOH.
"""

import csv
import math
from statistics import mean
from pathlib import Path
import sys

# ensure local package imports work when running from workspace root
sys.path.insert(0, str(Path(__file__).parent))

# import local modules by filepath to avoid PYTHONPATH issues when running
import importlib.util

def load_module_from_path(name, path: Path):
	spec = importlib.util.spec_from_file_location(name, str(path))
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod

rc_mod = load_module_from_path("rc_ecm", Path(__file__).parent / "rc_ecm.py")
soh_mod = load_module_from_path("soh_estimate", Path(__file__).parent / "soh_estimate.py")

TwoRCECM = rc_mod.TwoRCECM
StreamingSoH = soh_mod.StreamingSoH


def rmse(a, b):
	return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def mae(a, b):
	return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def load_rainflow_baseline(csv_path: Path, estimator: StreamingSoH):
	"""Compute a reference cumulative damage (f_d) from a rainflow CSV.

	CSV expected columns: range, mean, count
	"""
	f_d = 0.0
	T_ref = estimator.T_ref
	for row in csv.DictReader(open(csv_path)):
		r = float(row.get("range", row.get("DoD", 0.0)))
		m = float(row.get("mean", row.get("mSoC", 0.5)))
		cnt = float(row.get("count", 1.0))

		# build a Cycle-like object to reuse damage formula
		T_i = T_ref if estimator.T_ref is not None else 1e-6
		S_sigma = math.exp(estimator.k_sigma * (m - estimator.sigma_ref))
		S_T = math.exp(estimator.k_T * (T_i - estimator.T_ref) * (estimator.T_ref / T_i))
		DoD_safe = max(r, 1e-12)
		S_delta = 1.0 / (
			estimator.k_delta1 * (DoD_safe ** estimator.k_delta2) * (estimator.T_ref / T_i)
			+ estimator.k_delta3
		)

		f_d += cnt * 1.0 * S_delta * S_sigma * S_T

	return f_d


def main():
	# simulation settings
	dt = 1.0  # 1 second timestep
	steps = 3600  # 1 hour of 1 A pulses

	# create model and estimator
	model = TwoRCECM()
	estimator = StreamingSoH()

	# generate a more realistic current profile: random walk around a duty
	import random

	random.seed(0)
	current_profile = []
	base = 0.2
	for t in range(steps):
		# simulate drive-like load: short discharge bursts and occasional regen
		if t % 600 < 200:
			I = base + random.uniform(0.5, 1.5)
		else:
			I = base - random.uniform(0.0, 0.3)
		# add some high-frequency noise
		I += random.gauss(0.0, 0.02)
		current_profile.append(I)

	true_soc = []
	voltages = []

	# initial SOC to both
	model.soc = 1.0
	soc0 = model.soc

	for I in current_profile:
		model.step(I, dt)
		true_soc.append(model.soc)
		voltages.append(model.voltage(I))

	# SOC estimator: coulomb counting with measurement noise and drift
	Q = model.Q
	est_soc = []
	soc_est = soc0
	bias = 0.0
	for I in current_profile:
		# simulate current sensor bias / noise
		meas_I = I + random.gauss(0.0, 0.01) + bias
		soc_est -= (meas_I * dt) / Q
		# slowly accrue bias to emulate calibration drift
		bias += random.gauss(0.0, 1e-6)
		soc_est = max(0.0, min(1.0, soc_est))
		est_soc.append(soc_est)

	# compute SOC metrics
	soc_rmse = rmse(true_soc, est_soc)
	soc_mae = mae(true_soc, est_soc)

	print(f"SOC RMSE: {soc_rmse:.6f}")
	print(f"SOC MAE : {soc_mae:.6f}")

	# compute SOH: detect turning points (peaks and troughs) with hysteresis
	turning_points = []
	window = 5
	for i in range(window, len(true_soc) - window):
		segment = true_soc[i - window : i + window + 1]
		center = segment[window]
		if center == max(segment) or center == min(segment):
			turning_points.append((center, 25.0))

	# feed turning points
	est2 = StreamingSoH()
	for soc, T in turning_points:
		est2.add_sample(soc, T)
	soh_est = est2.finalize()

	# baseline damage from rainflow CSV
	csv_path = Path(__file__).parent / "data" / "rainflow_counts.csv"
	if csv_path.exists():
		f_d_ref = load_rainflow_baseline(csv_path, est2)
		soh_ref = math.exp(-f_d_ref)
		abs_err = abs(soh_ref - soh_est)
		rel_err = abs_err / max(1e-12, soh_ref)
		print(f"SOH estimate: {soh_est:.6f}")
		print(f"SOH reference: {soh_ref:.6f} (from {csv_path})")
		print(f"SOH abs error: {abs_err:.6e}")
		print(f"SOH rel error: {rel_err:.6%}")
	else:
		print("No rainflow CSV found; printing estimator SOH only.")
		print(f"SOH estimate: {soh_est:.6f}")


if __name__ == "__main__":
	main()

