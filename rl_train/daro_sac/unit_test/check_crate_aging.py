"""Does a C-rate-dependent aging model make the lifetime-optimal speed depend on SOH/SOC?

Background (check_speed_lifetime.py): with the plain Xu model the fade per
cycle is driven by depth of discharge, i.e. by energy, so the lifetime-optimal
speed equals the energy-optimal speed (~2.9 m/s) for every SOH/SOC. Battery
awareness cannot help there.

Here the Xu cycle fade gets a C-rate stress factor (charge-throughput weighted):

    fd_discharge = Xu_halfcycle(DoD, avg SOC, mean T) * mean_Q[ S_I(C(t)) ] + calendar
    fd_charge    = Xu_halfcycle(DoD, avg SOC, T_amb)  * S_I(C_charge)       + calendar
    S_I(C)       = exp(k_I * (C - C_ref)),   C = |I| / (Q_n * SOH)   (actual capacity)

Default k_I = 370.3 / (R * 298.15) = 0.149 per C: the C-rate term in the
activation energy of Wang et al., J. Power Sources 196 (2011) 3942
(Q_loss ~ exp(-(31700 - 370.3 C) / (R T))). It was fitted on LFP cells, so for
this LiPo pack it is an assumption; --k-i sweeps it. The LiPo dataset cannot
identify k_I: every cell was cycled with one protocol (1.01 Ah cells, 3 A
discharge = ~3 C; battery_model/parameter_fitting.py). k_cyc was calibrated at
that protocol, so C_ref = 3 keeps S_I = 1 there (the consistent choice).
C_ref is NOT a harmless scale: S_I multiplies only the cycle fade, not the
calendar fade, so C_ref (with k_I) sets the cycle/calendar balance and can move
the optimal speed. Use --c-ref to test that sensitivity.

Temperature already enters through Xu's S_T (mean mission temperature).

For each (SOH, SOC0) the local lifetime criterion is the fade per km,
fd_cycle / distance (smaller = more km before EOL from this state); the
lifetime-optimal speed minimises it.

    python rl_train/daro_sac/unit_test/check_crate_aging.py
    python rl_train/daro_sac/unit_test/check_crate_aging.py --k-i 0.149 1 3 --capacity-ah 5 2 1
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl_train/daro_sac

import numpy as np

import evaluation  # noqa: F401  (sets sys.path)
from evaluation import EVAL_CONFIG, make_test_env, TestScenario
from evaluation.metrics import run_episode
from battery_model import _DEFAULT_AGING_PARAMS, _xu_fd_halfcycle, _xu_calendar_fd
from check_speed_lifetime import ConstantPolicy, qos_frequency

R_GAS = 8.314
K_I_WANG = 370.3 / (R_GAS * 298.15)      # 0.149 per C
PAIRS = [(1.0, 1.0), (0.9, 1.0), (0.8, 1.0), (1.0, 0.5), (0.8, 0.5)]


def s_rate(c_rate, k_i, c_ref):
    return np.exp(k_i * (np.asarray(c_rate, dtype=float) - c_ref))


def mission_fd_crate(env, charge_c_rate, k_i, c_ref):
    """Per-cycle fade (mission + recharge) with the C-rate factor; k_i = 0 gives plain Xu."""
    ap = _DEFAULT_AGING_PARAMS
    traj = env.trajectory
    soc0, soh = env.soc_init, env.soh_init
    soc_end = float(traj[-1]['soc'])
    dod = max(0.0, soc0 - soc_end)
    avg = 0.5 * (soc0 + soc_end)
    dts = np.array([s['dt_s'] for s in traj])
    cur = np.abs([s['current_a'] for s in traj])
    temps_k = np.array([s['temperature_c'] for s in traj]) + 273.15
    t_mean = float(np.sum(dts * temps_k) / max(np.sum(dts), 1e-9))
    q_ah = env.robot.Q_n / 3600.0 * soh
    w = cur * dts
    s_dis = float(np.sum(w * s_rate(cur / q_ah, k_i, c_ref)) / max(np.sum(w), 1e-12))
    fd_dis = _xu_fd_halfcycle(dod, avg, t_mean, **ap) * s_dis
    soc_prev = soc0
    for s, dt, tk in zip(traj, dts, temps_k):
        fd_dis += _xu_calendar_fd(soc_prev, float(s['soc']), float(dt), float(tk), **ap)
        soc_prev = float(s['soc'])
    t_amb_k = env.robot.T_amb + 273.15
    t_charge_s = dod * 3600.0 / charge_c_rate
    fd_chg = (_xu_fd_halfcycle(dod, avg, t_amb_k, **ap) * float(s_rate(charge_c_rate, k_i, c_ref))
              + _xu_calendar_fd(soc_end, soc0, t_charge_s, t_amb_k, **ap))
    c_mean = float(np.sum(w * cur / q_ah) / max(np.sum(w), 1e-12))
    return fd_dis + fd_chg, c_mean, s_dis


def scan(soh, soc0, cfg, speeds, k_list, c_ref, charge_c_rate, capacity_ah, complexity):
    """Rows per speed: energy, mean C-rate, fade per km for each k_I."""
    rows = []
    for v in speeds:
        env = make_test_env(cfg, TestScenario(soh, soc0, complexity))
        if capacity_ah is not None:
            env.robot.Q_n = capacity_ah * 3600.0
            env.reset()
        f, _ = qos_frequency(env, v, complexity)
        m = run_episode(env, ConstantPolicy(v, f), seed=1, charge_c_rate=charge_c_rate)
        if m['completed'] < 1.0:
            continue
        km = env.distance_m / 1000.0
        r = {'v': float(v), 'energy_kj': m['energy_kj']}
        for k in k_list:
            fd, c_mean, s_dis = mission_fd_crate(env, charge_c_rate, k, c_ref)
            r[k] = fd / km
            r['c_mean'] = c_mean
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', type=float, default=1000.0)
    ap.add_argument('--complexity', type=float, default=1.0)
    ap.add_argument('--k-i', type=float, nargs='+', default=[0.0, K_I_WANG, 1.0, 3.0],
                    help='C-rate stress coefficients [1/C]; 0 = plain Xu, 0.149 = Wang 2011')
    ap.add_argument('--c-ref', type=float, default=3.0,
                    help='C-rate with S_I = 1 (LiPo dataset protocol ~3 C, where k_cyc was fitted)')
    ap.add_argument('--capacity-ah', type=float, nargs='+', default=[5.0],
                    help='pack capacity [Ah]; smaller = higher C-rate for the same power')
    ap.add_argument('--charge-c-rate', type=float, default=EVAL_CONFIG['charge_c_rate'])
    ap.add_argument('--vmin', type=float, default=1.0)
    ap.add_argument('--vmax', type=float, default=5.0)
    ap.add_argument('--dv', type=float, default=0.05)
    args = ap.parse_args()

    cfg = dict(EVAL_CONFIG, mission_distance_m=args.distance, test_complexity_jitter=0.0)
    speeds = np.round(np.arange(args.vmin, args.vmax + 1e-9, args.dv), 3)
    ks = args.k_i
    print(f"{args.distance:.0f} m, C={args.complexity:g}, C_ref={args.c_ref:g}, "
          f"charge {args.charge_c_rate:g} C. Lifetime-optimal speed = argmin fade per km.")
    for cap in args.capacity_ah:
        print(f"\n=== pack {cap:g} Ah ===")
        print(f"{'SOH':>4} {'SOC0':>4} | {'v_E*':>5} {'C@v_E*':>6} | "
              + ' '.join(f"{'v*(k=' + format(k, '.3g') + ')':>11}" for k in ks)
              + ' | life gain vs v_E* [%]')
        for soh, soc0 in PAIRS:
            rows = scan(soh, soc0, cfg, speeds, ks, args.c_ref, args.charge_c_rate, cap,
                        args.complexity)
            if not rows:
                print(f"{soh:4.2f} {soc0:4.2f} | no completed mission")
                continue
            be = min(rows, key=lambda r: r['energy_kj'])
            best = {k: min(rows, key=lambda r: r[k]) for k in ks}
            gain = {k: 100.0 * (be[k] / best[k][k] - 1.0) for k in ks}   # km-to-EOL gain
            print(f"{soh:4.2f} {soc0:4.2f} | {be['v']:5.2f} {be['c_mean']:6.3f} | "
                  + ' '.join(f"{best[k]['v']:11.2f}" for k in ks)
                  + ' | ' + ' '.join(f"{gain[k]:5.2f}" for k in ks))
    print("\nv_E* = energy-optimal speed; v*(k) = speed with the least fade per km for C-rate "
          "coefficient k; life gain = how much longer (km to EOL, locally) v*(k) lasts than v_E*.")


if __name__ == '__main__':
    main()
