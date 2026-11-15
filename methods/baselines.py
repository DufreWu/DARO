"""Comparison methods.

Rule-based baselines (this module; methods.make_baselines builds all of them):
  AS-MF    Average Speed, Maximum CPU Frequency. No battery or QoS awareness.
  QoS-EPD  QoS-aware Energy-Per-Distance optimiser (non-RL): each step picks
           the speed that minimises electrical energy per metre
               min_v  (P_motor(v) + P_nonmotor(v, f_req(v), c)) / v   (robot_env.non_motor_power)
               s.t.   throughput(v, f_req(v), c_safe) >= margin * QoS_ref,
                      f_req(v) <= f_max
           and runs the CPU at f_req(v), the minimum frequency meeting QoS.
           Workload complexity is not observed: it is estimated on-line from
           the observed throughput (EMA) and inflated by the worst-case
           disturbance (c_safe). Battery-state agnostic (SOC/SOH unused).
  BA-EPD   Battery-aware QoS-EPD (strong rule-based baseline for the claim
           "finish fast subject to charge and QoS"). Same complexity estimate
           and QoS frequency as QoS-EPD, plus a charge-budget check from the
           OBSERVED SOC, SOH and distance remaining (same information as DARO;
           nominal OCV / capacity model, nominal power model):
             fast     v = min(v_fast, fastest QoS-feasible speed), if the energy
                      for the rest of the mission at that speed fits above the
                      SOC reserve (with a safety factor);
             optimal  QoS-EPD action, if that fits above the reserve, or (with a
                      smaller safety factor) above the SOC cutoff, i.e. the reserve
                      is used before QoS is given up;
             budget   highest QoS ratio whose energy fits above the cutoff.
           Priority completion > QoS > reserve, as in the reward.
           With battery_obs='bms' the observed SOC is the BMS estimate and SOH reads
           1.0, i.e. BA-EPD then assumes a fresh battery.
  BA-EPD+SOH  BA-EPD with an on-line SOH estimate for battery_obs='bms': for each
           candidate SOH h (grid), SOC_true(h) = SOC0 - (SOC0 - SOC_bms) / h and the
           predicted terminal voltage OCV(SOC_true(h), h) - I (R0(h) + R1 + R2) (nominal
           ECM, steady RC); h = argmin of the EMA of squared voltage residuals over
           near-steady steps. BA-EPD then plans with (SOC_true(h), h).
Proposed method: methods/lag_rls.py (Lag-RLS); SOH estimator: methods/soh_estimator.py.

RL ablations (SAC, same reward / network / training budget as DARO; only the
battery observations differ, see obs_variants.py):
  RL-SOC-only  python sac_training.py --obs soc  -> rl_train/daro_sac/checkpoints/rl_soc/seed_k
  RL-SOH-only  python sac_training.py --obs soh  -> rl_train/daro_sac/checkpoints/rl_soh/seed_k
Load them like DARO (runner.load_daro / load_rl_policies); the variant is read
from the checkpoint directory.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from envs.robot_env import (RobotParams, motor_power, non_motor_power, compute_throughput,
                            throughput_freq_required, throughput_complexity, throughput_max_speed, _voc)

from .soh_estimator import SohEstimator


class ASMFBaseline:
    name = 'AS-MF'

    def __init__(self, robot: RobotParams, config: dict):
        self.action = np.array([config['baseline_v_nom'], robot.f_max], dtype=np.float32)

    def reset(self):
        pass

    def select_action(self, obs, deterministic=True):
        return self.action.copy()


class QoSAwareEPD:
    """QoS-aware energy-per-distance minimiser (see module docstring)."""

    name = 'QoS-EPD'

    def __init__(self, robot: RobotParams, config: dict, n_grid: int = 161):
        p = self.p = robot
        self.margin = float(config['baseline_qos_margin'])
        self.ema = float(config['baseline_c_ema'])
        self.jitter = float(config['test_complexity_jitter'])
        self.n = robot.n_apps
        self.v_grid = np.linspace(max(p.v_min, 0.05), p.v_max, n_grid)
        # Steady-speed motor + aux power on the grid; compute depends on c.
        self._p_motor = motor_power(self.v_grid, 0.0, p)
        self.reset()

    def reset(self):
        self.c_hat: Optional[float] = None

    # --- workload complexity estimate -----------------------------------
    def _update_complexity(self, v, f, thr):
        # Invert robot_env.compute_throughput for c (identifiable only for v > 0).
        c = throughput_complexity(v, f, thr) if v > 0.1 else None
        if c is not None:
            c = float(np.clip(c, 0.1, 10.0))
            self.c_hat = c if self.c_hat is None else (1 - self.ema) * self.c_hat + self.ema * c

    def _c_safe(self) -> float:
        return (1.0 if self.c_hat is None else self.c_hat) * (1.0 + self.jitter)

    # --- optimisation ---------------------------------------------------
    def optimise(self, q_ref: float, c: float):
        """(v*, f*) minimising energy per metre subject to QoS."""
        p, q = self.p, q_ref * self.margin
        f_req = throughput_freq_required(self.v_grid, q, c)
        feasible = f_req <= p.f_max + 1e-9
        f = np.clip(f_req, p.f_min, p.f_max)
        e_per_m = (self._p_motor + non_motor_power(self.v_grid, f, c, p)) / self.v_grid
        if feasible.any():
            k = int(np.argmin(np.where(feasible, e_per_m, np.inf)))
            return float(self.v_grid[k]), float(f[k])
        # QoS unreachable at any speed: best effort (f_max), energy-optimal speed.
        k = int(np.argmin(e_per_m))
        return float(self.v_grid[k]), float(p.f_max)

    def select_action(self, obs, deterministic=True):
        v, f = float(obs[0]), float(obs[1])
        thr = np.asarray(obs[8:8 + self.n], dtype=float)
        q_ref = float(np.max(obs[8 + self.n:8 + 2 * self.n]))
        self._update_complexity(v, f, float(np.min(thr)))
        v_cmd, f_cmd = self.optimise(q_ref, self._c_safe())
        return np.array([v_cmd, f_cmd], dtype=np.float32)


class BatteryAwareEPD(QoSAwareEPD):
    """QoS-EPD + charge-budget check (see module docstring)."""

    name = 'BA-EPD'
    F_GRID = np.linspace(0.5, 2.0, 31)
    SOC_CUTOFF = 0.02          # robot_env terminates at SOC <= 0.02

    def __init__(self, robot: RobotParams, config: dict, n_grid: int = 161):
        super().__init__(robot, config, n_grid)
        self.v_fast = float(config.get('baseline_v_fast', 3.9))
        self.soc_reserve = float(config.get('baseline_soc_reserve', robot.soc_reserve))
        self.safety = float(config.get('baseline_energy_safety', 0.05))
        self.last_safety = float(config.get('baseline_last_safety', 0.01))
        self.soc_floor = self.SOC_CUTOFF + float(config.get('baseline_floor_margin', 0.005))
        self.replan_s = int(config.get('baseline_replan_s', 10))
        self._s = np.linspace(0.0, 1.0, 401)

    def reset(self):
        super().reset()
        self._mode, self._action, self._k = 'fast', None, 0
        self._soc = self._soh = 1.0
        self._d_rem = 0.0

    # --- charge budget (nominal OCV(SOC, SOH), capacity Q_n * SOH) -------
    def _usable_energy_j(self, floor: float) -> float:
        if self._soc <= floor:
            return 0.0
        s = np.linspace(floor, self._soc, 25)
        ocv = np.array([_voc(x, self._soh) for x in s])
        return float(self.p.Q_n * self._soh * np.trapezoid(ocv, s))

    def _e_per_m(self, v, f, c) -> float:
        return float((motor_power(v, 0.0, self.p) + non_motor_power(v, f, c, self.p)) / v)

    def _fast_action(self, q_ref, c):
        p, q = self.p, q_ref * self.margin
        v = float(np.clip(min(self.v_fast, throughput_max_speed(p.f_max, q, c)), p.v_min, p.v_max))
        return v, float(np.clip(throughput_freq_required(v, q, c), p.f_min, p.f_max))

    def _plan(self, q_ref, c):
        p, d = self.p, self._d_rem
        v_f, f_f = self._fast_action(q_ref, c)
        v_o, f_o = QoSAwareEPD.optimise(self, q_ref, c)
        e_res = self._usable_energy_j(self.soc_reserve)
        k = 1.0 + self.safety
        if k * self._e_per_m(v_f, f_f, c) * d <= e_res:
            return 'fast', None
        if k * self._e_per_m(v_o, f_o, c) * d <= e_res:
            return 'optimal', None
        e_cut = self._usable_energy_j(self.soc_floor)
        k = 1.0 + self.last_safety
        if k * self._e_per_m(v_o, f_o, c) * d <= e_cut:
            return 'optimal', None
        budget = e_cut / max(k * d, 1e-9)
        best, best_key = None, None
        for f in self.F_GRID:
            e = (self._p_motor + non_motor_power(self.v_grid, f, c, p)) / self.v_grid
            ratio = np.minimum(compute_throughput(self.v_grid, f, c) / q_ref, 1.0)
            for i in np.flatnonzero(e <= budget):
                key = (ratio[i], -e[i])
                if best_key is None or key > best_key:
                    best_key, best = key, (float(self.v_grid[i]), float(f))
        if best is None:   # nothing fits: minimum energy, get as far as possible
            e = (self._p_motor + non_motor_power(self.v_grid, p.f_min, c, p)) / self.v_grid
            best = (float(self.v_grid[int(np.argmin(e))]), float(p.f_min))
        return 'budget', best

    def optimise(self, q_ref: float, c: float):
        # Mode re-decided every replan_s steps; fast / optimal actions follow the
        # current complexity estimate every step.
        if self._k % self.replan_s == 0:
            self._mode, self._action = self._plan(q_ref, c)
        self._k += 1
        if self._mode == 'fast':
            return self._fast_action(q_ref, c)
        if self._mode == 'optimal':
            return QoSAwareEPD.optimise(self, q_ref, c)
        return self._action

    def select_action(self, obs, deterministic=True):
        self._soc, self._soh = float(obs[2]), float(np.clip(obs[3], 1e-3, 1.0))
        self._d_rem = max(0.0, float(obs[6]))
        return super().select_action(obs, deterministic)


class BAEPDSohEstimate(BatteryAwareEPD):
    """BA-EPD + voltage-based SOH estimate under BMS observations (SohEstimator)."""

    name = 'BA-EPD+SOH'

    def __init__(self, robot: RobotParams, config: dict, n_grid: int = 161):
        super().__init__(robot, config, n_grid)
        self.est = SohEstimator(robot, config)

    def reset(self):
        super().reset()
        if hasattr(self, 'est'):
            self.est.reset()

    @property
    def soh_hat(self) -> float:
        return self.est.soh

    def select_action(self, obs, deterministic=True):
        self.est.update(obs)
        return super().select_action(self.est.corrected(obs), deterministic)
