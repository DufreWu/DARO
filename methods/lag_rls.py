"""Lag-RLS (proposed): battery-state-aware Lagrangian co-control of speed and CPU frequency.

Short-term state: the SOC budget is priced by a Lagrange multiplier (cost-based selection
with a terminal reserve penalty). Long-term state: SOH (capacity, R0, OCV shape) is unknown
under BMS observations and estimated on-line (soh_estimator.SohEstimator). Power models are
identified on-line by RLS. Ablation switches: lag_soh, lag_rls, lag_probe, lag_eta, lag_cd,
lag_lambda, lag_select.
"""

from __future__ import annotations

import copy

import numpy as np

from envs.robot_env import (RobotParams, motor_power, non_motor_power, throughput_freq_required,
                            throughput_complexity, throughput_max_speed, _voc)

from .baselines import QoSAwareEPD, BatteryAwareEPD
from .soh_estimator import SohEstimator


class LagRLSEPD(BatteryAwareEPD):
    """Lag-RLS: battery-state-aware Lagrangian co-control of speed and CPU frequency.

    Short-term battery state: the charge budget above the SOC reserve (SOC).
    Long-term battery state: SOH (capacity, R0, OCV shape), estimated on-line from BMS
    observations (SohEstimator) and used for the true SOC and the usable energy.

    Per step (1 s): v = argmin_v [(1 + lam) w_E/L P(v, f_q(v, c), c) + w_T/T_ref] / v at the
    current workload estimate c, f = f_q(v, c) (minimum QoS frequency). Every replan_s
    steps lam (energy price, LAMS grid) minimises the predicted remaining cost
        w_T (d / v) / T_ref + w_E eta e d / L + w_reserve * min(1, reserve shortfall)
    over the lams whose predicted energy eta (1 + last_safety) e(v*(lam)) d stays above the
    SOC cutoff (else BA-EPD's QoS-relaxing budget mode); the predicted end SOC uses the
    usable energy from (SOC, SOH), so the long-term state enters the price. lam is the
    co-state of the energy constraint with the terminal reserve penalty (Pontryagin /
    ECMS structure). lag_select='tier' (ablation): smallest lam fitting above the
    reserve, else above the cutoff.

    On-line models: recursive least squares on V * I = th0 (b0 + b1 v) + th1 b3 v^3 +
    th2 P_nonmotor,nom (payload / terrain / compute mismatch), with a short speed-probing
    phase (x1.0, x0.75, x1.15 for probe_s each) so the v^3 term is identifiable; loss
    factor eta = (cell energy implied by the SOC drop) / integrated V * I; workload level
    by segment mean with change detection (increases followed at once).

    Ablation switches (config): lag_soh 'estimate' | 'obs' (use the observation as is:
    true SOH with battery_obs='true', SOH = 1 with 'bms'), lag_rls, lag_probe, lag_eta,
    lag_cd, lag_lambda (False: BA-EPD fast / optimal / budget modes on the same models),
    lag_select 'cost' | 'tier'.
    """

    name = 'Lag-RLS'
    LAMS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 1e3)

    def __init__(self, robot: RobotParams, config: dict, n_grid: int = 161):
        self.soh_mode = str(config.get('lag_soh', 'estimate'))
        self.use_rls = bool(config.get('lag_rls', True))
        self.use_probe = bool(config.get('lag_probe', True)) and self.use_rls
        self.use_eta = bool(config.get('lag_eta', True))
        self.use_cd = bool(config.get('lag_cd', True))
        self.use_lambda = bool(config.get('lag_lambda', True))
        # lam selection: 'tier' = smallest lam fitting above the reserve, else above the
        # cutoff; 'cost' = minimum predicted remaining cost incl. the reserve penalty.
        self.select = str(config.get('lag_select', 'cost'))
        self._nom = {k: getattr(robot, k) for k in
                     ('motor_base_w', 'motor_linear_w_per_mps', 'motor_cubic_w_per_mps3')}
        self.p0 = float(config.get('baseline_rls_p0', 0.1))
        self.forget = float(config.get('baseline_rls_forget', 0.9995))
        self.th_clip = (0.3, 3.0)
        self.cd_rel = float(config.get('baseline_cd_rel', 0.25))
        self.cd_n = int(config.get('baseline_cd_n', 3))
        self.probe_s = int(config.get('baseline_probe_s', 30))
        self.probe = (1.0, 0.75, 1.15)
        self.soh_est = SohEstimator(robot, config)
        super().__init__(copy.copy(robot), config, n_grid)   # own params: RLS updates them

    def reset(self):
        super().reset()
        if hasattr(self, 'soh_est'):        # parent __init__ calls reset() first
            self.soh_est.reset()
            self.th, self.P = np.ones(3), np.eye(3) * self.p0
            self._v_prev = None
            self._apply()
            self._lam, self._L = 0.0, float(self.p.mission_distance_m)
            self._level, self._seg, self._pend = None, [], []
            self._c_sum, self._c_n = 0.0, 0
            self._soc0, self._e_load, self._eta = None, 0.0, 1.0

    # --- on-line power model (RLS) ---------------------------------------
    def _apply(self):
        n, p = self._nom, self.p
        p.motor_base_w = self.th[0] * n['motor_base_w']
        p.motor_linear_w_per_mps = self.th[0] * n['motor_linear_w_per_mps']
        p.motor_cubic_w_per_mps3 = self.th[1] * n['motor_cubic_w_per_mps3']
        p.compute_scale = self.th[2]
        self._p_motor = motor_power(self.v_grid, 0.0, p)

    def _update_rls(self, obs):
        if not self.use_rls:
            return
        v, f = float(obs[0]), float(obs[1])
        steady = self._v_prev is not None and abs(v - self._v_prev) < 0.02
        self._v_prev = v
        if v < 0.3 or not steady:          # the acceleration term is not modelled
            return
        n = self._nom
        c = 1.0 if self.c_hat is None else float(self.c_hat)
        p1 = copy.copy(self.p)
        p1.compute_scale = 1.0
        phi = np.array([n['motor_base_w'] + n['motor_linear_w_per_mps'] * v,
                        n['motor_cubic_w_per_mps3'] * v ** 3,
                        float(non_motor_power(v, f, c, p1))])
        y = float(obs[5]) * float(obs[7])   # terminal voltage x current = load power
        if y <= 0:
            return
        Pphi = self.P @ phi
        k = Pphi / (self.forget + phi @ Pphi)
        self.th = np.clip(self.th + k * (y - phi @ self.th), *self.th_clip)
        self.P = (self.P - np.outer(k, Pphi)) / self.forget
        self._apply()

    # --- cell-loss factor ---------------------------------------------------
    def _cell_energy(self, lo: float, hi: float) -> float:
        s = np.linspace(lo, hi, 25)
        ocv = np.array([_voc(x, self._soh) for x in s])
        return float(self.p.Q_n * self._soh * np.trapezoid(ocv, s))

    def _update_eta(self, obs):
        if not self.use_eta:
            return
        soc = float(obs[2])
        if self._soc0 is None:
            self._soc0 = soc
            return
        self._e_load += max(0.0, float(obs[5]) * float(obs[7])) * self.p.dt
        if self._e_load > 2e3 and self._soc0 - soc > 1e-3:
            self._eta = float(np.clip(self._cell_energy(soc, self._soc0) / self._e_load, 0.9, 1.3))

    # --- workload level -------------------------------------------------------
    def _update_complexity(self, v, f, thr):
        if not self.use_cd:
            QoSAwareEPD._update_complexity(self, v, f, thr)
        else:
            c = throughput_complexity(v, f, thr) if v > 0.1 else None
            if c is None:
                return
            c = float(np.clip(c, 0.1, 10.0))
            if self._level is None:
                self._seg = [c]
            elif c > (1.0 + self.cd_rel) * self._level:      # increase: follow at once (QoS)
                self._seg, self._pend = [c], []
            elif c < (1.0 - self.cd_rel) * self._level:      # decrease: confirm first
                self._pend.append(c)
                if len(self._pend) >= self.cd_n:
                    self._seg, self._pend = list(self._pend), []
            else:
                self._pend = []
                self._seg.append(c)
                self._seg = self._seg[-120:]
            self._level = float(np.mean(self._seg))
            self.c_hat = self._level
        if self.c_hat is not None:
            self._c_sum += self.c_hat
            self._c_n += 1

    # --- Lagrangian schedule ----------------------------------------------------
    def _schedule(self, c: float, q_ref: float, lam: float):
        """(v, f, energy per metre) of the Lagrangian rule at workload c."""
        p = self.p
        f_req = throughput_freq_required(self.v_grid, q_ref * self.margin, c * (1.0 + self.jitter))
        ok = f_req <= p.f_max + 1e-9
        f = np.clip(f_req, p.f_min, p.f_max)
        e = (self._p_motor + non_motor_power(self.v_grid, f, c, p)) / self.v_grid
        L = max(self._L, 1.0)
        cost = (1.0 + lam) * p.w_energy / L * e + p.w_time / (L / p.v_ref_mps) / self.v_grid
        if not ok.any():
            k = int(np.argmin(cost))
            return float(self.v_grid[k]), float(p.f_max), float(e[k])
        k = int(np.argmin(np.where(ok, cost, np.inf)))
        return float(self.v_grid[k]), float(f[k]), float(e[k])

    def _plan_lag(self, q_ref, c_safe):
        c_fc = self._c_sum / self._c_n if self._c_n else (self.c_hat or 1.0)
        d = self._d_rem
        if self.select == 'cost':
            return self._plan_cost(q_ref, c_safe, c_fc, d)
        for floor, safety in ((self.soc_reserve, self.safety), (self.soc_floor, self.last_safety)):
            e_av = self._usable_energy_j(floor)
            for lam in self.LAMS:
                if self._eta * (1.0 + safety) * self._schedule(c_fc, q_ref, lam)[2] * d <= e_av:
                    return 'lag', lam
        mode, action = BatteryAwareEPD._plan(self, q_ref, c_safe)   # QoS must be relaxed
        return ('budget', action) if mode == 'budget' else ('lag', self.LAMS[-1])

    def _plan_cost(self, q_ref, c_safe, c_fc, d):
        """lam minimising the predicted remaining cost (time + energy + terminal reserve
        penalty) among the lams whose predicted energy stays above the SOC cutoff; the
        reserve is used only where the time saved outweighs its penalty."""
        p = self.p
        L = max(self._L, 1.0)
        T_ref = L / p.v_ref_mps
        e_res = self._usable_energy_j(self.soc_reserve)
        e_cut = self._usable_energy_j(self.soc_floor)
        # Energy per unit SOC just below the reserve (shortfall -> SOC).
        j_per_soc = p.Q_n * self._soh * _voc(self.soc_reserve, self._soh)
        best, best_cost = None, np.inf
        for lam in self.LAMS:
            v, _, e = self._schedule(c_fc, q_ref, lam)
            e_cell = self._eta * (1.0 + self.safety) * e * d
            if self._eta * (1.0 + self.last_safety) * e * d > e_cut:
                continue                                   # predicted failure
            short = max(0.0, e_cell - e_res) / j_per_soc / max(p.soc_reserve, 1e-6)
            cost = (p.w_time * (d / v) / T_ref + p.w_energy * self._eta * e * d / L
                    + p.w_reserve * min(1.0, short))
            if cost < best_cost:
                best, best_cost = lam, cost
        if best is not None:
            return 'lag', best
        mode, action = BatteryAwareEPD._plan(self, q_ref, c_safe)   # QoS must be relaxed
        return ('budget', action) if mode == 'budget' else ('lag', self.LAMS[-1])

    def optimise(self, q_ref: float, c: float):
        if not self.use_lambda:                       # ablation: BA-EPD modes on our models
            v, f = BatteryAwareEPD.optimise(self, q_ref, c)
        else:
            if self._k % self.replan_s == 0:
                self._mode, x = self._plan_lag(q_ref, c)
                if self._mode == 'budget':
                    self._action = x
                else:
                    self._lam = x
            self._k += 1
            if self._mode == 'budget':
                return self._action
            v, f = self._schedule(self.c_hat or 1.0, q_ref, self._lam)[:2]
        ph = (self._k - 1) // self.probe_s
        if self.use_probe and ph < len(self.probe) and self.probe[ph] != 1.0 and self._mode != 'budget':
            p, c_safe = self.p, (self.c_hat or 1.0) * (1.0 + self.jitter)
            v = float(np.clip(v * self.probe[ph], p.v_min,
                              min(p.v_max, throughput_max_speed(p.f_max, q_ref * self.margin, c_safe))))
            f = float(np.clip(throughput_freq_required(v, q_ref * self.margin, c_safe), p.f_min, p.f_max))
        return v, f

    @property
    def soh_hat(self) -> float:
        return self.soh_est.soh

    def select_action(self, obs, deterministic=True):
        self._L = float(obs[-2])   # mission length (observation)
        if self.soh_mode == 'estimate':
            self.soh_est.update(obs)
            o = self.soh_est.corrected(obs)
        else:
            o = np.asarray(obs, dtype=np.float32)
        self._update_eta(o)
        self._update_rls(o)
        return BatteryAwareEPD.select_action(self, o, deterministic)
