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
                            throughput_complexity, throughput_max_speed, _voc, quantize_freq,
                            motor_thermal_factor, compute_throughput, CPU_FREQ_GHZ)

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

    QoS-aware DVFS (lag_qos='expected', default): instead of the fixed chance-constraint
    allowance, (v, f) are chosen jointly over the speed grid x DVFS levels by the per-metre
    cost incl. the expected QoS penalty of the reward, w_Q / T_ref E[1(viol) + viol] / v,
    under the known per-step complexity disturbance c_hat U(1 - j, 1 + j) and the workload-
    estimate error U(1 - lag_c_unc, 1 + lag_c_unc), subject to the
    violation probability <= qos_viol_target. A DVFS level that violates QoS now and then is
    used only where the energy it saves outweighs the expected penalty.
    lag_qos='allowance' (ablation): f = minimum level meeting QoS at c (1 + allowance).

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
        self.qos_mode = str(config.get('lag_qos', 'expected'))
        self.jit_raw = float(config.get('test_complexity_jitter', 0.0))   # known disturbance
        self.rho = float(config.get('qos_viol_target', 0.0))
        self.c_unc = float(config.get('lag_c_unc', 0.03))    # relative error of the workload estimate
        # Extra sensors (variants): 'split' = motor and compute power measured separately
        # (two decoupled RLS fits instead of one on V * I); 'events' = workload from the
        # event count per batch instead of the throughput inversion. Read from the last step
        # of the env attached with attach_sensors(), with relative noise.
        self.power_sensing = str(config.get('lag_power', 'total'))
        self.c_source = str(config.get('lag_c_source', 'throughput'))
        self.power_noise = float(config.get('meas_current_noise_frac', 0.01))
        self.event_noise = float(config.get('event_count_noise', 0.02))
        self._sensor_env, self._sensor_rng = None, np.random.default_rng(0)
        self._qc_key, self._qc = None, None
        self._p_nom = copy.copy(robot)       # nominal plant (for the motor thermal factor)
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

    # --- extra sensors (variants) ---------------------------------------------
    @property
    def needs_sensors(self) -> bool:
        return self.power_sensing == 'split' or self.c_source == 'events'

    def attach_sensors(self, env, seed: int = 0):
        """Give the variant access to the per-step motor / compute power and event count."""
        self._sensor_env, self._sensor_rng = env, np.random.default_rng(seed)

    def _sensor(self, key: str, noise: float):
        env = self._sensor_env
        if env is None or not env.trajectory:
            return None
        x = float(env.trajectory[-1][key])
        return x * (1.0 + float(self._sensor_rng.normal(0.0, noise)))

    # --- on-line power model (RLS) ---------------------------------------
    def _apply(self):
        n, p = self._nom, self.p
        p.motor_base_w = self.th[0] * n['motor_base_w']
        p.motor_linear_w_per_mps = self.th[0] * n['motor_linear_w_per_mps']
        p.motor_cubic_w_per_mps3 = self.th[1] * n['motor_cubic_w_per_mps3']
        p.compute_scale = self.th[2]
        self._p_motor = motor_power(self.v_grid, 0.0, p)

    def _rls_step(self, phi, y):
        Pphi = self.P @ phi
        k = Pphi / (self.forget + phi @ Pphi)
        self.th = np.clip(self.th + k * (y - phi @ self.th), *self.th_clip)
        self.P = (self.P - np.outer(k, Pphi)) / self.forget

    def _update_rls(self, obs):
        if not self.use_rls:
            return
        v, f = float(obs[0]), float(obs[1])
        steady = self._v_prev is not None and abs(v - self._v_prev) < 0.02
        self._v_prev = v
        split = self.power_sensing == 'split' and self._sensor_env is not None
        if split and v > 0.1:              # compute power: own scalar fit, any speed
            pc = self._sensor('compute_power_w', self.power_noise)
            if pc is not None and pc > 0:
                c = 1.0 if self.c_hat is None else float(self.c_hat)
                p1 = copy.copy(self.p)
                p1.compute_scale = 1.0
                self._rls_step(np.array([0.0, 0.0, float(non_motor_power(v, f, c, p1))]), pc)
        if v < 0.3 or not steady:          # the acceleration term is not modelled
            if split:
                self._apply()
            return
        n = self._nom
        c = 1.0 if self.c_hat is None else float(self.c_hat)
        p1 = copy.copy(self.p)
        p1.compute_scale = 1.0
        # Steady-state motor temperature loss at v (nominal plant); th then scales the
        # cold-fit coefficients and _apply() re-applies the factor via motor_power.
        k_th = float(motor_thermal_factor(v, self._p_nom))
        phi = np.array([k_th * (n['motor_base_w'] + n['motor_linear_w_per_mps'] * v),
                        k_th * n['motor_cubic_w_per_mps3'] * v ** 3,
                        float(non_motor_power(v, f, c, p1))])
        if split:                          # motor power: 2-parameter fit (P stays block-diagonal)
            y = self._sensor('motor_power_w', self.power_noise)
            if y is not None and y > 0:
                phi[2] = 0.0
                self._rls_step(phi, y)
            self._apply()
            return
        y = float(obs[5]) * float(obs[7])   # terminal voltage x current = load power
        if y <= 0:
            return
        self._rls_step(phi, y)
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
            if self.c_source == 'events' and self._sensor_env is not None:
                c = self._sensor('complexity', self.event_noise)   # event count per batch
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
    U_Q = np.linspace(-1.0, 1.0, 41)          # quadrature of the uniform complexity disturbance
    E_Q = np.linspace(-1.0, 1.0, 7)           # quadrature of the workload-estimate error

    def _coefs(self, lam: float):
        """Per-metre price of energy [1/J] and per-second price of time of the schedule."""
        p = self.p
        L = max(self._L, 1.0)
        return (1.0 + lam) * p.w_energy / L, p.w_time / (L / p.v_ref_mps)

    def _qos_cost(self, c: float, q_ref: float):
        """Expected QoS penalty per second E[1(viol) + viol] and violation probability,
        per (speed, DVFS level), at workload c with the known disturbance and the
        estimate error c_unc (both uniform; cached)."""
        key = (round(float(c), 4), float(q_ref))
        if key != self._qc_key:
            cs = (c * (1.0 + self.jit_raw * self.U_Q)[:, None] * (1.0 + self.c_unc * self.E_Q)[None, :]).ravel()
            thr = compute_throughput(self.v_grid[:, None, None], CPU_FREQ_GHZ[None, :, None],
                                     cs[None, None, :])
            viol = np.clip(1.0 - thr / q_ref, 0.0, 1.0)
            self._qc = (np.mean((viol > 0) + viol, axis=2), np.mean(viol > 0, axis=2))
            self._qc_key = key
        return self._qc

    def _schedule(self, c: float, q_ref: float, lam: float):
        """(v, f, energy per metre) of the Lagrangian rule at workload c."""
        p = self.p
        coef_e, coef_t = self._coefs(lam)
        if self.qos_mode == 'expected':
            qc, pv = self._qos_cost(c, q_ref)
            lv = CPU_FREQ_GHZ <= p.f_max + 1e-9
            e2 = (self._p_motor[:, None]
                  + non_motor_power(self.v_grid[:, None], CPU_FREQ_GHZ[None, :], c, p)) / self.v_grid[:, None]
            L = max(self._L, 1.0)
            coef_q = p.w_qos / (L / p.v_ref_mps)
            cost2 = coef_e * e2 + (coef_t + coef_q * qc) / self.v_grid[:, None]
            ok2 = (pv <= self.rho + 1e-9) & lv[None, :]
            if ok2.any():
                i, l = np.unravel_index(int(np.argmin(np.where(ok2, cost2, np.inf))), cost2.shape)
                return float(self.v_grid[i]), float(CPU_FREQ_GHZ[l]), float(e2[i, l])
        f_req = throughput_freq_required(self.v_grid, q_ref * self.margin, c * (1.0 + self.jitter))
        ok = f_req <= p.f_max + 1e-9
        f = quantize_freq(f_req)
        e = (self._p_motor + non_motor_power(self.v_grid, f, c, p)) / self.v_grid
        cost = coef_e * e + coef_t / self.v_grid
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
            f = quantize_freq(throughput_freq_required(v, q_ref * self.margin, c_safe))
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
