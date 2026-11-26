"""DA-Lag-RLS: degradation-aware Lag-RLS (battery short- AND long-term objective).

Lag-RLS (evaluation/baselines.LagRLSEPD, kept unchanged as the baseline) prices time,
energy and the terminal SOC reserve. DA-Lag-RLS adds the battery's long-term cost to the
objective:

    J = return - w_deg * dSOH_mission / (1 - SOH_EOL)

i.e. w_deg is the cost (in reward units) of consuming one full battery life
(SOH 1 -> SOH_EOL), e.g. the replacement cost. dSOH_mission is the Xu model exactly as
the evaluation computes it (evaluation/metrics.xu_mission_fd): discharge half-cycle
(DoD, mean SOC, mean temperature) + calendar ageing while driving, + charge half-cycle
at ambient + calendar ageing during the CC recharge (DoD / C-rate).

The controller predicts dSOH_mission from its battery state: the SOC at mission start
and now (BMS SOC corrected with the on-line SOH estimate), the estimated SOH, the elapsed
time, the temperature, and, per candidate energy price lam, the predicted end SOC and
remaining time. Two places use it:

  * lam selection (every replan_s steps): lam minimises the predicted remaining cost
        time + energy + reserve penalty + w_deg * dSOH_pred(lam) / (1 - SOH_EOL);
  * per-step speed choice: the degradation cost is linearised at the planned operating
    point into a price per joule mu_E (cycle ageing through the DoD) and per second mu_t
    (calendar ageing), so the Lagrangian rule keeps its form
        v = argmin_v [((1 + lam) w_E / L + eta mu_E) e(v, c) + (w_T / T_ref + mu_t)] / v.

deg_weight = 0 reduces exactly to Lag-RLS.
"""

from __future__ import annotations

import math

import numpy as np

from battery_model import _DEFAULT_AGING_PARAMS, _xu_fd_halfcycle, _xu_calendar_fd
from envs.robot_env import RobotParams, _voc, quantize_freq

from .baselines import BatteryAwareEPD
from .lag_rls import LagRLSEPD


class XuMissionModel:
    """Predicted mission dSOH (fraction of capacity), same terms as metrics.xu_mission_fd."""

    def __init__(self, robot: RobotParams, charge_c_rate: float):
        self.ap = _DEFAULT_AGING_PARAMS
        self.t_amb_k = robot.T_amb + 273.15
        self.c_rate = float(charge_c_rate)

    def dsoh(self, soc0, soc_now, soc_end, t_el, t_rem, temp_past_k, temp_now_k, soh) -> float:
        ap = self.ap
        soc_end = float(np.clip(soc_end, 0.0, 1.0))
        dod = max(0.0, soc0 - soc_end)
        avg = 0.5 * (soc0 + soc_end)
        t_tot = max(t_el + t_rem, 1e-9)
        t_mean = (t_el * temp_past_k + t_rem * temp_now_k) / t_tot
        fd = (_xu_fd_halfcycle(dod, avg, t_mean, **ap)
              + _xu_calendar_fd(soc0, soc_now, t_el, temp_past_k, **ap)
              + _xu_calendar_fd(soc_now, soc_end, t_rem, temp_now_k, **ap))
        t_chg = dod * 3600.0 / self.c_rate
        fd += (_xu_fd_halfcycle(dod, avg, self.t_amb_k, **ap)
               + _xu_calendar_fd(soc_end, soc0, t_chg, self.t_amb_k, **ap))
        return soh * (1.0 - math.exp(-fd))


class DegAwareLagRLS(LagRLSEPD):
    """Lag-RLS with the Xu degradation cost in the objective (see module docstring)."""

    name = 'DA-Lag-RLS'

    def __init__(self, robot: RobotParams, config: dict, n_grid: int = 161):
        self.w_deg = float(config.get('deg_weight', 1000.0))
        self.deg = XuMissionModel(robot, config.get('charge_c_rate', 1.0))
        self.soh_eol = float(robot.SOH_EOL)
        super().__init__(robot, config, n_grid)

    def reset(self):
        super().reset()
        self._t_el, self._temp_sum, self._temp_now = 0.0, 0.0, None
        self._soc_start = None
        self._mu_E, self._mu_t = 0.0, 0.0

    # --- bookkeeping --------------------------------------------------------
    def select_action(self, obs, deterministic=True):
        temp_k = float(obs[4]) + 273.15
        if self._temp_now is not None:          # time elapsed since the previous step
            self._t_el += self.p.dt
            self._temp_sum += self._temp_now * self.p.dt
        self._temp_now = temp_k
        return super().select_action(obs, deterministic)

    def _temp_past(self) -> float:
        return self._temp_sum / self._t_el if self._t_el > 0 else self._temp_now

    # --- degradation cost ------------------------------------------------------
    def _deg_cost(self, e_cell_rem: float, t_rem: float) -> float:
        """w_deg * predicted mission dSOH / (1 - SOH_EOL) for remaining cell energy / time."""
        if self.w_deg <= 0 or self._soc_start is None:
            return 0.0
        soh = max(self._soh, 1e-3)
        ocv = _voc(max(self._soc - 0.5 * e_cell_rem / (self.p.Q_n * soh * _voc(self._soc, soh)), 0.0), soh)
        soc_end = self._soc - e_cell_rem / (self.p.Q_n * soh * ocv)
        d = self.deg.dsoh(self._soc_start, self._soc, soc_end, self._t_el, t_rem,
                          self._temp_past(), self._temp_now, soh)
        return self.w_deg * d / max(1.0 - self.soh_eol, 1e-6)

    def _update_prices(self, e_cell_rem: float, t_rem: float):
        """Linearise the degradation cost into per-joule and per-second prices."""
        if self.w_deg <= 0:
            self._mu_E = self._mu_t = 0.0
            return
        c0 = self._deg_cost(e_cell_rem, t_rem)
        dE, dT = max(0.01 * e_cell_rem, 1.0), max(0.01 * t_rem, 1.0)
        self._mu_E = max(0.0, (self._deg_cost(e_cell_rem + dE, t_rem) - c0) / dE)
        self._mu_t = max(0.0, (self._deg_cost(e_cell_rem, t_rem + dT) - c0) / dT)

    # --- Lagrangian schedule with degradation prices -------------------------------
    def _coefs(self, lam: float):
        """Energy / time prices of the schedule plus the linearised degradation prices."""
        coef_e, coef_t = super()._coefs(lam)
        return coef_e + self._eta * self._mu_E, coef_t + self._mu_t

    def _f_req(self, c, q_ref):
        from envs.robot_env import throughput_freq_required
        return throughput_freq_required(self.v_grid, q_ref * self.margin, c * (1.0 + self.jitter))

    def _p_nonmotor(self, f, c):
        from envs.robot_env import non_motor_power
        return non_motor_power(self.v_grid, f, c, self.p)

    def _plan_lag(self, q_ref, c_safe):
        """lag_select='tier' (ablation): the tier rule on the schedules with the degradation
        prices (linearised at the current plan, then re-linearised at the chosen lam)."""
        if self.select == 'cost':
            return super()._plan_lag(q_ref, c_safe)
        if self._soc_start is None:
            self._soc_start = self._soc
        c_fc = self._c_sum / self._c_n if self._c_n else (self.c_hat or 1.0)
        d = self._d_rem
        v0, _, e0 = self._schedule(c_fc, q_ref, self._lam)
        self._update_prices(self._eta * e0 * d, d / v0)
        mode, x = super()._plan_lag(q_ref, c_safe)
        if mode == 'lag':
            v, _, e = self._schedule(c_fc, q_ref, x)
            self._update_prices(self._eta * e * d, d / v)
        return mode, x

    def _plan_cost(self, q_ref, c_safe, c_fc, d):
        if self._soc_start is None:
            self._soc_start = self._soc
        p = self.p
        L = max(self._L, 1.0)
        T_ref = L / p.v_ref_mps
        e_res = self._usable_energy_j(self.soc_reserve)
        e_cut = self._usable_energy_j(self.soc_floor)
        j_per_soc = p.Q_n * self._soh * _voc(self.soc_reserve, self._soh)
        # Prices at the current plan, then search lam, then re-linearise at the new plan.
        v0, _, e0 = self._schedule(c_fc, q_ref, self._lam)
        self._update_prices(self._eta * e0 * d, d / v0)
        best, best_cost, best_ve = None, np.inf, None
        for lam in self.LAMS:
            v, _, e = self._schedule(c_fc, q_ref, lam)
            if self._eta * (1.0 + self.last_safety) * e * d > e_cut:
                continue                                   # predicted failure
            e_cell = self._eta * (1.0 + self.safety) * e * d
            short = max(0.0, e_cell - e_res) / j_per_soc / max(p.soc_reserve, 1e-6)
            cost = (p.w_time * (d / v) / T_ref + p.w_energy * self._eta * e * d / L
                    + p.w_reserve * min(1.0, short) + self._deg_cost(self._eta * e * d, d / v))
            if cost < best_cost:
                best, best_cost, best_ve = lam, cost, (v, e)
        if best is not None:
            self._update_prices(self._eta * best_ve[1] * d, d / best_ve[0])
            return 'lag', best
        mode, action = BatteryAwareEPD._plan(self, q_ref, c_safe)   # QoS must be relaxed
        return ('budget', action) if mode == 'budget' else ('lag', self.LAMS[-1])
