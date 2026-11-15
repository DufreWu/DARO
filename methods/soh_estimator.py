"""On-line SOH estimation from BMS observations (shared by BA-EPD+SOH and Lag-RLS).

For each candidate SOH h (grid), SOC_true(h) = SOC0 - (SOC0 - SOC_bms) / h and the predicted
terminal voltage OCV(SOC_true(h), h) - I (R0(h) + R1 + R2) (nominal ECM, steady RC branches);
h = argmin of the EMA of squared voltage residuals over near-steady steps.
"""

from __future__ import annotations

import numpy as np

from envs.robot_env import RobotParams, ECMParams, _voc


class SohEstimator:
    """On-line SOH estimate from BMS observations (battery_obs='bms').

    For each candidate SOH h (grid), SOC_true(h) = SOC0 - (SOC0 - SOC_bms) / h (the BMS
    counts charge against the rated capacity) and the predicted terminal voltage is
    OCV(SOC_true(h), h) - I (R0(h) + R1 + R2) (nominal ECM, steady RC branches);
    h = argmin of the EMA of squared voltage residuals over near-steady steps."""

    H_GRID = np.linspace(0.70, 1.00, 31)

    def __init__(self, robot: RobotParams, config: dict):
        self.p = robot
        self.ecm = ECMParams.default(robot)
        self.res_ema = float(config.get('baseline_soh_ema', 0.02))
        self.settle_s = int(config.get('baseline_soh_settle_s', 30))
        self.di_max = float(config.get('baseline_soh_di_max', 0.05))   # [A] steady-step test
        self.reset()

    def reset(self):
        self.soc0, self._i_prev, self._t, self._n = None, None, 0, 0
        self._score = np.zeros(len(self.H_GRID))
        self.soh = 1.0

    def update(self, obs):
        soc_bms, v_term, i = float(obs[2]), float(obs[5]), float(obs[7])
        if self.soc0 is None:
            self.soc0 = soc_bms
        self._t += 1
        steady = self._i_prev is not None and abs(i - self._i_prev) < self.di_max
        self._i_prev = i
        if self._t < self.settle_s or not steady or i <= 0.05:
            return
        used = self.soc0 - soc_bms                       # charge used / rated capacity
        soc_h = np.clip(self.soc0 - used / self.H_GRID, 0.0, 1.0)
        pred = np.empty(len(self.H_GRID))
        for k, (h, s) in enumerate(zip(self.H_GRID, soc_h)):
            r0, r1, _, r2, _ = self.ecm.values(s)
            r0 *= 1.0 + self.ecm.resistance_aging_factor * (1.0 - h)
            pred[k] = _voc(s, h, self.p.V_nom) - i * (r0 + r1 + r2)
        err2 = (pred - v_term) ** 2
        self._score = err2 if self._n == 0 else (1 - self.res_ema) * self._score + self.res_ema * err2
        self._n += 1
        self.soh = float(self.H_GRID[int(np.argmin(self._score))])

    def corrected(self, obs) -> np.ndarray:
        """Observation with SOC replaced by SOC_true(SOH_hat) and SOH by SOH_hat."""
        o = np.array(obs, dtype=np.float32, copy=True)
        if self.soc0 is not None:
            o[2] = np.clip(self.soc0 - (self.soc0 - float(obs[2])) / self.soh, 0.0, 1.0)
        o[3] = self.soh
        return o
