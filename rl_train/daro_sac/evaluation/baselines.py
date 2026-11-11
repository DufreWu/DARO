"""Rule-based baselines (ablation over the battery information used).

  AS-MF     Average Speed, Maximum CPU Frequency. No battery or QoS awareness.
  SOC-only  QoS-aware DVFS + speed adapted to SOC only.
  Deg-only  QoS-aware DVFS + speed adapted to SOH (degradation) only.
  SOC+Deg   QoS-aware DVFS + speed adapted to SOC and SOH jointly
            (rule-based counterpart of DARO; isolates the gain of RL).

The battery-aware baselines share compute/QoS logic and speed law, differing
ONLY in which battery state they read:

    v = v_eco + w * (v_nom - v_eco),   w = w_soc^[SOC used] * w_soh^[SOH used]
    w_soc = clip((SOC - soc_low) / (soc_high - soc_low), 0, 1)
    w_soh = clip((SOH - SOH_EOL) / (1 - SOH_EOL), 0, 1)

v_nom is the AS-MF speed; v_eco minimises electrical energy per metre
(motor + compute + aux) at the QoS-required CPU frequency. Speed is capped so
QoS stays achievable at f_max. CPU frequency is the minimum meeting QoS, using
an on-line estimate of workload complexity from observed throughput
(complexity itself is not observed), inflated by the worst-case disturbance.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from robot_env import (RobotParams, motor_power, compute_cpu_power,
                       throughput_freq_required, throughput_max_speed,
                       throughput_complexity)
from .test_env import make_robot


class ASMFBaseline:
    name = 'AS-MF'

    def __init__(self, robot: RobotParams, config: dict):
        self.action = np.array([config['baseline_v_nom'], robot.f_max], dtype=np.float32)

    def reset(self):
        pass

    def select_action(self, obs, deterministic=True):
        return self.action.copy()


class RuleBasedBaseline:
    def __init__(self, robot: RobotParams, config: dict,
                 use_soc: bool, use_soh: bool, name: str):
        self.p, self.use_soc, self.use_soh, self.name = robot, use_soc, use_soh, name
        self.v_nom = float(config['baseline_v_nom'])
        self.soc_low = float(config['baseline_soc_low'])
        self.soc_high = float(config['baseline_soc_high'])
        self.margin = float(config['baseline_qos_margin'])
        self.ema = float(config['baseline_c_ema'])
        self.jitter = float(config['test_complexity_jitter'])
        self.n = robot.n_apps
        self.v_grid = np.linspace(robot.v_min, robot.v_max, 91)
        p = robot
        # Motor + aux power on the grid (steady speed); compute added per step.
        self._p_motor_aux = motor_power(self.v_grid, 0.0, p) + p.aux_power_w
        self.reset()

    def reset(self):
        self.c_hat: Optional[float] = None

    # --- compute side ---------------------------------------------------
    def _update_complexity(self, v, f, thr):
        # Invert robot_env.compute_throughput for c (needs v > 0 to be identifiable).
        c = throughput_complexity(v, f, thr) if v > 0.1 else None
        if c is not None:
            c = float(np.clip(c, 0.1, 10.0))
            self.c_hat = c if self.c_hat is None else (1 - self.ema) * self.c_hat + self.ema * c

    def _c_safe(self) -> float:
        c = 1.0 if self.c_hat is None else self.c_hat
        return c * (1.0 + self.jitter)

    def _f_required(self, v, q_ref, c):
        f = throughput_freq_required(v, q_ref * self.margin, c)
        return np.clip(f, self.p.f_min, self.p.f_max)

    def _v_qos_max(self, q_ref, c) -> float:
        # 0 when QoS is unreachable even at f_max -> clipped to v_min
        return max(self.p.v_min, throughput_max_speed(self.p.f_max, q_ref * self.margin, c))

    # --- speed side -----------------------------------------------------
    def _v_eco(self, q_ref, c) -> float:
        p = self.p
        f = self._f_required(self.v_grid, q_ref, c)
        compute = compute_cpu_power(f, p.compute_base_w, p.compute_dynamic_w)
        return float(self.v_grid[np.argmin((self._p_motor_aux + compute) / self.v_grid)])

    def _weight(self, soc, soh) -> float:
        w = 1.0
        if self.use_soc:
            w *= float(np.clip((soc - self.soc_low) / (self.soc_high - self.soc_low), 0, 1))
        if self.use_soh:
            w *= float(np.clip((soh - self.p.SOH_EOL) / (1.0 - self.p.SOH_EOL), 0, 1))
        return w

    def select_action(self, obs, deterministic=True):
        v, f, soc, soh = (float(x) for x in obs[:4])
        thr = np.asarray(obs[8:8 + self.n], dtype=float)
        q_ref = float(np.max(obs[8 + self.n:8 + 2 * self.n]))
        self._update_complexity(v, f, float(np.min(thr)))
        c = self._c_safe()
        v_eco = min(self._v_eco(q_ref, c), self.v_nom)
        v_cmd = v_eco + self._weight(soc, soh) * (self.v_nom - v_eco)
        v_cmd = float(np.clip(min(v_cmd, self._v_qos_max(q_ref, c)), self.p.v_min, self.p.v_max))
        f_cmd = float(self._f_required(v_cmd, q_ref, c))
        return np.array([v_cmd, f_cmd], dtype=np.float32)


def make_baselines(config: dict) -> Dict[str, object]:
    robot = make_robot(config)
    return {
        'AS-MF':    ASMFBaseline(robot, config),
        'SOC-only': RuleBasedBaseline(robot, config, True, False, 'SOC-only'),
        'Deg-only': RuleBasedBaseline(robot, config, False, True, 'Deg-only'),
        'SOC+Deg':  RuleBasedBaseline(robot, config, True, True, 'SOC+Deg'),
    }
