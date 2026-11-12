"""Comparison methods.

Rule-based (this module, make_baselines):
  AS-MF    Average Speed, Maximum CPU Frequency. No battery or QoS awareness.
  QoS-EPD  QoS-aware Energy-Per-Distance optimiser (non-RL): each step picks
           the speed that minimises electrical energy per metre
               min_v  (P_motor(v) + P_aux + P_cpu(f_req(v))) / v
               s.t.   throughput(v, f_req(v), c_safe) >= margin * QoS_ref,
                      f_req(v) <= f_max
           and runs the CPU at f_req(v), the minimum frequency meeting QoS.
           Workload complexity is not observed: it is estimated on-line from
           the observed throughput (EMA) and inflated by the worst-case
           disturbance (c_safe). Battery-state agnostic (SOC/SOH unused).

RL ablations (SAC, same reward / network / training budget as DARO; only the
battery observations differ, see obs_variants.py):
  RL-SOC-only  python sac_training.py --obs soc  -> checkpoints/rl_soc/seed_k
  RL-SOH-only  python sac_training.py --obs soh  -> checkpoints/rl_soh/seed_k
Load them like DARO (runner.load_daro / load_rl_policies); the variant is read
from the checkpoint directory.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from robot_env import (RobotParams, motor_power, compute_cpu_power,
                       throughput_freq_required, throughput_complexity)
from .test_env import make_robot


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
        self._p_motor_aux = motor_power(self.v_grid, 0.0, p) + p.aux_power_w
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
        e_per_m = (self._p_motor_aux + compute_cpu_power(f, p.compute_base_w,
                                                         p.compute_dynamic_w)) / self.v_grid
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


def make_baselines(config: dict) -> Dict[str, object]:
    """Rule-based baselines. RL ablations are checkpoints (see module docstring)."""
    robot = make_robot(config)
    return {
        'AS-MF':   ASMFBaseline(robot, config),
        'QoS-EPD': QoSAwareEPD(robot, config),
    }
