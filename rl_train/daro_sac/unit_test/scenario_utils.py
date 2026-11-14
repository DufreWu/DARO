"""Shared helpers for the scenario checks (no RL training needed).

Policies are variants of the QoS-EPD optimiser (evaluation/baselines.py), so
differences come only from what each one knows:

  EPD           QoS-EPD: model of `robot` (nominal calibration unless given the
                true one), complexity estimated online (EMA) + jitter margin.
  EPD(known_c)  oracle complexity: uses the true nominal complexity each step.
  FixedSpeed    constant speed, CPU frequency from the same complexity estimate.
  BatteryAware  = evaluation.baselines.BatteryAwareEPD (BA-EPD): drive fast while the
                usable energy (observed SOC, SOH; nominal OCV) covers the rest of the
                mission, else energy-optimal (using the reserve before QoS), else keep
                as much QoS as the energy budget allows to finish.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl_train/daro_sac

import numpy as np

import evaluation  # noqa: F401  (sets sys.path)
from evaluation import EVAL_CONFIG, make_test_env, make_robot, TestScenario
from evaluation.baselines import QoSAwareEPD, ASMFBaseline, BatteryAwareEPD
from robot_env import (motor_power, non_motor_power, throughput_freq_required,
                       throughput_max_speed, compute_throughput, _voc)


def make_config(distance, jitter=0.2):
    return dict(EVAL_CONFIG, mission_distance_m=float(distance), test_complexity_jitter=jitter)


def make_env(distance, soh=1.0, soc0=1.0, complexity=1.0, jitter=0.2, **robot_overrides):
    """Test env; robot_overrides change the TRUE plant (e.g. motor_cubic_w_per_mps3)."""
    cfg = make_config(distance, jitter)
    env = make_test_env(cfg, TestScenario(soh, soc0, complexity))
    for k, v in robot_overrides.items():
        if not hasattr(env.robot, k):
            raise AttributeError(k)
        setattr(env.robot, k, v)
    return env, cfg


def run(env, policy, seed: int,
        hook: Optional[Callable[[object, int], None]] = None) -> dict:
    """One episode. hook(env, step) runs before every action (e.g. complexity changes)."""
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, 'reset'):
        policy.reset()
    if hasattr(policy, 'attach'):
        policy.attach(env)
    done, steps, viol, qos_sum, info = False, 0, 0, 0.0, {}
    while not done:
        if hook is not None:
            hook(env, steps)
        obs, _, term, trunc, info = env.step(policy.select_action(obs))
        done = term or trunc
        viol += int(info['qos_violation'] > 0.0)
        qos_sum += 1.0 - min(1.0, info['qos_violation'])
        steps += 1
    t = float(info.get('t_s', 0.0))
    return {
        'energy_kj': env.total_energy_j / 1e3,
        'wh_per_km': env.total_energy_j / 3.6 / max(env.distance_m, 1e-9),
        'time_s': t,
        'avg_speed': env.distance_m / max(t, 1e-9),
        'qos_viol_pct': 100.0 * viol / max(steps, 1),          # steps below QoS_ref
        'qos_level_pct': 100.0 * qos_sum / max(steps, 1),      # mean delivered thr / QoS_ref (capped at 1)
        'completed': float(bool(info.get('completed', False))),
        'final_soc': float(env.battery.soc),
        'distance_km': env.distance_m / 1000.0,
    }


def mean_runs(results):
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


class EPD(QoSAwareEPD):
    """QoS-EPD with an optional known complexity (oracle) and margin choice."""

    def __init__(self, robot, config, known_c: Optional[float] = None,
                 jitter_margin: bool = True, name: str = 'QoS-EPD'):
        super().__init__(robot, config)
        self.known_c = known_c
        self.jitter_margin = jitter_margin
        self.name = name

    def _c_safe(self) -> float:
        if self.known_c is not None:
            return self.known_c * ((1.0 + self.jitter) if self.jitter_margin else 1.0)
        return super()._c_safe()


class FixedSpeed(EPD):
    """Constant speed; CPU frequency = QoS frequency for the estimated complexity."""

    def __init__(self, robot, config, speed: float, **kw):
        super().__init__(robot, config, **kw)
        self.speed = float(speed)

    def optimise(self, q_ref: float, c: float):
        p = self.p
        f = float(throughput_freq_required(self.speed, q_ref * self.margin, c))
        return self.speed, float(np.clip(f, p.f_min, p.f_max))


def nominal_robot(config):
    return make_robot(config)


def true_robot(env):
    """Copy of the env's (true) plant parameters, for oracle models."""
    return copy.deepcopy(env.robot)


class BatteryAware(BatteryAwareEPD):
    """evaluation.baselines.BatteryAwareEPD (BA-EPD; observed SOC / SOH / distance)
    with the scenario-check options: v_fast and an optional oracle complexity."""

    def __init__(self, robot, config, v_fast: float = 3.9, known_c: Optional[float] = None,
                 jitter_margin: bool = True, name: str = 'battery-aware'):
        super().__init__(robot, dict(config, baseline_v_fast=v_fast))
        self.known_c, self.jitter_margin, self.name = known_c, jitter_margin, name

    def attach(self, env):   # kept for run(); the rule uses observations only
        pass

    def _c_safe(self) -> float:
        if self.known_c is not None:
            return self.known_c * ((1.0 + self.jitter) if self.jitter_margin else 1.0)
        return super()._c_safe()
