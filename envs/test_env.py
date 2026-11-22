"""Test environment: scenario grid and environment factory.

RobotEnv in mode='test' (no observation noise) with a seeded per-step
complexity disturbance (``test_complexity_jitter``). Scenarios form a grid
SOH x initial SOC x complexity; every policy sees the same seeds (common
random numbers), so differences come from the policy only.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .robot_env import RobotEnv, RobotParams, ECMParams, StressParams

EVAL_CONFIG = {
    "control_dt":             1.0,
    "max_steps_margin":       1.2,
    "mission_distance_m":     500.0,
    "seed":                   42,
    # Test grid
    "test_soh_levels":        [1.00, 0.95, 0.90, 0.85, 0.80],
    "test_soc_levels":        [1.00, 0.60, 0.30],
    "test_complexity_levels": [1.0, 2.0, 3.0],
    "test_complexity_jitter": 0.1,
    "test_episodes":          10,     # seeds per scenario
    # Baselines
    "baseline_v_nom":         2.5,    # AS-MF speed [m/s]
    "baseline_qos_margin":    1.05,   # used only if qos_viol_target = 0 (worst-case design)
    "qos_viol_target":        0.05,   # acceptable share of steps below the QoS requirement
    "baseline_c_ema":         0.3,
    # BA-EPD (battery-aware QoS-EPD); v_fast ~ reward-optimal speed with enough charge
    "baseline_v_fast":        3.9,
    "baseline_energy_safety": 0.05,   # on the fast / optimal-above-reserve checks
    "baseline_last_safety":   0.01,   # on the optimal-above-cutoff check (before dropping QoS)
    "baseline_floor_margin":  0.005,  # above the SOC cutoff 0.02
    "baseline_replan_s":      10,
    # Lifetime (charge half-cycle after each mission)
    "charge_c_rate":          1.0,    # pack datasheet: recommended charge 1-3 C
    "log_dir":                str(Path(__file__).resolve().parents[1] / 'logs'),   # <repo>/logs
}


@dataclass(frozen=True)
class TestScenario:
    soh: float
    soc_init: float
    complexity: float

    @property
    def name(self) -> str:
        return f"soh{self.soh:.2f}_soc{self.soc_init:.2f}_c{self.complexity:g}"


def test_suite(config: dict) -> List[TestScenario]:
    return [TestScenario(float(h), float(s), float(c)) for h, s, c in itertools.product(
        config['test_soh_levels'], config['test_soc_levels'],
        config['test_complexity_levels'])]


def episode_max_steps(config: dict, robot: RobotParams) -> int:
    """Enough steps to finish the mission at v_min, plus a margin."""
    t_mission = robot.mission_distance_m / robot.v_min
    return int(math.ceil(config['max_steps_margin'] * t_mission / config['control_dt']))


def make_robot(config: dict) -> RobotParams:
    robot = RobotParams(dt=config['control_dt'])
    robot.battery_obs = str(config.get('battery_obs', robot.battery_obs))
    if config.get('battery_realism', False):   # sensor errors + per-episode cell variation
        robot.meas_voltage_noise_v = float(config.get('meas_voltage_noise_v', 0.015))
        robot.meas_current_noise_frac = float(config.get('meas_current_noise_frac', 0.01))
        robot.bms_current_gain_err = float(config.get('bms_current_gain_err', 0.01))
        robot.cell_variation = True
        k = float(config.get('cell_var_scale', 1.0))   # scales the cell-variation ranges
        if k != 1.0:
            lo, hi = robot.cell_r0_scale
            robot.cell_r0_scale = (1.0 + k * (lo - 1.0), 1.0 + k * (hi - 1.0))
            robot.cell_ocv_offset_v = k * robot.cell_ocv_offset_v
            lo, hi = robot.cell_aging_shape
            robot.cell_aging_shape = (1.0 + k * (lo - 1.0), 1.0 + k * (hi - 1.0))
    robot.mission_distance_m = float(config.get('mission_distance_m', robot.mission_distance_m))
    robot.max_steps = episode_max_steps(config, robot)
    return robot


def make_test_env(config: dict, scenario: TestScenario) -> RobotEnv:
    robot = make_robot(config)
    return RobotEnv(robot=robot, ecm=ECMParams.default(robot), stress=StressParams(),
                    soc_init=scenario.soc_init, soh_init=scenario.soh,
                    complexity=scenario.complexity, mode='test',
                    complexity_jitter=config['test_complexity_jitter'])
