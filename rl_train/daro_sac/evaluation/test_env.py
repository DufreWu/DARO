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
from typing import List

from robot_env import RobotEnv, RobotParams, ECMParams, StressParams

EVAL_CONFIG = {
    "control_dt":             1.0,
    "max_steps_margin":       1.2,
    "mission_distance_m":     500.0,
    "seed":                   42,
    # Test grid
    "test_soh_levels":        [1.00, 0.95, 0.90, 0.85, 0.80],
    "test_soc_levels":        [1.00, 0.60, 0.30],
    "test_complexity_levels": [1.0, 2.0, 3.0],
    "test_complexity_jitter": 0.2,
    "test_episodes":          10,     # seeds per scenario
    # Baselines
    "baseline_v_nom":         2.5,    # AS-MF speed [m/s]
    "baseline_soc_low":       0.20,
    "baseline_soc_high":      0.60,
    "baseline_qos_margin":    1.05,
    "baseline_c_ema":         0.3,
    # Lifetime (charge half-cycle after each mission)
    "charge_c_rate":          0.5,
    "log_dir":                "./logs",
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
    robot.mission_distance_m = float(config.get('mission_distance_m', robot.mission_distance_m))
    robot.max_steps = episode_max_steps(config, robot)
    return robot


def make_test_env(config: dict, scenario: TestScenario) -> RobotEnv:
    robot = make_robot(config)
    return RobotEnv(robot=robot, ecm=ECMParams.default(robot), stress=StressParams(),
                    soc_init=scenario.soc_init, soh_init=scenario.soh,
                    complexity=scenario.complexity, mode='test',
                    complexity_jitter=config['test_complexity_jitter'])
