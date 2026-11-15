"""Claim-oriented test suite: "finish fast subject to charge and QoS, adapting to
model mismatch, workload changes and battery degradation".

Families (all in the training distribution: <= 5 km, SOH 0.8-1.0):

  nominal      1 km, full charge, SOH {1, 0.9, 0.8} x C {1, 2, 3}: speed with ample charge.
  tight        5 km, SOH {1, 0.8} x C {1, 2, 3} x k {0.9, 1.0, 1.1, 1.25}: SOC0 such that
               the usable energy above the 5 % reserve is k x the energy-optimal mission
               energy (envs.robot_env.tight_soc_init). k = 1.25: fast is affordable; k ~ 1:
               energy-optimal only; k = 0.9: the reserve must be used.
  switch       3 km, full charge, complexity re-drawn from {1, 2, 3} with mean period
               {10, 30, 120} s (start C = 2).
  mismatch     2 km, full charge, true plant != nominal model (payload, terrain, compute),
               C {1, 2, 3}. Rule-based baselines keep the nominal model.
  tight_mismatch  5 km, charge-limited as in 'tight' but for the TRUE plant (as training's
               tight episodes): SOH {1, 0.8} x C {1, 2, 3} x k {1.0, 1.1, 1.25} x PLANTS.
               Rule-based baselines budget charge with the nominal model, so model error
               flips their fast / optimal decision (overspend -> reserve / failure, or
               underspend -> needlessly slow).
  degradation  "Same mission, same charge, ageing battery": 5 km, SOC0 fixed per C so that
               at SOH 0.8 energy-optimal driving ends at the reserve (k_eol = 1), SOH swept
               1.0 -> 0.8. Expected battery-aware behaviour: fast while young, energy-optimal
               near end of life, always completing.

The evaluation CLI over this suite is evaluation/scenarios.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .robot_env import RobotEnv, RobotParams, ECMParams, StressParams, tight_soc_init
from .test_env import make_robot

FAMILIES = ('nominal', 'tight', 'switch', 'mismatch', 'tight_mismatch', 'degradation')

_NOM = RobotParams()
PLANTS = {   # true-plant overrides (envs.robot_env.PLANT_KEYS), cf. unit_test/check_model_mismatch.py
    'payload+50%': {'motor_base_w': 1.5 * _NOM.motor_base_w,
                    'motor_linear_w_per_mps': 1.5 * _NOM.motor_linear_w_per_mps,
                    'motor_cubic_w_per_mps3': 1.5 * _NOM.motor_cubic_w_per_mps3},
    'rough_v3x2': {'motor_cubic_w_per_mps3': 2.0 * _NOM.motor_cubic_w_per_mps3},
    'smooth_v3x0.7': {'motor_cubic_w_per_mps3': 0.7 * _NOM.motor_cubic_w_per_mps3},
    'compute_x1.3': {'compute_scale': 1.3},
    'compute_x0.8': {'compute_scale': 0.8},
}


@dataclass
class Scenario:
    family: str
    name: str
    distance_m: float
    soh: float
    soc0: float
    complexity: float
    plant: dict = field(default_factory=dict)
    switch_period_s: float = 0.0
    tags: dict = field(default_factory=dict)   # grouping keys for the printout

    def options(self) -> dict:
        return {'plant': self.plant, 'complexity_switch_period_s': self.switch_period_s,
                'complexity_levels': [1.0, 2.0, 3.0]}


def build_suite(families=FAMILIES, k_eol: float = 1.0) -> List[Scenario]:
    out: List[Scenario] = []
    if 'nominal' in families:
        for soh in (1.0, 0.9, 0.8):
            for c in (1.0, 2.0, 3.0):
                out.append(Scenario('nominal', f'soh{soh:.2f}_c{c:g}', 1000.0, soh, 1.0, c,
                                    tags={'soh': soh, 'c': c}))
    if 'tight' in families:
        for soh in (1.0, 0.8):
            for c in (1.0, 2.0, 3.0):
                for k in (0.9, 1.0, 1.1, 1.25):
                    soc0 = tight_soc_init({}, soh, c, 5000.0, k)
                    out.append(Scenario('tight', f'soh{soh:.2f}_c{c:g}_k{k:g}', 5000.0, soh,
                                        soc0, c, tags={'k': k, 'soh': soh, 'c': c}))
    if 'switch' in families:
        for period in (10.0, 30.0, 120.0):
            out.append(Scenario('switch', f'period{period:g}s', 3000.0, 1.0, 1.0, 2.0,
                                switch_period_s=period, tags={'period_s': period}))
    if 'mismatch' in families:
        for pname, plant in PLANTS.items():
            for c in (1.0, 2.0, 3.0):
                out.append(Scenario('mismatch', f'{pname}_c{c:g}', 2000.0, 1.0, 1.0, c,
                                    plant=plant, tags={'plant': pname, 'c': c}))
    if 'tight_mismatch' in families:
        for pname, plant in PLANTS.items():
            for soh in (1.0, 0.8):
                for c in (1.0, 2.0, 3.0):
                    for k in (1.0, 1.1, 1.25):
                        soc0 = tight_soc_init(plant, soh, c, 5000.0, k)
                        out.append(Scenario('tight_mismatch', f'{pname}_soh{soh:.2f}_c{c:g}_k{k:g}',
                                            5000.0, soh, soc0, c, plant=plant,
                                            tags={'plant': pname, 'k': k, 'soh': soh, 'c': c}))
    if 'degradation' in families:
        for c in (1.0, 2.0, 3.0):
            soc0 = tight_soc_init({}, 0.8, c, 5000.0, k_eol)
            for soh in np.round(np.arange(1.0, 0.8 - 1e-9, -0.025), 3):
                out.append(Scenario('degradation', f'c{c:g}_soh{soh:.3f}', 5000.0, float(soh),
                                    soc0, c, tags={'soh': float(soh), 'c': c, 'soc0': soc0}))
    return out


def make_env(config: dict, sc: Scenario) -> RobotEnv:
    robot = make_robot(dict(config, mission_distance_m=sc.distance_m))
    return RobotEnv(robot=robot, ecm=ECMParams.default(robot), stress=StressParams(),
                    soc_init=sc.soc0, soh_init=sc.soh, complexity=sc.complexity, mode='test',
                    complexity_jitter=config['test_complexity_jitter'])
