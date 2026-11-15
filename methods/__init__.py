"""Controllers compared in the paper.

Modules
-------
  baselines          AS-MF, QoS-EPD, BA-EPD, BA-EPD+SOH (rule-based)
  soh_estimator      on-line SOH estimation from BMS voltage / current (shared)
  lag_rls            Lag-RLS, the proposed method (SOC budget priced by a Lagrange
                     multiplier + on-line SOH + RLS-identified power models)
  degradation_aware  DA-Lag-RLS: Lag-RLS with predicted Xu ageing in the objective
  oracle             Lagrangian oracle on the true plant (reference, hindsight lambda)

The SAC agent (sac_agent.py, sac_training.py, hierarchical.py) stays at the top level.
"""

from typing import Dict

import envs  # noqa: F401  (sys.path: repository root + battery_model)
from envs.test_env import make_robot
from .baselines import ASMFBaseline, QoSAwareEPD, BatteryAwareEPD, BAEPDSohEstimate
from .soh_estimator import SohEstimator
from .lag_rls import LagRLSEPD


def make_baselines(config: dict) -> Dict[str, object]:
    """Rule-based baselines and Lag-RLS. RL policies are checkpoints (evaluation.runner)."""
    robot = make_robot(config)
    return {
        'AS-MF':   ASMFBaseline(robot, config),
        'QoS-EPD': QoSAwareEPD(robot, config),
        'BA-EPD':  BatteryAwareEPD(robot, config),
        'BA-EPD+SOH': BAEPDSohEstimate(robot, config),
        'Lag-RLS': LagRLSEPD(robot, config),
    }
