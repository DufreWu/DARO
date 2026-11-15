"""Simulation environment of the battery-powered mobile robot.

Modules
-------
  robot_env   RobotEnv (gymnasium), RobotParams, 2RC ECM battery with OCV(SOC, SOH) and
              R0(SOH), motor / compute power models, BMS observation model, reward,
              tight_soc_init; lipo_fit_parameters.json holds the fitted LiPo ECM
  test_env    EVAL_CONFIG, make_robot (config -> RobotParams incl. battery realism),
              SOH x SOC x complexity test grid, make_test_env
  scenarios   paper scenario suite (nominal / tight / switch / mismatch / tight_mismatch /
              degradation), true-plant overrides PLANTS, make_env
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]          # envs / methods / evaluation live here
RL_DIR = REPO_ROOT / 'rl_train' / 'daro_sac'               # SAC code and checkpoints
for _p in (REPO_ROOT, REPO_ROOT / 'battery_model'):        # Xu ageing model (battery_model.py)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
