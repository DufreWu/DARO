"""DARO evaluation package: test environment, baselines, metrics, runners.

Modules
-------
  test_env        EVAL_CONFIG, TestScenario grid, make_test_env
  baselines       AS-MF, QoS-EPD (rule-based); RL-SOC-only / RL-SOH-only are
                  SAC checkpoints trained with sac_training.py --obs soc|soh
  metrics         run_episode, Xu capacity-fade metrics, aggregation
  runner          evaluate_policies, summaries, DARO checkpoint loading
  lifetime        missions/km to end of life (extrapolated from Xu fade)
  distance_sweep  results for mission distances 500/1000/2500/5000 m

Run from rl_train/daro_sac:
    python -m evaluation --baselines-only
    python -m evaluation --ckpt ./checkpoints/seed_42
    python -m evaluation.distance_sweep [--ckpt ./checkpoints/seed_42]
"""

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
for _p in (_PKG_DIR.parent, _PKG_DIR.parents[2] / 'battery_model'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from .test_env import (EVAL_CONFIG, TestScenario, test_suite, make_robot,  # noqa: E402
                       episode_max_steps, make_test_env)
from .baselines import ASMFBaseline, QoSAwareEPD, make_baselines  # noqa: E402
from .metrics import run_episode, aggregate  # noqa: E402
from .runner import (evaluate_policies, print_summary, save_results, load_daro,  # noqa: E402
                     load_rl_policies)
from .lifetime import lifetime_from_results  # noqa: E402
