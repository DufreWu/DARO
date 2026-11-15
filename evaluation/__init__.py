"""DARO evaluation package: metrics, runners, benchmark and paper outputs.

Environment and scenarios live in envs/, controllers in methods/; the names below are
re-exported for convenience (from evaluation import EVAL_CONFIG, make_baselines, ...).

Modules
-------
  metrics         run_episode, Xu capacity-fade metrics, aggregation
  runner          evaluate_policies, summaries, SAC checkpoint loading
  lifetime        missions/km to end of life (extrapolated from Xu fade)
  scenarios       CLI over the paper scenario suite (envs/scenarios.py)
  benchmark       full benchmark: baselines, Lag-RLS, ablations, references (paper tables)
  paper_results   SOH-estimation accuracy, robustness sweeps, trajectory example
  paper_pdf       paper figures (PDF), LaTeX tables and the compiled results PDF
  distance_sweep, short_term, reward_check   older studies / checks

Run from the repository root:
    python -m evaluation.benchmark --episodes 10 --out logs/benchmark
    python -m evaluation.paper_results soh-accuracy|robustness|trajectory
    python -m evaluation.paper_pdf
    python -m evaluation.scenarios --baselines-only
"""

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR.parent) not in sys.path:      # repository root, so that envs / methods import
    sys.path.insert(0, str(_PKG_DIR.parent))
import envs  # noqa: E402,F401  (adds battery_model to sys.path)

from envs.test_env import (EVAL_CONFIG, TestScenario, test_suite, make_robot,  # noqa: E402
                           episode_max_steps, make_test_env)
from methods import ASMFBaseline, QoSAwareEPD, BatteryAwareEPD, BAEPDSohEstimate, LagRLSEPD, SohEstimator, make_baselines  # noqa: E402
from .metrics import run_episode, aggregate  # noqa: E402
from .runner import (evaluate_policies, print_summary, save_results, load_daro,  # noqa: E402
                     load_rl_policies)
from .lifetime import lifetime_from_results  # noqa: E402
