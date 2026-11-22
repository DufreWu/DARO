"""Informed reference policy (upper reference for the headroom of learning).

Lagrangian speed / frequency schedule with the TRUE plant:

    v_t = argmin_v  [ (1 + lam) * (w_E / L) * P_true(v, f_q(v, c_t), c_t) + w_T / T_ref ] / v
    f_t = f_q(v_t, c_t) = minimum frequency meeting margin * QoS at complexity c_t (1 + jitter)

lam >= 0 is the shadow price of battery energy (0 = unconstrained time/energy optimum,
large = minimum energy per metre), constant over the mission, which is the optimal
structure for this separable cost under a single energy budget. The policy knows the
true plant (robot_env.PLANT_KEYS of the episode) and the current NOMINAL complexity
level (incl. mid-mission switches), but not the per-step jitter; it uses the same QoS
margin and jitter allowance as the rule-based baselines. lam is chosen in hindsight
per episode (best return over LAM_GRID on the same seed), i.e. it also "knows" the
battery losses and the future workload mix through that choice.

observe_c=True ('Oracle-obs'): the complexity is NOT read from the env but estimated
from the observed throughput exactly as the rule-based baselines do (EMA of the
inverted throughput model), so the gap to 'Oracle' is the value of knowing the hidden
workload level, and Oracle-obs - planner is what is reachable from the observations
(given the true plant and the hindsight lam).

It is a strong reference, not the true optimum (no QoS degradation under a budget, no
acceleration planning, a grid over lam): use it to bound how much better than a
rule-based planner a learned scheduler can get on a scenario, as (oracle - planner).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from envs.robot_env import (RobotEnv, motor_power, non_motor_power, throughput_freq_required,
                       throughput_complexity, quantize_freq, freq_action)

LAM_GRID = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 1e3)


class LagrangianOracle:
    name = 'Oracle'

    def __init__(self, env: RobotEnv, lam: float, config: dict, n_grid: int = 181,
                 observe_c: bool = False):
        self.env, self.lam, self.observe_c = env, float(lam), bool(observe_c)
        self.ema = float(config['baseline_c_ema'])
        self.c_hat: Optional[float] = None
        from .baselines import qos_allowance
        self.margin, self.jitter = qos_allowance(config)
        p = env.robot
        self.v = np.linspace(max(p.v_min, 0.05), p.v_max, n_grid)
        self._cache = {}

    def reset(self):
        self._cache = {}   # plant may change between episodes (reset options)
        self.c_hat = None

    def _estimate_c(self, obs) -> float:
        n = self.env.robot.n_apps
        v, f, thr = float(obs[0]), float(obs[1]), float(np.min(obs[8:8 + n]))
        c = throughput_complexity(v, f, thr) if v > 0.1 else None
        if c is not None:
            c = float(np.clip(c, 0.1, 10.0))
            self.c_hat = c if self.c_hat is None else (1 - self.ema) * self.c_hat + self.ema * c
        return 1.0 if self.c_hat is None else self.c_hat

    def _schedule(self, c: float, q_ref: float):
        key = (round(c, 6), round(q_ref, 6))
        if key not in self._cache:
            p = self.env.robot   # true plant of this episode
            c_safe = c * (1.0 + self.jitter)
            f_req = throughput_freq_required(self.v, q_ref * self.margin, c_safe)
            ok = f_req <= p.f_max + 1e-9
            f = quantize_freq(f_req)
            power = motor_power(self.v, 0.0, p) + non_motor_power(self.v, f, c, p)
            L, T_ref = p.mission_distance_m, p.mission_distance_m / p.v_ref_mps
            cost = ((1.0 + self.lam) * p.w_energy / L * power + p.w_time / T_ref) / self.v
            if ok.any():
                k = int(np.argmin(np.where(ok, cost, np.inf)))
                self._cache[key] = (float(self.v[k]), float(f[k]))
            else:   # QoS unreachable: f_max, cheapest speed
                k = int(np.argmin(cost))
                self._cache[key] = (float(self.v[k]), float(p.f_max))
        return self._cache[key]

    def select_action(self, obs, deterministic=True):
        q_ref = float(np.max(self.env.robot.qos_ref))
        c = self._estimate_c(obs) if self.observe_c else float(self.env.complexity)
        if self.observe_c:   # continuous estimate: no cache
            self._cache = {}
        return freq_action(*self._schedule(c, q_ref))


def best_in_hindsight(make_env_fn, config: dict, seed: int, options: Optional[dict] = None,
                      lams: Iterable[float] = LAM_GRID, charge_c_rate: float = 1.0,
                      observe_c: bool = False) -> dict:
    """Run the oracle for every lam on the same seed; return the best episode's metrics
    (plus 'lam'). make_env_fn() must return a fresh env for the scenario."""
    from evaluation.metrics import run_episode   # lazy: evaluation imports methods
    best = None
    for lam in lams:
        env = make_env_fn()
        m = run_episode(env, LagrangianOracle(env, lam, config, observe_c=observe_c), seed=seed,
                        charge_c_rate=charge_c_rate, options=options)
        if best is None or m['return'] > best['return']:
            best = dict(m, lam=float(lam))
    return best
