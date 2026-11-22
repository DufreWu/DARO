"""Hierarchical control: slow RL speed set-point + fast (1 s) QoS layer.

    slow layer (SAC)  every decision_s seconds, or at once when the workload estimate
                      jumps by more than trigger_rel: v_ref in [v_min, v_max]
    fast layer (1 s)  c_hat from the observed throughput (same estimator as QoS-EPD),
                      v = min(v_ref, fastest QoS-feasible speed at c_hat),
                      f = minimum frequency meeting margin * QoS at (v, c_hat)

The physics always steps at control_dt (1 s); workload changes are handled by the fast
layer within one step, QoS feasibility is enforced by construction, and the agent only
chooses the speed / energy / time trade-off. One agent transition spans up to
decision_steps env steps (rewards summed); the discount is per decision
(sac_training.control_gamma; early, event-triggered decisions use the same gamma,
a small approximation at the multi-thousand-second horizon).

The agent observes the raw observation plus [c_hat, v_cap / v_max] and, with
charge_feature, the charge margin m = (energy above the SOC reserve, nominal OCV(SOC, SOH)
x capacity) / (loss_factor x energy for the remaining distance at the energy-optimal
QoS-feasible speed at c_hat, nominal power model), clipped to [0, 5]. m < 1: even
energy-optimal driving ends below the reserve; m >> 1: charge is not binding. Same
observed quantities (SOC, SOH, distance) and nominal model as BA-EPD. The layer works on
a frozen copy of the NOMINAL robot parameters (model-mismatch episodes change the env's
own parameters to the true plant, which the controller must not see).

Training-only reward shaping (shaping_w > 0): potential-based,
    r' = r + gamma * Phi(s') - Phi(s),   Phi(s) = -shaping_w * max(0, 1 - m_true(s)),
Phi = 0 at terminal states, with m_true the charge margin from the TRUE SOC, SOH,
remaining distance and workload level (the observation is noisy in training). It turns
the terminal reserve penalty into a per-decision signal and leaves the optimal policy
unchanged (Ng et al. 1999). Evaluation always uses the unshaped reward.

Used on both sides:
  training    HierEnv(env, layer)        env.step(v_ref); stores augmented obs
  evaluation  HierPolicy(agent, layer)   policy.select_action(raw obs) -> (v, f)
Checkpoints store the setting in hierarchical.json (read by load_hier).
"""

from __future__ import annotations

import copy
import json
import os
from typing import Optional

import numpy as np

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]   # repository root: envs / methods / evaluation
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from envs.robot_env import (motor_power, non_motor_power, throughput_freq_required,
                            throughput_max_speed, _voc, freq_action)

HIER_FILE = 'hierarchical.json'
_SUM_KEYS = ('energy_j', 'stress_proxy', 'E_d', 'D_d', 'U_t', 'T_d', 'Q_d', 'R_end')


class FastQoSLayer:
    """1 s layer: workload estimate + QoS-feasible (v, f) for a speed set-point."""

    LOSS_FACTOR = 1.05   # cell losses not in the load-side model (as robot_env.tight_soc_init)

    def __init__(self, robot, config: dict, decision_steps: int, trigger_rel: float,
                 charge_feature: bool = False):
        from methods import QoSAwareEPD   # same complexity estimator
        self.p = robot
        self.charge_feature = bool(charge_feature)
        self.n_aug = 3 if self.charge_feature else 2
        self.est = QoSAwareEPD(robot, config)
        self.margin = float(config['baseline_qos_margin'])
        self.decision_steps = max(1, int(decision_steps))
        self.trigger_rel = float(trigger_rel)
        self.reset()

    def reset(self):
        self.est.reset()
        self.q_ref = float(max(self.p.qos_ref))
        self.c_dec = None   # workload estimate at the last decision

    def observe(self, obs):
        """Update the workload estimate from a (raw) observation; once per env step."""
        n = self.p.n_apps
        self.q_ref = float(np.max(obs[8 + n:8 + 2 * n]))
        self.est._update_complexity(float(obs[0]), float(obs[1]), float(np.min(obs[8:8 + n])))

    @property
    def c_hat(self) -> float:
        return 1.0 if self.est.c_hat is None else float(self.est.c_hat)

    def v_cap(self) -> float:
        p = self.p
        v = throughput_max_speed(p.f_max, self.q_ref * self.margin, self.est._c_safe())
        return float(np.clip(v, p.v_min, p.v_max))

    def act(self, v_ref: float) -> np.ndarray:
        p = self.p
        v = float(np.clip(min(float(v_ref), self.v_cap()), p.v_min, p.v_max))
        f = throughput_freq_required(v, self.q_ref * self.margin, self.est._c_safe())
        return freq_action(v, f)                 # [speed, DVFS level index]

    def margin_at(self, soc: float, soh: float, d_rem: float, c: float) -> float:
        """Charge margin (nominal model) for SOC, SOH, remaining distance, workload c."""
        p = self.p
        soh = float(np.clip(soh, 1e-3, 1.0))
        if d_rem < 1.0:
            return 5.0
        if soc <= p.soc_reserve:
            return 0.0
        s = np.linspace(p.soc_reserve, soc, 25)
        e_av = p.Q_n * soh * float(np.trapezoid([_voc(x, soh) for x in s], s))
        v, f = self.est.optimise(self.q_ref, c * (1.0 + self.est.jitter))
        e_m = float(motor_power(v, 0.0, p) + non_motor_power(v, f, c, p)) / v
        return float(np.clip(e_av / (self.LOSS_FACTOR * e_m * d_rem), 0.0, 5.0))

    def charge_margin(self, obs) -> float:
        """Charge margin from the observation and the workload estimate."""
        return self.margin_at(float(obs[2]), float(obs[3]), max(0.0, float(obs[6])), self.c_hat)

    def mark_decision(self):
        self.c_dec = self.c_hat

    def triggered(self) -> bool:
        """Workload estimate moved by more than trigger_rel since the last decision."""
        return (self.trigger_rel > 0 and self.c_dec is not None
                and abs(self.c_hat - self.c_dec) > self.trigger_rel * self.c_dec)

    def augment(self, obs) -> np.ndarray:
        extra = [self.c_hat, self.v_cap() / self.p.v_max]
        if self.charge_feature:
            extra.append(self.charge_margin(obs))
        return np.concatenate([np.asarray(obs, dtype=np.float32), extra]).astype(np.float32)


class HierEnv:
    """Training-side wrapper: action = v_ref (1-d), obs = [obs, c_hat, v_cap / v_max]."""

    def __init__(self, env, layer: FastQoSLayer, shaping_w: float = 0.0, gamma: float = 1.0):
        self.env, self.layer = env, layer
        self.shaping_w, self.gamma = float(shaping_w), float(gamma)
        self._phi = 0.0
        p = env.robot
        self.obs_dim = env.obs_dim + layer.n_aug
        self.act_dim = 1
        self.action_low = np.array([p.v_min], dtype=np.float32)
        self.action_high = np.array([p.v_max], dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.layer.reset()
        self.layer.observe(obs)
        self._phi = self._potential()
        return self.layer.augment(obs), info

    def _potential(self) -> float:
        if self.shaping_w <= 0:
            return 0.0
        e = self.env
        m = self.layer.margin_at(float(e.battery.soc), float(e.battery.soh),
                              max(0.0, e.robot.mission_distance_m - e.distance_m),
                              float(e.complexity))
        return -self.shaping_w * max(0.0, 1.0 - m)

    def step(self, action):
        """Hold v_ref for up to decision_steps env steps (fast layer every step); stop early
        on a workload jump. Returns the summed reward; info sums the additive terms,
        qos_violation = max over the held steps."""
        v_ref = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        self.layer.mark_decision()
        total, sums, viol = 0.0, {k: 0.0 for k in _SUM_KEYS}, 0.0
        for _ in range(self.layer.decision_steps):
            obs, reward, term, trunc, info = self.env.step(self.layer.act(v_ref))
            self.layer.observe(obs)
            total += reward
            for k in _SUM_KEYS:
                sums[k] += float(info.get(k, 0.0))
            viol = max(viol, float(info['qos_violation']))
            if term or trunc or self.layer.triggered():
                break
        if self.shaping_w > 0:
            phi_next = 0.0 if term else self._potential()
            total += self.gamma * phi_next - self._phi
            self._phi = phi_next
        info = dict(info, **sums, qos_violation=viol)
        return self.layer.augment(obs), total, term, trunc, info

    def obs_normalization(self, max_distance_m: Optional[float] = None):
        off, scl = self.env.obs_normalization(max_distance_m)
        off_x, scl_x = [0.0, 0.0, 1.0][:self.layer.n_aug], [3.0, 1.0, 1.0][:self.layer.n_aug]
        return (np.concatenate([off, off_x]).astype(np.float32),
                np.concatenate([scl, scl_x]).astype(np.float32))

    def __getattr__(self, name):   # robot, ecm_params, battery, ... pass through
        return getattr(self.env, name)


class HierPolicy:
    """Evaluation-side wrapper: raw obs -> (v, f) every step; the agent (which sees the
    augmented obs) is queried every decision_steps steps or on a workload jump, exactly
    as in HierEnv."""

    def __init__(self, agent, layer: FastQoSLayer):
        self.agent, self.layer = agent, layer
        self._k, self._v_ref = 0, None

    def reset(self):
        self.layer.reset()
        self._k, self._v_ref = 0, None
        if hasattr(self.agent, 'reset'):
            self.agent.reset()

    def select_action(self, obs, deterministic: bool = True):
        self.layer.observe(obs)
        if self._v_ref is None or self._k >= self.layer.decision_steps or self.layer.triggered():
            a = self.agent.select_action(self.layer.augment(obs), deterministic=deterministic)
            self._v_ref = float(np.asarray(a).reshape(-1)[0])
            self.layer.mark_decision()
            self._k = 0
        self._k += 1
        return self.layer.act(self._v_ref)

    def __getattr__(self, name):
        return getattr(self.agent, name)


# --- settings / checkpoints ---------------------------------------------------

def hier_setting(config: dict) -> Optional[dict]:
    """From the training config (config['hierarchical'] True/False)."""
    if not config.get('hierarchical', False):
        return None
    return {'decision_s': float(config['hier_decision_s']),
            'trigger_rel': float(config['hier_trigger_rel']),
            'charge_feature': bool(config.get('hier_charge_feature', True)),
            'shaping_w': float(config.get('hier_shaping_w', 0.0))}


def decision_steps(setting: Optional[dict], control_dt: float = 1.0) -> int:
    if not setting:
        return 1
    return max(1, int(round(setting['decision_s'] / control_dt)))


def build_layer(setting: dict, robot, control_dt: float = 1.0) -> FastQoSLayer:
    """Fast layer on a frozen copy of the NOMINAL robot parameters."""
    from evaluation import EVAL_CONFIG
    from envs.robot_env import PLANT_KEYS, RobotParams
    cfg = dict(EVAL_CONFIG, control_dt=control_dt)
    robot = copy.copy(robot)
    nominal = RobotParams()
    for k in PLANT_KEYS:
        setattr(robot, k, getattr(nominal, k))
    return FastQoSLayer(robot, cfg, decision_steps(setting, control_dt), setting['trigger_rel'],
                        setting.get('charge_feature', False))   # old checkpoints: no feature


def save_hier(ckpt_dir: str, setting: Optional[dict]):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, HIER_FILE)
    if setting:
        with open(path, 'w') as f:
            json.dump(setting, f, indent=2)
    elif os.path.exists(path):
        os.remove(path)


def load_hier(ckpt_dir: str) -> Optional[dict]:
    path = os.path.join(ckpt_dir, HIER_FILE)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)
