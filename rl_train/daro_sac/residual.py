"""Residual RL: the agent corrects a model-based base controller.

    a_base = base.select_action(obs)                      (v_b, f_b), nominal model
    v = clip(v_b * (1 + rho_v * a_res[0]), v_min, v_max)
    f = clip(f_b * (1 + rho_f * a_res[1]), f_min, f_max),   a_res in [-1, 1]^2

a_res = 0 reproduces the base controller exactly, so training starts from a policy
that already completes missions and meets QoS; the agent only learns where the
base controller's nominal power / complexity model is wrong and which way to
correct it (model mismatch, workload switching, charge budget).

The agent observes the raw observation plus the base action (v_b, f_b). The base
controller always receives the raw (unmasked) observation; obs ablations
(obs_variants.py) mask only the agent's input.

The residual is re-chosen every `hold_steps` control steps (config residual_hold_s,
default 10 s) and held in between, while the base controller still runs every step.
Fewer, longer decisions make the small per-step effect of a correction visible to
the critic (one agent transition = hold_steps env steps, rewards summed; the
discount is per decision, sac_training.control_gamma). The actor's mean head is
zero-initialised (zero_init_actor) so the initial policy is the base controller.

Used on both sides:
  training    ResidualEnv(env, ctrl)       env.step(a_res); stores augmented obs
  evaluation  ResidualPolicy(agent, ctrl)  policy.select_action(raw obs) -> (v, f)
Each calls the (stateful) base controller exactly once per control step.

Mode 'model' (residual_mode='model', base 'ba_epd' only): instead of correcting the
action, the agent corrects the base controller's NOMINAL energy model,
    kappa = (1 + rho_k) ** a_res[0],   a_res in [-1, 1]   (kappa in [1/(1+rho_k), 1+rho_k])
which multiplies BA-EPD's energy-per-metre in its charge-budget plan (fast / optimal /
budget). kappa = 1 (a_res = 0) is BA-EPD. The action (v, f) always comes from BA-EPD, so
QoS handling is untouched. The agent observes the raw observation plus
[a_prev, base mode (fast 1 / optimal 0 / budget -1), EMA of measured / nominal power].
kappa is set before the base runs, and hold_s = BA-EPD's replan period (10 s) makes every
decision coincide with a re-plan.

Checkpoints store the setting in residual.json (read by load_residual).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from robot_env import motor_power, non_motor_power

RESIDUAL_FILE = 'residual.json'
BASES = ('ba_epd', 'qos_epd')
BASE_LABEL = {'ba_epd': 'BA-EPD', 'qos_epd': 'QoS-EPD'}


def make_base(name: str, control_dt: float = 1.0):
    """Base controller on the NOMINAL model (evaluation.EVAL_CONFIG calibration)."""
    from evaluation import EVAL_CONFIG, make_robot
    from evaluation.baselines import BatteryAwareEPD, QoSAwareEPD
    cfg = dict(EVAL_CONFIG, control_dt=control_dt)
    cls = {'ba_epd': BatteryAwareEPD, 'qos_epd': QoSAwareEPD}[name]
    return cls(make_robot(cfg), cfg)


class ResidualController:
    mode = 'action'
    n_act, n_aug = 2, 2

    def __init__(self, base, robot, rho_v: float, rho_f: float, hold_steps: int = 1):
        self.base = base
        self.hold_steps = max(1, int(hold_steps))
        self.rho = np.array([rho_v, rho_f], dtype=np.float32)
        self.low = np.array([robot.v_min, robot.f_min], dtype=np.float32)
        self.high = np.array([robot.v_max, robot.f_max], dtype=np.float32)

    def reset(self):
        self.base.reset()

    def base_action(self, raw_obs) -> np.ndarray:
        return np.asarray(self.base.select_action(raw_obs), dtype=np.float32)

    def combine(self, a_base, a_res) -> np.ndarray:
        a_res = np.clip(np.asarray(a_res, dtype=np.float32), -1.0, 1.0)
        return np.clip(a_base * (1.0 + self.rho * a_res), self.low, self.high).astype(np.float32)

    @staticmethod
    def augment(obs, a_base) -> np.ndarray:
        return np.concatenate([np.asarray(obs, dtype=np.float32), a_base]).astype(np.float32)


class ModelResidualController:
    """'model' mode: the residual scales the base's nominal energy model (see module doc)."""
    mode = 'model'
    n_act, n_aug = 1, 3
    MODE_CODE = {'fast': 1.0, 'optimal': 0.0, 'budget': -1.0}

    def __init__(self, base, robot, rho_k: float, hold_steps: int = 1, power_ema: float = 0.1):
        if not hasattr(base, 'energy_scale'):
            raise ValueError("residual_mode='model' needs a base with energy_scale (ba_epd)")
        self.base = base
        self.hold_steps = max(1, int(hold_steps))
        self.log_k = float(np.log1p(rho_k))
        self.ema = float(power_ema)
        self.reset()

    def reset(self):
        self.base.reset()
        self.base.energy_scale = 1.0
        self._a_prev, self._ratio = 0.0, 1.0

    def set_residual(self, a_res):
        a = float(np.clip(np.asarray(a_res, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        self._a_prev = a
        self.base.energy_scale = float(np.exp(a * self.log_k))

    def act(self, raw_obs) -> np.ndarray:
        """Base action under the current kappa (call exactly once per control step)."""
        self._update_ratio(raw_obs)
        return np.asarray(self.base.select_action(raw_obs), dtype=np.float32)

    def _update_ratio(self, obs):
        # Measured load power V * I vs the base's nominal model at the observed (v, f)
        # and its complexity estimate: identifies the plant mismatch.
        v, f = float(obs[0]), float(obs[1])
        if v < 0.3:
            return
        p = self.base.p
        c = 1.0 if self.base.c_hat is None else float(self.base.c_hat)
        nom = float(motor_power(v, 0.0, p) + non_motor_power(v, f, c, p))
        meas = float(obs[5]) * float(obs[7])
        if nom > 1e-6 and meas > 0:
            r = float(np.clip(meas / nom, 0.0, 3.0))
            self._ratio = (1 - self.ema) * self._ratio + self.ema * r

    def augment(self, obs) -> np.ndarray:
        mode = self.MODE_CODE.get(getattr(self.base, '_mode', 'optimal'), 0.0)
        return np.concatenate([np.asarray(obs, dtype=np.float32),
                               [self._a_prev, mode, self._ratio]]).astype(np.float32)


def hold_steps(setting: Optional[dict], control_dt: float = 1.0) -> int:
    if not setting:
        return 1
    return max(1, int(round(float(setting.get('hold_s', control_dt)) / control_dt)))


def build_controller(setting: dict, robot, control_dt: float = 1.0):
    if setting.get('mode', 'action') == 'model':
        return ModelResidualController(make_base(setting['base'], control_dt), robot,
                                       setting['rho_k'], hold_steps(setting, control_dt))
    return ResidualController(make_base(setting['base'], control_dt), robot,
                              setting['rho_v'], setting['rho_f'],
                              hold_steps(setting, control_dt))


def zero_init_actor(agent):
    """Mean head = 0 -> deterministic residual 0 (= base controller) at start."""
    import torch
    with torch.no_grad():
        for actor in (agent.actor, getattr(agent.actor, 'target', None)):
            if actor is not None:
                actor.mean_head.weight.zero_()
                actor.mean_head.bias.zero_()


_SUM_KEYS = ('energy_j', 'stress_proxy', 'E_d', 'D_d', 'U_t', 'T_d', 'Q_d', 'R_end')


class ResidualEnv:
    """Training-side wrapper: action = residual in [-1, 1]^2, obs = [obs, v_b, f_b]."""

    def __init__(self, env, ctrl):
        self.env, self.ctrl = env, ctrl
        self.obs_dim = env.obs_dim + ctrl.n_aug
        self.act_dim = ctrl.n_act
        self.action_low = -np.ones(ctrl.n_act, dtype=np.float32)
        self.action_high = np.ones(ctrl.n_act, dtype=np.float32)
        self._a_base = self._raw = None

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.ctrl.reset()
        if self.ctrl.mode == 'model':
            self._raw = obs
            return self.ctrl.augment(obs), info
        self._a_base = self.ctrl.base_action(obs)
        return self.ctrl.augment(obs, self._a_base), info

    def _step_model(self, a_res):
        self.ctrl.set_residual(a_res)
        total, sums, viol = 0.0, {k: 0.0 for k in _SUM_KEYS}, 0.0
        for _ in range(self.ctrl.hold_steps):
            obs, reward, term, trunc, info = self.env.step(self.ctrl.act(self._raw))
            self._raw = obs
            total += reward
            for k in _SUM_KEYS:
                sums[k] += float(info.get(k, 0.0))
            viol = max(viol, float(info['qos_violation']))
            if term or trunc:
                break
        info = dict(info, **sums, qos_violation=viol)
        return self.ctrl.augment(obs), total, term, trunc, info

    def step(self, a_res):
        """Hold a_res for hold_steps env steps (base action recomputed every step).
        Returns the summed reward; info sums the additive terms, qos_violation = max."""
        if self.ctrl.mode == 'model':
            return self._step_model(a_res)
        total, sums, viol = 0.0, {k: 0.0 for k in _SUM_KEYS}, 0.0
        for _ in range(self.ctrl.hold_steps):
            action = self.ctrl.combine(self._a_base, a_res)
            obs, reward, term, trunc, info = self.env.step(action)
            total += reward
            for k in _SUM_KEYS:
                sums[k] += float(info.get(k, 0.0))
            viol = max(viol, float(info['qos_violation']))
            done = term or trunc
            if not done:
                self._a_base = self.ctrl.base_action(obs)
            if done:
                break
        info = dict(info, **sums, qos_violation=viol)
        return self.ctrl.augment(obs, self._a_base), total, term, trunc, info

    def obs_normalization(self, max_distance_m: Optional[float] = None):
        off, scl = self.env.obs_normalization(max_distance_m)
        p = self.env.robot
        if self.ctrl.mode == 'model':
            return (np.concatenate([off, [0.0, 0.0, 1.0]]).astype(np.float32),
                    np.concatenate([scl, [1.0, 1.0, 0.5]]).astype(np.float32))
        return (np.concatenate([off, [0.0, 0.0]]).astype(np.float32),
                np.concatenate([scl, [p.v_max, p.f_max]]).astype(np.float32))

    def __getattr__(self, name):   # robot, ecm_params, battery, ... pass through
        return getattr(self.env, name)


class ResidualPolicy:
    """Evaluation-side wrapper: raw obs -> (v, f). `agent` sees the augmented obs
    (wrap it in obs_variants.MaskedPolicy first for the obs ablations)."""

    def __init__(self, agent, ctrl):
        self.agent, self.ctrl = agent, ctrl
        self._k, self._a_res = 0, np.zeros(ctrl.n_act, dtype=np.float32)

    def reset(self):
        self.ctrl.reset()
        self._k, self._a_res = 0, np.zeros(self.ctrl.n_act, dtype=np.float32)
        if hasattr(self.agent, 'reset'):
            self.agent.reset()

    def select_action(self, obs, deterministic: bool = True):
        if self.ctrl.mode == 'model':
            if self._k % self.ctrl.hold_steps == 0:
                self.ctrl.set_residual(self.agent.select_action(self.ctrl.augment(obs),
                                                                deterministic=deterministic))
            self._k += 1
            return self.ctrl.act(obs)
        a_base = self.ctrl.base_action(obs)
        if self._k % self.ctrl.hold_steps == 0:
            self._a_res = self.agent.select_action(self.ctrl.augment(obs, a_base),
                                                   deterministic=deterministic)
        self._k += 1
        return self.ctrl.combine(a_base, self._a_res)

    def __getattr__(self, name):
        return getattr(self.agent, name)


def save_residual(ckpt_dir: str, setting: Optional[dict]):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, RESIDUAL_FILE)
    if setting:
        with open(path, 'w') as f:
            json.dump(setting, f, indent=2)
    elif os.path.exists(path):
        os.remove(path)


def load_residual(ckpt_dir: str) -> Optional[dict]:
    """Residual setting stored with a checkpoint, or None (end-to-end policy)."""
    path = os.path.join(ckpt_dir, RESIDUAL_FILE)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def residual_setting(config: dict) -> Optional[dict]:
    """From the training config (config['residual'] = base name or None)."""
    base = config.get('residual')
    if not base:
        return None
    if base not in BASES:
        raise ValueError(f'unknown residual base {base!r}; use one of {BASES}')
    mode = config.get('residual_mode', 'action')
    hold = float(config.get('residual_hold_s', config.get('control_dt', 1.0)))
    if mode == 'model':
        return {'base': base, 'mode': 'model', 'rho_k': float(config['residual_rho_k']),
                'hold_s': hold}
    if mode != 'action':
        raise ValueError(f"unknown residual_mode {mode!r}; use 'action' or 'model'")
    return {'base': base, 'rho_v': float(config['residual_rho_v']),
            'rho_f': float(config['residual_rho_f']), 'hold_s': hold}
