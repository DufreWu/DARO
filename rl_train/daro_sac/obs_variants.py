"""Ablation variants for RL training: observation (--obs) and reward (--reward).

The two are independent; any combination can be trained. Each checkpoint
stores its variants in obs_variant.json, and the evaluation scripts label
(and mask) it from that file.

Observation variants (which battery state the agent sees):

    full  DARO         all observations
    soc   RL-SOC-only  battery block reduced to SOC
    soh   RL-SOH-only  battery block reduced to SOH

Battery block of the observation (robot_env.RobotEnv._get_obs):
    2 SOC, 3 SOH, 4 temperature, 5 terminal voltage, 7 last current.
Temperature, voltage and current are hidden as well in the ablations because
they leak SOC/SOH (OCV(SOC, SOH), R0(SOH), I = P/V). Hidden features are set
to 0 (same network shape as DARO; a constant input is equivalent to dropping
it). Mission features (v, f, distance remaining, throughput, QoS targets) are
always visible.

Reward variants (which objective term is trained on; robot_env.reward_terms):
    full       -E_d - D_d + U_t    (DARO)
    no_stress  -E_d       + U_t    w_stress = 0
    no_energy       - D_d + U_t    w_energy = 0
Only training uses the variant: evaluation always scores with the full reward,
so returns stay comparable across methods.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

BATTERY_IDX = {'soc': 2, 'soh': 3, 'temperature': 4, 'voltage': 5, 'current': 7}
VARIANTS = {          # variant -> battery features kept visible
    'full': tuple(BATTERY_IDX),
    'soc': ('soc',),
    'soh': ('soh',),
}
LABELS = {'full': 'DARO', 'soc': 'RL-SOC-only', 'soh': 'RL-SOH-only'}
VARIANT_FILE = 'obs_variant.json'

REWARD_VARIANTS = {   # variant -> RobotParams weight overrides
    'full': {},
    'no_stress': {'w_stress': 0.0},
    'no_energy': {'w_energy': 0.0},
}
REWARD_SUFFIX = {'full': '', 'no_stress': ' w/o D_d', 'no_energy': ' w/o E_d'}


def obs_mask(variant: str, obs_dim: int) -> Optional[np.ndarray]:
    """Multiplicative 0/1 mask, or None when nothing is hidden."""
    if variant not in VARIANTS:
        raise ValueError(f'unknown obs variant {variant!r}; use one of {list(VARIANTS)}')
    hidden = [i for k, i in BATTERY_IDX.items() if k not in VARIANTS[variant]]
    if not hidden:
        return None
    m = np.ones(obs_dim, dtype=np.float32)
    m[hidden] = 0.0
    return m


def apply_mask(obs, mask: Optional[np.ndarray]):
    return obs if mask is None else np.asarray(obs, dtype=np.float32) * mask


class MaskedPolicy:
    """Wrap an agent so it receives the masked observation (evaluation side)."""

    def __init__(self, agent, variant: str, obs_dim: int):
        self.agent, self.variant = agent, variant
        self.mask = obs_mask(variant, obs_dim)

    def reset(self):
        if hasattr(self.agent, 'reset'):
            self.agent.reset()

    def select_action(self, obs, deterministic: bool = True):
        return self.agent.select_action(apply_mask(obs, self.mask), deterministic=deterministic)

    def __getattr__(self, name):  # alpha, save, ... pass through
        return getattr(self.agent, name)


def apply_reward_variant(robot, reward: str):
    """Set the training reward weights in place (RobotParams)."""
    if reward not in REWARD_VARIANTS:
        raise ValueError(f'unknown reward variant {reward!r}; use one of {list(REWARD_VARIANTS)}')
    for k, v in REWARD_VARIANTS[reward].items():
        setattr(robot, k, v)
    return robot


def label(obs: str = 'full', reward: str = 'full') -> str:
    """Method name, e.g. 'DARO', 'RL-SOC-only', 'DARO w/o D_d', 'RL-SOH-only w/o E_d'."""
    return LABELS[obs] + REWARD_SUFFIX[reward]


def run_subdir(obs: str = 'full', reward: str = 'full') -> str:
    """Output sub-folder below checkpoints/ and logs/ ('' for DARO).

    rl_soc, rl_soh, rw_no_stress, rw_no_energy, rl_soc_rw_no_stress, ...
    """
    parts = ([f'rl_{obs}'] if obs != 'full' else []) + \
            ([f'rw_{reward}'] if reward != 'full' else [])
    return '_'.join(parts)


def save_variant(ckpt_dir: str, variant: str, reward: str = 'full'):
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, VARIANT_FILE), 'w') as f:
        json.dump({'obs_variant': variant, 'visible_battery': list(VARIANTS[variant]),
                   'reward_variant': reward,
                   'reward_overrides': REWARD_VARIANTS[reward]}, f, indent=2)


def read_variants(ckpt_dir: str) -> dict:
    """{'obs': ..., 'reward': ...}; missing file / keys default to 'full' (DARO)."""
    path = os.path.join(ckpt_dir, VARIANT_FILE)
    d = {}
    if os.path.isfile(path):
        with open(path) as f:
            d = json.load(f)
    return {'obs': d.get('obs_variant', 'full'), 'reward': d.get('reward_variant', 'full')}


def load_variant(ckpt_dir: str) -> str:
    """Observation variant stored with a checkpoint ('full' if none)."""
    return read_variants(ckpt_dir)['obs']
