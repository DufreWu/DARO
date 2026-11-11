"""
sac_training.py
---------------
Training pipeline for the DARO SAC agent on robot_env.RobotEnv.

Control frequency
-----------------
The agent acts at ``control_dt`` (default 1 s, i.e. 1 Hz). This is set by the
robot/mission time scales, not by the battery ECM:

  * ECM RC time constants from lipo_fit_parameters.json are tau1 ~ 0.4-3 ms
    and tau2 ~ 4-140 ms (EIS high-frequency arcs). They settle within one
    step at any practical control rate; RobotEnv uses exact exponential RC
    updates, so the battery stays accurate (quasi-static) at dt = 1 s.
  * Speed ramp v_min -> v_max takes (v_max - v_min) / a_max = 6 s,
    thermal tau = 500 s, a 500 m mission lasts 100-1000 s.
  * Higher rates only lengthen the episode: gamma is derived from the desired
    planning horizon (``horizon_s``) so the effective horizon in seconds is
    unchanged if ``control_dt`` is changed.

Episodes end on mission completion, battery limits, or truncation. Only true
terminations are stored as ``done`` in the replay buffer (time-limit
truncation bootstraps).

Usage
-----
    python sac_training.py                         # full training run
    python sac_training.py --control-dt 0.5        # 2 Hz control
    python sac_training.py --seed 7                # -> checkpoints/seed_7, logs/seed_7
    python sac_training.py --seeds 0 1 2 3 4       # one run per seed

Testing lives in the evaluation/ package (run from this directory):
    python -m evaluation --ckpt ./checkpoints/seed_42
    python -m evaluation.distance_sweep --ckpt ./checkpoints/seed_42
"""

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List

import numpy as np

from robot_env import RobotEnv, RobotParams, ECMParams, StressParams
from sac_agent import SACAgent
from evaluation import make_robot

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Control timing
    "control_dt":       1.0,            # [s] agent decision period (1 Hz)
    "horizon_s":        100.0,          # [s] effective planning horizon -> gamma
    "max_steps_margin": 1.2,            # max_steps = margin * mission / (v_min * dt)
    # Training scale
    "total_steps":      500_000,
    "max_episodes":     10_000,
    "batch_size":       64,
    "buffer_capacity":  100_000,
    # SAC hyperparameters
    "lr":               3e-4,
    "polyak":           0.995,
    "alpha_init":       0.2,
    "auto_alpha":       True,
    "hidden_sizes":     [128, 64],
    # Curriculum / robustness (uniform draw each episode)
    "soh_range":        [0.80, 1.00],
    "soc_range":        [0.20, 1.00],   # covers the low-SOC test scenarios
    "complexity_levels": [1.0, 2.0, 3.0],  # low / medium / high
    # Mission distance drawn per episode; obs[6] (distance remaining) must
    # cover the tested range, otherwise DARO extrapolates at long missions.
    "mission_distances_m": [500.0, 1000.0, 2500.0, 5000.0],
    # Warm-up before first update
    "warmup_steps":     5_000,
    # Periodic evaluation during training (evaluation package, DARO only;
    # 0 disables). Uses the 500 m test grid with few seeds.
    "eval_every_ep":    200,
    "eval_episodes":    2,              # seeds per scenario
    # Checkpointing
    "ckpt_every_ep":    500,
    "ckpt_dir":         "./checkpoints",
    # Logging
    "log_dir":          "./logs",
    "seed":             42,             # training seed (CLI: --seed / --seeds)
    "torch_deterministic": False,       # deterministic cuDNN/torch kernels (slower)
    "seed_subdir":      True,           # write to <ckpt_dir>/seed_<seed>, <log_dir>/seed_<seed>
}


def control_gamma(config: dict) -> float:
    """Discount so that the effective horizon is horizon_s seconds."""
    return float(math.exp(-config['control_dt'] / config['horizon_s']))


# ---------------------------------------------------------------------------
# Metric aggregator
# ---------------------------------------------------------------------------

class EpisodeMetrics:
    """Accumulate per-step metrics across an episode."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.rewards: List[float] = []
        self.energy_j: List[float] = []
        self.stress: List[float] = []
        self.qos_violation: List[float] = []
        self.terms = {'E_d': 0.0, 'D_d': 0.0, 'U_t': 0.0}
        self.thr_violations = 0
        self.steps = 0
        self.last_info: dict = {}

    def record(self, reward, info):
        self.rewards.append(reward)
        self.energy_j.append(info['energy_j'])
        self.stress.append(info['stress_proxy'])
        self.qos_violation.append(info['qos_violation'])
        for k in self.terms:
            self.terms[k] += info.get(k, 0.0)
        if info['qos_violation'] > 0.0:
            self.thr_violations += 1
        self.steps += 1
        self.last_info = info

    def summarise(self) -> dict:
        n = max(self.steps, 1)
        li = self.last_info
        return {
            'return':          float(np.sum(self.rewards)),
            'sum_E_d':         self.terms['E_d'],
            'sum_D_d':         self.terms['D_d'],
            'sum_U_t':         self.terms['U_t'],
            'total_energy_j':  float(np.sum(self.energy_j)),
            'total_stress':    float(np.sum(self.stress)),
            'mean_qos_viol':   float(np.mean(self.qos_violation)) if self.steps else 0.0,
            'thr_viol_pct':    100.0 * self.thr_violations / n,
            'mission_time_s':  float(li.get('t_s', 0.0)),
            'distance_m':      float(li.get('distance_m', 0.0)),
            'final_soc':       float(li.get('soc', float('nan'))),
            'completed':       float(bool(li.get('completed', False))),
            'steps':           self.steps,
        }


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False):
    """Seed Python, NumPy (replay sampling) and torch (network init, SAC noise)."""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


class SeedStreams:
    """Independent RNG streams derived from one training seed.

    Separate streams keep e.g. the curriculum sequence unchanged when the
    number of replay samples or warm-up steps changes.
    """
    def __init__(self, seed: int):
        ss = np.random.SeedSequence(seed)
        cur, warm, env_w, env_t = ss.spawn(4)
        self.curriculum = np.random.default_rng(cur)     # SOH/SOC/complexity/distance
        self.warmup_actions = np.random.default_rng(warm)
        self.warmup_env_seed = int(env_w.generate_state(1)[0])
        self.train_env_seed = int(env_t.generate_state(1)[0])

    def episode_options(self, config: dict) -> dict:
        r = self.curriculum
        return {
            'soh_init': float(r.uniform(*config['soh_range'])),
            'soc_init': float(r.uniform(*config['soc_range'])),
            'complexity': float(r.choice(config['complexity_levels'])),
            'mission_distance_m': float(r.choice(config['mission_distances_m'])),
        }


def build_env(config: dict, mode: str = 'train',
              soh_init: float = 1.0,
              complexity: float = 1.0) -> RobotEnv:
    """Instantiate the environment at the configured control frequency."""
    # max_steps sized for the longest mission; the episode distance is set by reset().
    robot = make_robot(dict(config, mission_distance_m=max(config['mission_distances_m'])))
    ecm = ECMParams.default(robot)  # SOC-dependent LiPo fit (lipo_fit_parameters.json)
    env = RobotEnv(robot=robot, ecm=ecm, stress=StressParams(),
                   soc_init=1.0, soh_init=soh_init,
                   complexity=complexity, mode=mode)
    return env


def build_agent(config: dict, env: RobotEnv) -> SACAgent:
    """Construct SAC agent with dimensions inferred from environment."""
    return SACAgent(
        state_dim=env.obs_dim,
        action_dim=env.act_dim,
        action_low=env.action_low,
        action_high=env.action_high,
        hidden_sizes=tuple(config['hidden_sizes']),
        lr=config['lr'],
        gamma=control_gamma(config),
        polyak=config['polyak'],
        alpha_init=config['alpha_init'],
        auto_alpha=config['auto_alpha'],
        buffer_capacity=config['buffer_capacity'],
        batch_size=config['batch_size'],
    )


def _reset(env: RobotEnv, seed=None):
    seed = None if seed is None else int(seed)
    obs, _ = env.reset(seed=seed)
    return obs


def periodic_eval(agent: SACAgent, config: dict) -> dict:
    """Quick DARO check on the evaluation test grid (overall means only)."""
    import evaluation as ev
    # Test seeds stay EVAL_CONFIG['seed'] for every training seed, so runs
    # are compared on identical test episodes.
    ecfg = dict(ev.EVAL_CONFIG, control_dt=config['control_dt'],
                max_steps_margin=config['max_steps_margin'])
    res = ev.evaluate_policies({'DARO': agent}, ecfg,
                               n_episodes=config['eval_episodes'], verbose=False)
    o = res['DARO']['overall']
    print(f"  [eval DARO] R={o['return']:.2f} E={o['energy_kj']:.2f} kJ "
          f"T={o['mission_time_s']:.0f} s QoSv={o['qos_viol_pct']:.1f}% "
          f"done={100 * o['completed']:.0f}%")
    return {'DARO': o}


def log_results(results: dict, log_path: str, episode: int, global_step: int = None):
    """Append evaluation results to a JSONL file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(json.dumps({'episode': episode, 'global_step': global_step,
                            'results': results}) + '\n')


def save_reward_plot(log_dir: str):
    """Reward curve PNG after training; never let plotting break a run."""
    try:
        from plot_rewards import plot_training_curve
        path = plot_training_curve(log_dir)
        if path:
            print(f"Reward curve -> {path}")
    except Exception as exc:  # e.g. matplotlib missing
        print(f"[warn] reward plot skipped: {exc}")


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def train(config: dict, resume_ckpt: str = None):
    """DARO training loop with per-episode SOH / complexity randomisation."""
    config = dict(config)
    if config.get('seed_subdir', True):
        config['ckpt_dir'] = os.path.join(config['ckpt_dir'], f"seed_{config['seed']}")
        config['log_dir'] = os.path.join(config['log_dir'], f"seed_{config['seed']}")
    set_seed(config['seed'], config.get('torch_deterministic', False))
    rngs = SeedStreams(config['seed'])
    os.makedirs(config['ckpt_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    log_path = os.path.join(config['log_dir'], 'eval_log.jsonl')
    with open(os.path.join(config['log_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    ref_env = build_env(config)
    agent = build_agent(config, ref_env)
    if resume_ckpt:
        agent.load(resume_ckpt)

    print("\n" + "=" * 60)
    print("DARO SAC Training")
    print(f"  Device        : {agent.device}")
    print(f"  Seed          : {config['seed']}"
          f"{' (deterministic torch)' if config.get('torch_deterministic') else ''}")
    print(f"  Output        : {config['ckpt_dir']} | {config['log_dir']}")
    print(f"  Control       : dt={config['control_dt']} s "
          f"({1.0 / config['control_dt']:.2f} Hz), gamma={control_gamma(config):.4f} "
          f"(horizon {config['horizon_s']:.0f} s)")
    print(f"  Max ep steps  : {ref_env.robot.max_steps}")
    print(f"  ECM source    : {ref_env.ecm_params.source}")
    print(f"  State dim     : {ref_env.obs_dim}")
    print(f"  Action dim    : {ref_env.act_dim}")
    print(f"  Buffer cap    : {config['buffer_capacity']:,}")
    print(f"  Total steps   : {config['total_steps']:,}")
    sizes = agent.model_size_kb()
    print(f"  Critic size   : {sizes['critic_kb']:.2f} kB ({sizes['critic_params']:,} params)")
    print(f"  Actor size    : {sizes['actor_kb']:.2f} kB ({sizes['actor_params']:,} params)")
    print("=" * 60 + "\n")

    global_step = 0
    episode = 0
    train_log = []

    # Warm-up: fill replay buffer with uniformly random actions.
    print(f"[Warm-up] collecting {config['warmup_steps']} random steps...")
    warmup_env = build_env(config, mode='train')
    obs, _ = warmup_env.reset(seed=rngs.warmup_env_seed,
                              options=rngs.episode_options(config))
    for _ in range(config['warmup_steps']):
        action = rngs.warmup_actions.uniform(warmup_env.action_low,
                                             warmup_env.action_high).astype(np.float32)
        next_obs, reward, terminated, truncated, _ = warmup_env.step(action)
        agent.store(obs, action, reward, next_obs, terminated)
        if terminated or truncated:
            obs, _ = warmup_env.reset(options=rngs.episode_options(config))
        else:
            obs = next_obs
    print("[Warm-up] done.\n")

    t_start = time.time()
    env = build_env(config, mode='train')
    env.reset(seed=rngs.train_env_seed)  # seeds the env disturbance/noise stream once

    while global_step < config['total_steps'] and episode < config['max_episodes']:
        opts = rngs.episode_options(config)
        soh_ep, soc_ep = opts['soh_init'], opts['soc_init']
        c_ep, d_ep = opts['complexity'], opts['mission_distance_m']
        obs, _ = env.reset(options=opts)
        metrics = EpisodeMetrics()
        done = False

        while not done and global_step < config['total_steps']:
            action = agent.select_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store(obs, action, reward, next_obs, terminated)
            agent.update()
            metrics.record(reward, info)
            obs = next_obs
            global_step += 1

        # Do not treat an episode cut short by the training budget as terminal.
        # It has no meaningful completed-mission return or terminal penalty.
        if not done:
            print(f'[train] episode {episode} interrupted at step {global_step}; '
                  'exclude partial episode from reward statistics.')
            break
        summary = metrics.summarise()
        summary.update({'episode': episode, 'soh_init': round(soh_ep, 3),
                        'soc_init': round(soc_ep, 3), 'target_distance_m': d_ep,
                        'complexity': c_ep, 'global_step': global_step})
        # The original 'distance_m' is actual travelled distance; do not overwrite.
        summary['return_per_km'] = (summary['return'] / (summary['distance_m'] / 1000.0)
                                    if summary['completed'] and summary['distance_m'] > 0 else None)
        train_log.append(summary)

        elapsed = time.time() - t_start
        print(f"Ep {episode:5d} | step {global_step:9,} | "
              f"D={d_ep:4.0f} SOH={soh_ep:.2f} SOC0={soc_ep:.2f} C={c_ep:.0f} | "
              f"R={summary['return']:8.2f} "
              f"(-E {summary['sum_E_d']:.2f} -D {summary['sum_D_d']:.2f} "
              f"+U {summary['sum_U_t']:.2f}) | "
              f"E={summary['total_energy_j'] / 1e3:6.2f} kJ | "
              f"T={summary['mission_time_s']:5.0f} s | "
              f"Viol={summary['thr_viol_pct']:5.1f}% | "
              f"done={int(summary['completed'])} | "
              f"a={agent.alpha:.4f} | t={elapsed:.0f}s")

        if config['eval_every_ep'] and (episode + 1) % config['eval_every_ep'] == 0:
            log_results(periodic_eval(agent, config), log_path, episode + 1, global_step)

        if (episode + 1) % config['ckpt_every_ep'] == 0:
            agent.save(config['ckpt_dir'])

        episode += 1

    agent.save(config['ckpt_dir'])
    train_log_path = os.path.join(config['log_dir'], 'train_log.json')
    with open(train_log_path, 'w') as f:
        json.dump(train_log, f, indent=2)
    print(f"\nTraining complete. Log -> {train_log_path}")
    save_reward_plot(config['log_dir'])
    return agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='DARO SAC Training')
    parser.add_argument('--ckpt', type=str, default=None, help='Resume from checkpoint directory')
    parser.add_argument('--seed', type=int, default=None,
                        help=f"Training seed (default {DEFAULT_CONFIG['seed']})")
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Train one run per seed sequentially, e.g. --seeds 0 1 2 3 4')
    parser.add_argument('--deterministic', action='store_true',
                        help='Deterministic torch/cuDNN kernels (bit-reproducible, slower)')
    parser.add_argument('--steps', type=int, default=None, help='Override total training steps')
    parser.add_argument('--episodes', type=int, default=None, help='Override max training episodes')
    parser.add_argument('--control-dt', type=float, default=None,
                        help='Agent decision period in seconds (default 1.0 = 1 Hz)')
    parser.add_argument('--horizon-s', type=float, default=None,
                        help='Effective planning horizon in seconds (sets gamma)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = dict(DEFAULT_CONFIG)
    if args.seed is not None and args.seeds is not None:
        raise SystemExit('Use either --seed or --seeds, not both.')
    seeds = args.seeds or [args.seed if args.seed is not None else config['seed']]
    if args.ckpt and len(seeds) > 1:
        raise SystemExit('--ckpt (resume) works with a single seed only.')
    if args.deterministic:
        config['torch_deterministic'] = True
    if args.steps is not None:
        config['total_steps'] = args.steps
    if args.episodes is not None:
        config['max_episodes'] = args.episodes
    if args.control_dt is not None:
        config['control_dt'] = args.control_dt
    if args.horizon_s is not None:
        config['horizon_s'] = args.horizon_s

    for seed in seeds:
        train(dict(config, seed=int(seed)), resume_ckpt=args.ckpt)
    if len(seeds) > 1 and config.get('seed_subdir', True):
        try:
            from plot_rewards import plot_seed_comparison
            dirs = [os.path.join(config['log_dir'], f'seed_{s}') for s in seeds]
            out = plot_seed_comparison(dirs, os.path.join(config['log_dir'], 'reward_curve_seeds.png'))
            print(f"Seed comparison -> {out}")
        except Exception as exc:
            print(f"[warn] seed comparison plot skipped: {exc}")
