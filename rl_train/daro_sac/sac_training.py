"""
sac_training.py
---------------
Full training pipeline for the DARO SAC agent (§V-A).

Reproduces the training setup from the paper:
  - 10^4 episodes, 10^6 environment steps total
  - Randomised SOH ∈ [0.80, 1.00] and complexity ∈ {low, medium, high}
    at each episode to ensure robustness across battery states
  - Gaussian power-sensor noise proportional to actual consumption
  - Periodic evaluation against all baselines and SOH levels
  - Checkpoint saving at regular intervals

Usage
-----
    python sac_training.py                      # full training run
    python sac_training.py --eval --ckpt ./ckpt # load checkpoint, evaluate only
"""

import argparse
import os
import time
import random
import json
from typing import List, Dict

import numpy as np

from robot_env import DARORobotEnv, RobotParams, ECMParams, DegradationParams
from sac_agent import SACAgent

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Training scale (paper §V-A)
    "total_steps":      1_000_000,
    "max_episodes":     10_000,
    "max_steps_ep":     1_000,          # steps per episode (1 s each → ~17 min mission)
    "batch_size":       128,
    "buffer_capacity":  100_000,
    # SAC hyperparameters (§V-A)
    "lr":               3e-4,
    "gamma":            0.99,
    "polyak":           0.995,
    "alpha_init":       0.2,
    "auto_alpha":       True,
    "hidden_sizes":     [128, 64],
    # Curriculum / robustness
    "soh_range":        [0.80, 1.00],   # uniform draw each episode
    "complexity_levels": [1.0, 2.0, 3.0],  # low / medium / high
    # Warm-up before first update
    "warmup_steps":     5_000,
    # Evaluation
    "eval_every_ep":    200,            # evaluate every N training episodes
    "eval_episodes":    20,             # independent runs per eval condition (§V-A)
    "eval_soh_levels":  [1.00, 0.95, 0.90, 0.85],
    # Checkpointing
    "ckpt_every_ep":    500,
    "ckpt_dir":         "./checkpoints",
    # Logging
    "log_dir":          "./logs",
    "seed":             42,
}


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

class ASMFBaseline:
    """Average Speed – Maximum CPU Frequency (§V-A)."""
    def select_action(self, obs):
        return np.array([2.5, 2.0], dtype=np.float32)


class BACOBaseline:
    """
    Battery-Aware Co-Optimisation (SOC-aware, §V-A [4]).

    Simplified: reduces speed proportionally when SOC falls below 50%.
    """
    def select_action(self, obs):
        soc = float(obs[2])
        speed = 2.5 if soc > 0.5 else max(1.0, 2.5 * soc / 0.5)
        freq  = min(2.0, 0.499 + soc * 1.501)
        return np.array([speed, freq], dtype=np.float32)


class HAEMBaseline:
    """
    Health-Aware Energy Management (SOH-aware mechanical, §V-A [7]).

    Reduces speed when SOH degrades; CPU frequency set to minimum
    that satisfies QoS (independent optimisation).
    """
    def select_action(self, obs):
        soh   = float(obs[3])
        speed = max(1.0, 3.0 * soh)      # scale with battery health
        freq  = 1.0                       # minimum frequency heuristic
        return np.array([speed, freq], dtype=np.float32)


# ---------------------------------------------------------------------------
# Metric aggregator
# ---------------------------------------------------------------------------

class EpisodeMetrics:
    """Accumulate per-step metrics across an episode."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.rewards:      List[float] = []
        self.energy:       List[float] = []
        self.degradation:  List[float] = []
        self.qos_utility:  List[float] = []
        self.thr_violations: int = 0
        self.steps:        int  = 0

    def record(self, reward, info):
        self.rewards.append(reward)
        self.energy.append(info['Ed'])
        self.degradation.append(info['Dd'])
        self.qos_utility.append(info['U'])
        if info['U'] < 1.0:
            self.thr_violations += 1
        self.steps += 1

    def summarise(self) -> dict:
        n = max(self.steps, 1)
        return {
            'return':          float(np.sum(self.rewards)),
            'mean_Ed':         float(np.mean(self.energy)),
            'mean_Dd':         float(np.mean(self.degradation)),
            'mean_U':          float(np.mean(self.qos_utility)),
            'thr_viol_pct':    100.0 * self.thr_violations / n,
            'steps':           self.steps,
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env(config: dict, mode: str = 'train',
              soh_init: float = 1.0,
              complexity: float = 1.0) -> DARORobotEnv:
    """Instantiate environment with paper-default parameters (Table I)."""
    robot  = RobotParams()
    ecm    = ECMParams()
    deg    = DegradationParams()
    return DARORobotEnv(
        robot=robot, ecm=ecm, deg=deg,
        soc_init=1.0,
        soh_init=soh_init,
        complexity=complexity,
        mode=mode,
        max_steps=config['max_steps_ep'],
    )


def build_agent(config: dict, env: DARORobotEnv) -> SACAgent:
    """Construct SAC agent with dimensions inferred from environment."""
    return SACAgent(
        state_dim=env.observation_dim,
        action_dim=env.act_dim,
        action_low=env.action_low,
        action_high=env.action_high,
        hidden_sizes=tuple(config['hidden_sizes']),
        lr=config['lr'],
        gamma=config['gamma'],
        polyak=config['polyak'],
        alpha_init=config['alpha_init'],
        auto_alpha=config['auto_alpha'],
        buffer_capacity=config['buffer_capacity'],
        batch_size=config['batch_size'],
    )


def run_episode(env: DARORobotEnv, agent,
                train: bool = True) -> EpisodeMetrics:
    """
    Run a single episode.

    Parameters
    ----------
    train : if True, interactions are used to update the agent.
    """
    obs = env.reset()
    metrics = EpisodeMetrics()
    done = False

    while not done:
        deterministic = not train
        action = agent.select_action(obs, deterministic=deterministic)

        next_obs, reward, done, info = env.step(action)

        if train:
            agent.store(obs, action, reward, next_obs, done)
            agent.update()

        metrics.record(reward, info)
        obs = next_obs

    return metrics


def evaluate(agent: SACAgent, config: dict,
             baselines: dict = None) -> dict:
    """
    Evaluate DARO and baseline policies across all SOH / complexity
    combinations (paper §V-B: 20 independent runs each).

    Returns a nested dict: results[policy][soh][complexity] = metrics dict.
    """
    if baselines is None:
        baselines = {
            'AS-MF': ASMFBaseline(),
            'BACO':  BACOBaseline(),
            'HAEM':  HAEMBaseline(),
        }

    policies = {'DARO': agent, **baselines}
    results: Dict = {}

    for pol_name, pol in policies.items():
        results[pol_name] = {}
        for soh in config['eval_soh_levels']:
            results[pol_name][soh] = {}
            for c_idx, c in enumerate(config['complexity_levels']):
                ep_summaries = []
                for _ in range(config['eval_episodes']):
                    env = build_env(config, mode='hil',
                                    soh_init=soh, complexity=c)
                    m = run_episode(env, pol, train=False)
                    ep_summaries.append(m.summarise())

                # Aggregate over runs
                agg = {k: float(np.mean([ep[k] for ep in ep_summaries]))
                       for k in ep_summaries[0]}
                agg['std_return'] = float(np.std([ep['return']
                                                  for ep in ep_summaries]))
                results[pol_name][soh][c] = agg
                print(f"  [{pol_name}] SOH={soh:.2f} "
                      f"C={'LMH'[c_idx]}  "
                      f"Ed={agg['mean_Ed']:.3f}  "
                      f"Viol={agg['thr_viol_pct']:.1f}%  "
                      f"R={agg['return']:.2f}")
    return results


def log_results(results: dict, log_path: str, episode: int):
    """Serialise evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {'episode': episode, 'results': results}
    with open(log_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def train(config: dict, resume_ckpt: str = None):
    """
    Full DARO training loop.

    Episodes are terminated by SOC depletion or max_steps.  At the start of
    each episode, SOH and complexity are drawn from the ranges specified in
    config so the policy learns to generalise over all battery health states
    and environmental conditions (§V-A).
    """
    set_seed(config['seed'])
    os.makedirs(config['ckpt_dir'], exist_ok=True)
    os.makedirs(config['log_dir'],  exist_ok=True)
    log_path = os.path.join(config['log_dir'], 'eval_log.jsonl')

    # Build a reference environment (dimensions only)
    ref_env = build_env(config)
    agent   = build_agent(config, ref_env)

    if resume_ckpt:
        agent.load(resume_ckpt)

    print("\n" + "="*60)
    print("DARO SAC Training")
    print(f"  Device        : {agent.device}")
    print(f"  State dim     : {ref_env.observation_dim}")
    print(f"  Action dim    : {ref_env.act_dim}")
    print(f"  Buffer cap    : {config['buffer_capacity']:,}")
    print(f"  Total steps   : {config['total_steps']:,}")
    sizes = agent.model_size_kb()
    print(f"  Critic size   : {sizes['critic_kb']:.2f} kB  "
          f"({sizes['critic_params']:,} params)")
    print(f"  Actor size    : {sizes['actor_kb']:.2f} kB  "
          f"({sizes['actor_params']:,} params)")
    print("="*60 + "\n")

    global_step = 0
    episode     = 0
    train_log   = []

    # Warm-up: fill replay buffer with random actions before first update
    print(f"[Warm-up] collecting {config['warmup_steps']} random steps…")
    warmup_env = build_env(config, mode='train', soh_init=1.0, complexity=1.0)
    obs = warmup_env.reset()
    for _ in range(config['warmup_steps']):
        action = np.random.uniform(warmup_env.action_low,
                                   warmup_env.action_high).astype(np.float32)
        next_obs, reward, done, info = warmup_env.step(action)
        agent.store(obs, action, reward, next_obs, done)
        obs = next_obs if not done else warmup_env.reset()
    print("[Warm-up] done.\n")

    t_start = time.time()

    while global_step < config['total_steps'] and episode < config['max_episodes']:

        # ---- Curriculum: randomise SOH and complexity each episode ----
        soh_lo, soh_hi = config['soh_range']
        soh_ep = np.random.uniform(soh_lo, soh_hi)
        c_ep   = random.choice(config['complexity_levels'])

        env = build_env(config, mode='train',
                        soh_init=soh_ep, complexity=c_ep)
        obs = env.reset()
        metrics = EpisodeMetrics()
        done = False
        ep_steps = 0

        while not done and global_step < config['total_steps']:
            action = agent.select_action(obs, deterministic=False)
            next_obs, reward, done, info = env.step(action)
            agent.store(obs, action, reward, next_obs, done)

            losses = agent.update()

            metrics.record(reward, info)
            obs = next_obs
            global_step += 1
            ep_steps    += 1

        summary = metrics.summarise()
        summary.update({'episode': episode, 'soh_init': round(soh_ep, 3),
                        'complexity': c_ep, 'global_step': global_step})
        train_log.append(summary)

        # ---- Console output ----
        elapsed = time.time() - t_start
        print(f"Ep {episode:5d} | step {global_step:8,} | "
              f"SOH={soh_ep:.2f} C={c_ep:.0f} | "
              f"R={summary['return']:8.2f} | "
              f"Ed={summary['mean_Ed']:.3f} | "
              f"Viol={summary['thr_viol_pct']:.1f}% | "
              f"α={agent.alpha:.4f} | "
              f"t={elapsed:.0f}s")

        # ---- Periodic evaluation ----
        if (episode + 1) % config['eval_every_ep'] == 0:
            print(f"\n--- Evaluation at episode {episode+1} ---")
            eval_results = evaluate(agent, config)
            log_results(eval_results, log_path, episode + 1)
            print()

        # ---- Checkpoint ----
        if (episode + 1) % config['ckpt_every_ep'] == 0:
            agent.save(config['ckpt_dir'])

        episode += 1

    # Final save
    agent.save(config['ckpt_dir'])
    # Save training log
    train_log_path = os.path.join(config['log_dir'], 'train_log.json')
    with open(train_log_path, 'w') as f:
        json.dump(train_log, f, indent=2)
    print(f"\nTraining complete. Log → {train_log_path}")
    return agent


# ---------------------------------------------------------------------------
# Evaluation-only mode
# ---------------------------------------------------------------------------

def evaluate_only(ckpt_dir: str, config: dict):
    """Load a trained checkpoint and run the full evaluation suite."""
    ref_env = build_env(config)
    agent   = build_agent(config, ref_env)
    agent.load(ckpt_dir, eval_only=True)

    print("\n--- Full evaluation ---")
    results = evaluate(agent, config)
    out_path = os.path.join(config['log_dir'], 'eval_results.json')
    os.makedirs(config['log_dir'], exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")
    return results


# ---------------------------------------------------------------------------
# Long-term battery lifetime evaluation  (Table II, §V-B)
# ---------------------------------------------------------------------------

def evaluate_lifetime(agent: SACAgent, config: dict,
                       n_runs: int = 20) -> dict:
    """
    Count the number of repeated 10 km missions before battery EOL for
    each complexity level (Table II).

    Uses the same degradation model as training (HIL setting).
    SOH is tracked across missions and updated via Rainflow counting.
    """
    baselines = {
        'AS-MF': ASMFBaseline(),
        'BACO':  BACOBaseline(),
        'HAEM':  HAEMBaseline(),
    }
    policies = {'DARO': agent, **baselines}
    results  = {}

    for pol_name, pol in policies.items():
        results[pol_name] = {}
        for c_idx, c in enumerate(config['complexity_levels']):
            mission_counts = []
            for run in range(n_runs):
                soh     = 1.0
                count   = 0
                while soh > 0.80:
                    env = build_env(config, mode='hil',
                                    soh_init=soh, complexity=c)
                    obs = env.reset(soh_init=soh, complexity=c)
                    done = False
                    while not done:
                        action   = pol.select_action(obs, deterministic=True)
                        obs, _, done, info = env.step(action)
                    soh = info['SOH']
                    count += 1
                    if count > 5000:     # safety cap
                        break
                mission_counts.append(count)

            mean_c = float(np.mean(mission_counts))
            std_c  = float(np.std(mission_counts))
            cname  = ['Low', 'Medium', 'High'][c_idx]
            results[pol_name][cname] = {'mean': mean_c, 'std': std_c}
            print(f"  [{pol_name}] {cname}: {mean_c:.0f} ± {std_c:.0f} missions")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='DARO SAC Training')
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation only (requires --ckpt)')
    parser.add_argument('--lifetime', action='store_true',
                        help='Run lifetime evaluation (Table II)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to checkpoint directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=None,
                        help='Override total training steps')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Override max training episodes')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = dict(DEFAULT_CONFIG)
    config['seed'] = args.seed
    if args.steps is not None:
        config['total_steps'] = args.steps
    if args.episodes is not None:
        config['max_episodes'] = args.episodes

    if args.eval:
        if not args.ckpt:
            raise ValueError("--eval requires --ckpt <dir>")
        evaluate_only(args.ckpt, config)

    elif args.lifetime:
        if not args.ckpt:
            raise ValueError("--lifetime requires --ckpt <dir>")
        ref_env = build_env(config)
        agent   = build_agent(config, ref_env)
        agent.load(args.ckpt, eval_only=True)
        print("\n--- Lifetime evaluation (Table II) ---")
        lifetime_results = evaluate_lifetime(agent, config,
                                              n_runs=config['eval_episodes'])
        out_path = os.path.join(config['log_dir'], 'lifetime_results.json')
        os.makedirs(config['log_dir'], exist_ok=True)
        import json
        with open(out_path, 'w') as f:
            json.dump(lifetime_results, f, indent=2)
        print(f"Lifetime results saved → {out_path}")

    else:
        train(config, resume_ckpt=args.ckpt)