# DARO — Degradation-Adaptive Runtime Energy Optimization

Reference implementation of the DARO framework from the paper
**"DARO: Battery-Degradation-Adaptive Runtime Energy Optimization in Mobile Robots with Computational Awareness"**
(IEEE RA-L submission 26-1279).

DARO is a Soft Actor-Critic (SAC) reinforcement learning controller that
jointly optimizes motor speed and CPU DVFS frequency at runtime, using both
State Of Charge (SOC) and State Of Health (SOH) of the battery to minimize
energy consumption and battery degradation while satisfying application
Quality-of-Service (QoS) requirements.

---

## Repository layout

```
.
├── network_torch.py   # Neural network definitions (actor + critic)
├── robot_env.py       # Robot environment, battery models, reward functions
├── sac_agent.py       # SAC agent with replay buffer and update logic
├── sac_training.py    # Training loop and CLI (no test code)
└── evaluation/        # Test environment, baselines, metrics, distance sweep
    ├── test_env.py        # EVAL_CONFIG, SOH × SOC × complexity grid, make_test_env
    ├── baselines.py       # AS-MF, SOC-only, Deg-only, SOC+Deg
    ├── metrics.py         # run_episode, Xu capacity fade per mission
    ├── runner.py          # evaluate_policies, summaries, load_daro
    ├── lifetime.py        # missions / km to EOL (extrapolated Xu fade)
    └── distance_sweep.py  # 500 / 1000 / 2500 / 5000 m sweep
```

---

## Requirements

| Package     | Tested version |
|-------------|---------------|
| Python      | ≥ 3.9         |
| PyTorch     | ≥ 2.0         |
| NumPy       | ≥ 1.24        |

Install dependencies:

```bash
pip install torch numpy
```

No additional robotics middleware is required to run training and evaluation.
Hardware-In-the-Loop (HIL) and real-robot experiments require a ROS
environment and a physical NVIDIA Jetson TX2 board (not covered here).

---

## Quick start

### Train from scratch

```bash
python sac_training.py
```

Each run writes to a per-seed folder: checkpoints every 500 episodes to
`./checkpoints/seed_<seed>/`, and `config.json`, `train_log.json` and
`eval_log.jsonl` (quick DARO evaluation every 200 episodes) to
`./logs/seed_<seed>/`.

### Reward curves

Training ends by writing `./logs/seed_<seed>/reward_curve.png`. It shows the
training return per mission distance (moving average) and the periodic DARO
evaluation return. A `--seeds` run also writes `./logs/reward_curve_seeds.png`
(mean ± std across seeds, return per km). To re-plot existing logs:

```bash
python plot_rewards.py logs/seed_42
python plot_rewards.py logs/seed_0 logs/seed_1 logs/seed_2   # + seed comparison
```

### Random seeds

```bash
python sac_training.py --seed 7                 # single run, seed 7
python sac_training.py --seeds 0 1 2 3 4        # five runs, one per seed
python sac_training.py --seed 7 --deterministic # bit-reproducible torch kernels
```

The seed fixes network initialisation, SAC exploration noise, replay sampling,
warm-up actions, the episode curriculum (SOH, initial SOC, complexity,
distance) and the environment disturbance stream; each comes from its own
`SeedSequence` stream. The test seeds in `evaluation/` stay the same for every
training seed, so runs are compared on identical test episodes.

### Resume from a checkpoint

```bash
python sac_training.py --ckpt ./checkpoints/seed_42
```

### Evaluate a trained policy

All testing lives in `evaluation/` (run from this directory). Policies: DARO,
AS-MF, SOC-only, Deg-only, SOC+Deg; grid SOH × initial SOC × complexity with
seeded complexity disturbance and common random numbers across policies:

```bash
python -m evaluation --ckpt ./checkpoints/seed_42          # DARO + baselines
python -m evaluation --baselines-only --distance 1000
```

Results (incl. extrapolated missions/km to EOL) → `./logs/test_results.json`.

### Mission-distance sweep

```bash
python -m evaluation.distance_sweep --ckpt ./checkpoints/seed_42   # 500/1000/2500/5000 m
python -m evaluation.distance_sweep --baselines-only --distances 500 5000
```

Results → `./logs/distance_sweep.json` and `./logs/distance_sweep.csv`.

### Override training scale

```bash
python sac_training.py --steps 200000 --episodes 2000 --seed 7
```

---

## Module reference

### `network_torch.py`

Neural network building blocks for the SAC agent (paper §IV-B-3, Fig. 8).

| Class / function | Description |
|-----------------|-------------|
| `mlp(sizes, ...)` | Builds a ReLU MLP from a list of layer widths |
| `QNetwork` | Single Q(s, a) → ℝ network |
| `TwinQNetwork` | Clipped double-Q critic with main (θ₁, θ₂) and Polyak target networks (θ̄₁, θ̄₂). Exposes `min_q` (actor update) and `target_min_q` (Bellman target) |
| `GaussianActor` | Stochastic policy π_φ(·\|s). Outputs a tanh-squashed Gaussian with affine rescaling into action bounds. Includes reparameterised sampling and Jacobian-corrected log-probabilities |

Both networks use two hidden layers of **128 → 64** neurons, matching §V-A.

---

### `robot_env.py`

Self-contained robot environment and battery models. Has no PyTorch
dependency and can be used or tested independently.

#### Battery models

**`ECMBattery`** — Short-term SOC model (§IV-A-1, Eqs. 1–5)

A 2nd-order RC Equivalent Circuit Model (ECM) with parameters from Table I:

| Parameter | Value |
|-----------|-------|
| Rₛ (internal resistance) | 0.035 Ω |
| R₁, C₁ | 0.22 Ω, 120 F |
| R₂, C₂ | 0.019 Ω, 1500 F |
| Coulombic efficiency η | 0.98 |

Each call to `ECMBattery.step(Pe)` advances the RC dynamics by one control
period (1 s), computes battery current Iₑ from total power demand via Eq. (4),
and returns updated (Iₑ, Vₑ, SOC). Gaussian sensor noise (σ ∝ Pₑ) is injected
during training to improve policy robustness (§V-A).

**`SOHModel`** — Long-term SOH model (§IV-A-2, Eqs. 6–8)

A throughput-based capacity-fade model using Rainflow cycle counting to
decompose the SOC history into stress cycles. SOH is expressed as:

```
SOH(t) = exp(-f_d(N))
```

where f_d accumulates contributions from DOD stress S_δ, average-SOC stress
S_σ, and temperature stress S_T (Eqs. 8a–8c). Battery EOL is defined at
SOH = 0.80, consistent with §II-B.

#### Environment

**`DARORobotEnv`** — Gym-compatible environment implementing the full
observe–decide–act control loop (§IV-B-4).

**State vector** `s(t)` (dim = 4 + 2N, default N=2 applications):

```
[v_obs, f_obs, SOC, SOH, thr_1, thr_2, QoS_ref_1, QoS_ref_2]
```

**Action vector** `a(t)` (dim = 2):

```
[v_cmd (m/s), f_cmd (GHz)]
```

| Action | Range |
|--------|-------|
| Motor speed v | 1.0 – 5.0 m/s |
| CPU frequency f | 0.499 – 2.0 GHz |

**Reward** (`RobotEnv.reward_terms`):

```
r_t = -E_d(t) - D_d(t) + U_t
```

| Term | Definition | Meaning |
|------|------------|---------|
| E_d(t) | w_E · E_t / E_ref,  E_t = I·V·Δt [J] | Energy drawn from the pack this step |
| D_d(t) | w_D · S_t / S_ref,  S_t = Δt·[w_I (C/C_ref)² + w_T ((T−T_th)⁺/T_s)² + w_S ((SOC−SOC_th)⁺/(1−SOC_th))²] | Battery-stress proxy (`BatteryStress`); shaping only, **not** ΔSOH |
| U_t | B·Δx/L − w_T·Δt/(L/v_max) − w_Q·q_t·Δt/Δt_ctrl − P_fail·𝟙[fail] | Task utility: progress, time pressure, QoS, terminal failure |

q_t = (max(0, 1 − min_i thr_i/QoS_ref_i))², L = mission distance.
Defaults (`RobotParams`, `StressParams`): E_ref = 20 kJ, S_ref = 1000 s,
w_E = w_D = w_Q = 1, w_T = 0.15, B = 2, P_fail = 3; w_I = w_T(thermal) = 1,
w_S = 0, C_ref = 1C, T_th = 35 °C, T_s = 10 °C.

Each step's `info` carries `E_d`, `D_d`, `U_t`; the training log stores the
per-episode sums `sum_E_d`, `sum_D_d`, `sum_U_t`.

**Environment modes:**

| Mode | Description |
|------|-------------|
| `'train'` | Simulated; sensor noise injected; SOH updated at episode end |
| `'hil'` | Real computing-board power supplied externally; simulated battery |
| `'real'` | Full hardware; power readings passed via `ext_Pe` argument |

---

### `sac_agent.py`

SAC agent implementing the six-step update from Fig. 8 of the paper.

**`ReplayBuffer`** — Pre-allocated ring buffer of capacity 10⁵. Stores
(s, a, r, s′, done) tuples and supports uniform random batch sampling.

**`SACAgent`**

```python
agent = SACAgent(
    state_dim   = 8,
    action_dim  = 2,
    action_low  = [1.0, 0.499],
    action_high = [5.0, 2.0],
)
```

| Method | Description |
|--------|-------------|
| `select_action(obs, deterministic)` | Returns action array; uses mean for eval, stochastic sample for training |
| `store(s, a, r, s′, done)` | Pushes one transition to the replay buffer |
| `update()` | One gradient step: critic MSE → actor KL loss (Eq. 14) → α update → Polyak soft-update. Returns loss dict |
| `save(path)` | Saves networks + optimiser states to `path/daro_checkpoint.pt` |
| `load(path, eval_only)` | Loads checkpoint; sets eval mode when `eval_only=True` |
| `model_size_kb()` | Reports parameter count and approximate size in kB |

**Key hyperparameters** (§V-A):

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 3 × 10⁻⁴ |
| Discount factor γ | 0.99 |
| Polyak coefficient τ | 0.995 |
| Replay buffer size | 10⁵ |
| Batch size | 128 |
| Temperature α (initial) | 0.2 (auto-tuned) |
| Target entropy | −dim(A) = −2 |
| Hidden layers | [128, 64] |

---

### `sac_training.py`

End-to-end training and evaluation pipeline.

**Training curriculum** — At the start of each episode, SOH is drawn
uniformly from [0.80, 1.00] and complexity from {low, medium, high} so the
policy learns to generalize across the full range of battery health states
and environmental conditions without overfitting to a single operating point.

**Baseline policies** (§V-A):

| Baseline | Description |
|----------|-------------|
| `AS-MF` | Static: v = 2.5 m/s, f = f_max (`evaluation/baselines.py`) |
| `SOC-only` | QoS-aware DVFS, speed scaled by SOC |
| `Deg-only` | QoS-aware DVFS, speed scaled by SOH |
| `SOC+Deg` | QoS-aware DVFS, speed scaled by SOC × SOH (rule-based DARO counterpart) |

**Key functions:**

| Function | Description |
|----------|-------------|
| `train(config)` | Main training loop; handles warm-up, curriculum, checkpointing, periodic eval |
| `periodic_eval(agent, config)` | Quick DARO check on the evaluation grid during training |

Evaluation entry points are in `evaluation/` (`evaluate_policies`,
`lifetime_from_results`, `distance_sweep.sweep`).

---

## Reproducing paper results

### Energy consumption and QoS (Fig. 9)

```bash
# Train
python sac_training.py

# Evaluate across all SOH × SOC × complexity conditions
python -m evaluation --ckpt ./checkpoints/seed_42
# → ./logs/test_results.json
```

### Battery lifetime (Table II)

```bash
python -m evaluation --ckpt ./checkpoints/seed_42   # lifetime printed and saved in test_results.json
```

Expected results (averaged over 20 runs, medium complexity):

| Method | Missions to EOL |
|--------|----------------|
| AS-MF  | 1256 ± 10 |
| BACO   | 1327 ± 11 |
| HAEM   | 1647 ± 13 |
| **DARO** | **1933 ± 15** |

### Ablation study (Table III)

Train three variants by modifying `DEFAULT_CONFIG['auto_alpha']` and the
reward function in `robot_env.py`:

| Variant | Change |
|---------|--------|
| SOC-only | Set `w_stress = 0` in `RobotParams` to disable D_d |
| SOH-only | Set `w_energy = 0` in `RobotParams` to disable E_d |
| SOC+SOH (full DARO) | Default config |

---

## Extending DARO

### Adding a new application or sensor

Increase `RobotParams.n_apps` and extend `RobotParams.qos_ref`. The
observation dimension updates automatically to `4 + 2 * n_apps`.

### Deploying on hardware

1. Instantiate `DARORobotEnv` with `mode='real'`.
2. At each control step, read actual power from the power management board
   and pass it as `ext_Pe` to `env.step(action, ext_Pe=...)`.
3. Use `agent.load(ckpt_dir, eval_only=True)` and call
   `agent.select_action(obs, deterministic=True)` for inference.
4. The pre-trained model is small enough for embedded deployment:
   critic ≈ 85 kB, actor ≈ 47 kB; end-to-end latency < 200 ms on Jetson TX2
   at minimum CPU frequency (§V-B).

### Switching the RL algorithm

`SACAgent` is decoupled from `DARORobotEnv`. The environment exposes a
standard `reset() / step()` interface, making it straightforward to drop in
PPO, TD3, or any other continuous-control algorithm.

---

## Citation

Waiting for accept, god bless.
```