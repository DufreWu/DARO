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
    ├── baselines.py       # AS-MF, QoS-EPD (rule-based)
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

Each run writes to a per-seed folder `./checkpoints/seed_<seed>/`:
`daro_checkpoint.pt` is the **best** checkpoint by periodic-evaluation return
(`best.json` records episode and metrics); the latest weights go to `last/`
(every 500 episodes and at the end). `config.json`, `train_log.json` and
`eval_log.jsonl` (quick evaluation every 100 episodes on validation seeds
`val_seed`, not the test seed) go to `./logs/seed_<seed>/`.

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
python sac_training.py --seeds 0 1 2 3 4 --workers 5   # five seeds in parallel processes
```

`--workers N` trains up to N seeds at once, one process each, with
1 torch thread per process (override with `--threads`). Each parallel run
writes its console output to `./logs/seed_<seed>/train_stdout.log`.

### Device

Training runs on the CPU by default (`--device cpu`). The networks are
small (128×64, batch 64, one update per environment step), so a GPU is
slower here: kernel launches and host↔device copies cost more than the
compute. On an RTX 5090 one update took 2.3 ms on CUDA and 1.7 ms on the
CPU. Use `--device cuda` only with much larger networks or batches. To get
more throughput, train several seeds in parallel with `--workers`.

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
RL-SOC-only, RL-SOH-only (SAC checkpoints, see below), AS-MF, QoS-EPD, BA-EPD; grid SOH × initial SOC × complexity with
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

### Scenario suite (main evaluation for the claim)

"Finish fast subject to charge and QoS, adapting to model mismatch, workload
changes and battery degradation" (`evaluation/scenarios.py`): families
`nominal` (1 km, full charge), `tight` (5 km, SOC0 = k × energy-optimal need above
the 5 % reserve, k ∈ {0.9, 1, 1.1, 1.25}), `switch` (complexity switching every
10/30/120 s), `mismatch` (payload / terrain / compute ≠ nominal model) and
`degradation` (same 5 km mission and SOC0, SOH 1.0 → 0.8). Also used (minus
`degradation`, validation seeds) for checkpoint selection during training.

```bash
python -m evaluation.scenarios --baselines-only                      # ~1 min
python -m evaluation.scenarios --ckpts checkpoints/seed_{0..4} checkpoints/rl_soc/seed_{0..4} \
    checkpoints/rl_soh/seed_{0..4}
```

Results → `./logs/scenarios.json` / `.csv`. Baseline reference (2 seeds):
BA-EPD drives fast (22.5 vs 25.3 min per 5 km) while the charge allows and
energy-optimally near end of life; Fast-QoS (fixed 3.9 m/s) ends below the
reserve at SOH ≤ 0.825; under payload / rough terrain BA-EPD's nominal model
makes fast driving cost more than it gains; with complexity switching every
10 s both QoS-EPD and BA-EPD violate QoS in ~7.5 % of steps.

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
| E_d(t) | w_E · E_t / L  [J/m],  E_t = I·OCV(SOC,SOH)·Δt [J] | Energy drawn from the cell this step per mission metre (includes R0/RC losses) |
| D_d(t) | w_D · S_t / S_ref,  S_t = Δt·[w_I (C/C_ref)²·g_SOC + w_T ((T−T_th)⁺/T_s)² + w_S ((SOC−SOC_th)⁺/(1−SOC_th))²] | Battery-stress proxy (`BatteryStress`); shaping only, **not** ΔSOH |
| U_t | B·Δx/L − w_T·Δt/(L/v_max) − w_S·(Δv/(a_max·Δt))²·Δt/(L/v_max) − w_Q·q_t·Δt/Δt_ctrl − P_fail·𝟙[fail] | Progress, time pressure, speed-change penalty, QoS, terminal failure |

q_t = max(0, 1 − min_i thr_i/QoS_ref_i); U_t uses (1 − q_t)³. L = mission distance.

Battery dependence (so the optimal action depends on SOC/SOH; motivation:
`unit_test/check_battery_sensitivity.py`):

* `RobotParams.energy_at_cell = True`: E_t = I·OCV·Δt (cell side). `False` gives
  the old load-side I·V_terminal·Δt, which is independent of SOC/SOH.
* `StressParams.c_rate_actual_capacity = True`: C = |I| / (Q_n·SOH).
* `StressParams.low_soc_gain = 3`, `low_soc_threshold = 0.5`:
  g_SOC = 1 + gain·((0.5 − SOC)⁺/0.5)² amplifies current stress at low SOC (illustrative).

Reward-optimal constant speed (1000 m, C = 1): 2.50 m/s at SOH 1.0 / SOC 1.0,
2.40 at SOH 0.8 or SOC 0.3, 2.25 at SOH 0.8 and SOC 0.3 (old env: 2.45–2.50 everywhere).
Defaults (`RobotParams`, `StressParams`): w_E = 3 / e_ref = 3 / 5.17 ≈ 0.580 1/(J/m)
(e_ref = 5.17 J/m = 1.44 Wh/km, minimum at C = 1), w_U = B = 2,
w_T = 0, w_S = 0.2, stress_ref = 15.5 s/km (stress at the C = 1 energy optimum), w_D = w_Q = 1, P_fail = 3.

Power model: P = P_motor(v) + P_nonmotor(v, f, c). `RobotParams.compute_model = 'measured'`
(default) takes P_comp(v, c) from the energy-modelling tool (`COMP_TABLE_*` in
`robot_env.py`, estimated from its energy-per-metre figure; includes auxiliaries),
interpolated in v and c, plus `compute_dynamic_w·((f/2)³ − (f_q/2)³)` for a CPU
frequency f away from the QoS frequency f_q. Energy-optimal speed: 3.3 / 3.0 / 3.7 m/s
at complexity 1 / 2 / 3. `'legacy'` restores base + (f/2)³ + aux (optimum ~2.9 m/s
for all complexities). QoS-EPD uses the same `non_motor_power`.

Design: w_E · e_ref = w_U with e_ref ≈ minimum energy per metre makes the per-second
(myopic, discounted) optimum equal to the per-mission energy optimum, for any L.
`python evaluation/reward_check.py` verifies that the reward-optimal constant
speed / CPU frequency are the energy-optimal, QoS-safe ones and that speed
oscillation is penalised. Run it after any change to the reward or power models.

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
| `save(path)` | Saves networks + optimiser states + observation normalisation to `path/daro_checkpoint.pt` |
| `load(path, eval_only)` | Loads checkpoint; sets eval mode when `eval_only=True`. Checkpoints without stored normalisation (older runs) load with identity normalisation |
| `set_lr(lr)` | Sets the learning rate of all optimisers (lr decay) |
| `model_size_kb()` | Reports parameter count and approximate size in kB |

**Key hyperparameters** (§V-A):

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 3 × 10⁻⁴ |
| Discount factor γ | exp(-dt / horizon_s) = 0.99967 (horizon 3000 s, covers the longest 5 km mission) |
| Polyak coefficient τ | 0.995 |
| Replay buffer size | 10⁵ |
| Batch size | 128 |
| Temperature α (initial) | 0.2 (auto-tuned) |
| Target entropy | −dim(A) = −2 |
| Hidden layers | [128, 64] |

**Training stability** (`DEFAULT_CONFIG` in `sac_training.py`, passed to `SACAgent`):

| Key | Default | Effect |
|-----|---------|--------|
| `obs_norm` | True | Fixed `(obs − offset)/scale` from `RobotEnv.obs_normalization` (physical ranges); masked ablation features stay 0 |
| `reward_scale` | 10 | Critic reward scale (per-step reward is ~5·10⁻³) |
| `alpha_min` | 10⁻³ | Floor on α |
| `grad_clip_norm` | 10 | Gradient-norm clipping, critic and actor |
| `lr_final_frac` | 0.1 | Linear lr decay to 10 % at `total_steps` |
| `save_best`, `val_seed` | True, 2024 | Keep the best periodic-eval checkpoint, selected on validation seeds |
| `eval_every_ep`, `eval_episodes` | 100, 3 | Periodic evaluation frequency / seeds per scenario |

---

### `sac_training.py`

End-to-end training and evaluation pipeline.

**Training curriculum** (`SeedStreams.episode_options`) — per episode:

* SOH ~ U[0.80, 1.00], complexity from {1, 2, 3};
* mission 1 / 2 / 5 km (p = 0.5 / 0.3 / 0.2), SOC0 ~ U[0.20, 1.00];
* 30 % charge-limited episodes: 2 or 5 km with SOC0 chosen so that the usable
  energy above the 5 % reserve is k × the energy-optimal mission energy,
  k ~ U[0.8, 1.3] (`tight_soc_init`; k = 1 ends at SOC ≈ 0.048 when driving energy-optimally);
* 30 % with mid-mission complexity switching (period 30 or 120 s);
* 70 % with a randomised true plant: motor fit × U[0.9, 1.5] (payload), v³ term
  × log-U[0.7, 2.0] (terrain), compute × U[0.8, 1.3] (hardware). Rule-based
  baselines keep the nominal model.

**Baseline policies** (§V-A):

| Baseline | Description |
|----------|-------------|
| `AS-MF` | Static: v = 2.5 m/s, f = f_max (`evaluation/baselines.py`) |
| `RL-SOC-only` | SAC, same reward/network/budget as DARO; battery observation reduced to SOC (`sac_training.py --obs soc`) |
| `RL-SOH-only` | SAC, same reward/network/budget as DARO; battery observation reduced to SOH (`sac_training.py --obs soh`) |
| `QoS-EPD` | Non-RL: each step minimises energy per metre (P_motor + P_aux + P_cpu(f_req(v)))/v subject to QoS; f = minimum QoS frequency, complexity estimated on-line |
| `BA-EPD` | Battery-aware QoS-EPD (strong baseline): fast (3.9 m/s, QoS-capped) if the energy for the rest of the mission fits above the 5 % reserve (observed SOC, SOH, distance), else energy-optimal (using the reserve before giving up QoS), else best QoS within the budget |
| `Fast-QoS` | Constant 3.9 m/s (QoS-capped), QoS frequency; battery-agnostic (scenario suite only) |

In the RL ablations SOH (resp. SOC), temperature, terminal voltage and current
are zeroed in the observation, since they leak the hidden state
(`obs_variants.py`). The variant is stored in `<ckpt>/obs_variant.json`, and
the evaluation scripts label and mask each checkpoint from it:

```bash
python sac_training.py --obs soc --seeds 0 1 2 3 4   # -> checkpoints/rl_soc/seed_k
python sac_training.py --obs soh --seeds 0 1 2 3 4   # -> checkpoints/rl_soh/seed_k
python -m evaluation.short_term --by soh --ckpts checkpoints/seed_{0..4} \
    checkpoints/rl_soc/seed_{0..4} checkpoints/rl_soh/seed_{0..4}
```

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

Two independent ablation axes, set on the command line (`obs_variants.py`);
any combination can be trained:

| Flag | Variant | Change | Label | Output folder |
|------|---------|--------|-------|---------------|
| `--obs` | `soc` | battery observation reduced to SOC | RL-SOC-only | `rl_soc` |
| `--obs` | `soh` | battery observation reduced to SOH | RL-SOH-only | `rl_soh` |
| `--reward` | `with_stress` | `w_stress = 1` (adds battery-stress proxy D_d) | DARO + D_d | `rw_with_stress` |
| `--reward` | `no_time` | `w_time = 0` (no T_d, energy-oriented) | DARO w/o T_d | `rw_no_time` |
| `--reward` | `no_energy` | `w_energy = 0` (no E_d) | DARO w/o E_d | `rw_no_energy` |

Combined, e.g. `--obs soc --reward no_energy` -> `RL-SOC-only w/o E_d` in
`checkpoints/rl_soc_rw_no_energy/seed_k`. The reward variant applies to
training only; evaluation always scores with the full reward.

```bash
python sac_training.py --reward with_stress --seeds 0 1 2 3 4
python sac_training.py --reward no_time --seeds 0 1 2 3 4
python sac_training.py --reward no_energy --seeds 0 1 2 3 4
python -m evaluation.short_term --ckpts checkpoints/seed_{0..4} \
    checkpoints/rw_with_stress/seed_{0..4} checkpoints/rw_no_time/seed_{0..4} \
    checkpoints/rw_no_energy/seed_{0..4}
```

---

## Extending DARO

### Adding a new application or sensor

Increase `RobotParams.n_apps` and extend `RobotParams.qos_ref`. The
observation dimension updates automatically to `10 + 2 * n_apps`
(8 robot/battery features, throughput and QoS target per app, mission length L and
fraction of the mission remaining).

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