# unit_test: motivation check for DARO

`check_battery_sensitivity.py` asks whether the battery state changes the
optimal action. It runs constant-speed policies (CPU at the lowest frequency
that meets QoS) for several (SOH, initial SOC) pairs and reports the
reward-optimal speed. If that speed does not move with SOC/SOH, a
battery-aware policy (DARO) cannot beat RL-SOC-only / RL-SOH-only / QoS-EPD.

It uses the environment in `envs/robot_env.py`.

```bash
python unit_test/check_battery_sensitivity.py            # current env
python unit_test/check_battery_sensitivity.py --legacy   # old env behaviour
```

## Result (1000 m, complexity 1)

| SOH / SOC0 | old env: v* [m/s] | current env: v* [m/s] | stress at v* (current env) |
|---|---|---|---|
| 1.0 / 1.0 | 2.50 | 2.50 | 7.2 |
| 0.8 / 1.0 | 2.50 | 2.40 | 11.0 |
| 1.0 / 0.3 | 2.50 | 2.40 | 11.2 |
| 0.8 / 0.3 | 2.45 | 2.25 | 17.8 |

* **Old env**: energy was load-side (`V_terminal·I·dt`) and the stress C-rate used
  the rated capacity, so energy was identical for every SOC/SOH (4.017 kJ at 2.5 m/s)
  and the optimum hardly moved. Battery observations carried no useful information.
* **Current env**: energy from the cell (`OCV·I·dt`), C-rate relative to
  `Q_n·SOH`, and current stress amplified at low SOC. The optimum moves to lower
  speeds as the battery degrades or discharges.
* A battery-blind policy at the fresh-battery optimum loses 0.4 % (SOH or SOC low)
  to 1.4 % (both low) of the return. Energy changes by < 1 % because the pack
  current is only ~0.15 C, so stress, not energy, is where battery awareness pays off.

## Speed vs energy, final SOC/SOH and missions to EOL (`check_speed_lifetime.py`)

For each mission distance, constant speeds 0.5–5 m/s (CPU at the lowest QoS
frequency) are run fully charged (SOC0 = 1), followed by a recharge (1 C, datasheet range 1–3 C; the table below was run at 0.5 C — at 1 C missions to EOL are ~0.2–0.4 % higher).
Reported per speed:

* energy (cell side) and Wh/km,
* `final_soc`: SOC after one mission on a fresh battery; `final_soc_eol`: same at SOH 0.8,
* `final_soh`: SOH after one mission + recharge from SOH 1 (`delta_soh_ppm` = loss),
* missions / km to EOL (SOH 0.8): the per-cycle Xu fade measured at SOH 1.0…0.8
  is integrated, the same method as `evaluation/lifetime.py`. NaN if the
  mission cannot be completed at some SOH level.

```bash
python unit_test/check_speed_lifetime.py                     # 10/20/50 km, ~2 min
python unit_test/check_speed_lifetime.py --distances 500 5000 --complexity 2 \
    --plot speed_lifetime.png --csv speed_lifetime.csv
```

Result (complexity 1, step 0.25 m/s); the energy- and lifetime-optimal speed coincide:

| distance | optimal speed | energy | final SOC (SOH 1) | final SOC (SOH 0.8) | final SOH | ΔSOH / mission | missions to EOL | km to EOL |
|---|---|---|---|---|---|---|---|---|
| 10 km | 3.0 m/s | 39.6 kJ | 0.854 | 0.816 | 0.999920 | 80 ppm | 2 582 | 25 820 |
| 20 km | 3.0 m/s | 79.2 kJ | 0.708 | 0.631 | 0.999867 | 133 ppm | 1 526 | 30 512 |
| 50 km | 3.0 m/s | 198.1 kJ | 0.264 | 0.053 | 0.999645 | 355 ppm | 499 | 24 933 |

* The lowest-energy speed (~2.9–3.0 m/s, 1.10 Wh/km) does not depend on the distance.
* The speed with the most missions to EOL is the same: under the Xu model the
  fade per cycle is driven by the depth of discharge, i.e. by energy.
* 50 km: not completed at ≤ 1.25 m/s even on a fresh battery; at SOH 0.8 only
  2.5–3.5 m/s completes. km to EOL peak at 20 km (very deep cycles cost more).

## C-rate-dependent aging (`check_crate_aging.py`)

Question: if cycle aging also depends on C-rate, does the lifetime-optimal speed
move with SOH/SOC (so battery awareness pays off)? The Xu cycle fade gets a
charge-weighted factor `S_I = exp(k_I (C − C_ref))`, C relative to the actual
capacity `Q_n·SOH`. Criterion: least fade per km at the given (SOH, SOC0).

* `k_I = 0.149 /C`: C-rate term of Wang et al. 2011 (LFP), the only literature value used.
* `k_I` and `C_ref` cannot be identified from `battery_model/LiPo_dataset`: one
  protocol only (1.01 Ah cells, 3 A ≈ 3 C). `C_ref = 3` is consistent with how
  `k_cyc` was calibrated. `C_ref` is not a pure scale: it sets the cycle/calendar balance.

```bash
python unit_test/check_crate_aging.py --capacity-ah 5 1            # ~15 s
python unit_test/check_crate_aging.py --capacity-ah 5 2 1 --c-ref 0.5
```

Lifetime-optimal speed [m/s], 1000 m, complexity 1 (energy-optimal: 2.85):

| pack | C-rate at 2.85 m/s | k_I = 0.149, C_ref 3 | k_I = 1, C_ref 3 | k_I = 3, C_ref 3 | k_I = 3, C_ref 0.5 |
|---|---|---|---|---|---|
| 5 Ah, SOH 1.0 → 0.8 | 0.15 → 0.19 C | 2.90 → 2.85 | 3.15 → 3.05 | 5.00 → 5.00 | 2.60 → 2.50 |
| 1 Ah, SOH 1.0 → 0.8 | 0.75 → 0.94 C | 2.75 → 2.70 | 2.35 → 2.20 | 3.30 → 2.95 | 1.65 → 1.45 |

* With the literature coefficient the optimum barely moves (≤ 0.05 m/s, < 0.2 % life gain),
  for both pack sizes: at this robot's C-rates C-rate aging is negligible.
* Only with a much stronger C-rate effect (k_I ≥ 1) and a higher C-rate (small pack)
  does the optimum shift with SOH, and then even its direction (slower or faster
  than the energy optimum) depends on `C_ref`, i.e. on the unknown cycle/calendar balance.
* So option 1 alone does not give DARO a defensible advantage. It needs C-rate
  aging data for this LiPo (cycling at ≥ 2 discharge rates) and an operating
  point with a meaningful C-rate.

## Pack datasheet vs model

Pack: 5000 mAh, 14.8 V, 4S1P LiPo, 60 C continuous discharge, EC5, charge 1–3 C.

| item | model | status |
|---|---|---|
| capacity / voltage | `Q_n` 5 Ah, 4S OCV fit | matches |
| recharge in lifetime evaluation | was 0.5 C | now 1 C (`EVAL_CONFIG['charge_c_rate']`); 0.5→1→3 C changes missions to EOL by only +0.2 / +0.5 % |
| current limit | was 30 A placeholder | now 300 A (60 C); the robot draws < 3 A |
| internal resistance | 4S × 4.95P of the 1.01 Ah dataset cells → ~24 mΩ per 5 Ah cell (DC, R0+R1+R2) | not changed; likely high for a 60 C pack, measure it (e.g. ΔV/ΔI under a load step) |
| operating C-rate | ~0.15 C at 2.9 m/s, ≤ 0.35 C at 5 m/s | ≈ 0.25–0.6 % of the 60 C rating |

Consequence: the pack is far oversized in power, so C-rate-dependent aging and
I²R losses stay negligible; the 1 Ah "small pack" cases in `check_crate_aging.py`
do not correspond to this hardware.

## Measured compute-power model (energy-modelling tool)

The legacy compute model (2 W + 1.5 (f/2)³ W + 2 W aux ≈ 4.2–5.5 W) was about half
the computing power from the energy-modelling tool (7.4–11.6 W, rising with speed
and complexity). `robot_env.py` now uses `compute_model = 'measured'`: a table
P_comp(v, c) estimated from the tool's energy-per-metre figure (total − motor fit),
reproducing it within 0.01 J/m. Reward references were recomputed
(e_ref 3.96 → 5.17 J/m, stress_ref 8 → 15.5 s/km).

Rerun (`check_speed_lifetime.py --distances 1000 10000 20000 --dv 0.1 --complexity c`, charge 1 C):

| complexity | energy- = lifetime-optimal speed | Wh/km | missions to EOL (1 km / 10 km / 20 km) | final SOC 10 km (SOH 1 / 0.8) |
|---|---|---|---|---|
| 1 | 3.3 m/s | 1.43 | 9 216 / 2 157 / 1 217 | 0.810 / 0.760 |
| 2 | 3.0 m/s | 1.48 | 9 023 / 2 098 / 1 174 | 0.803 / 0.751 |
| 3 | 3.7 m/s | 1.66 | 8 470 / 1 929 / 1 047 | 0.780 / 0.721 |

Cost of a complexity-blind fixed speed (10 km, extra energy vs the per-complexity optimum):

| fixed speed | C = 1 | C = 2 | C = 3 | mean |
|---|---|---|---|---|
| 2.5 m/s (AS-MF) | 6.4 % | 4.4 % | 7.3 % | 6.0 % |
| 3.0 m/s | 0.6 % | 0.0 % | 3.7 % | 1.4 % |
| 3.3 m/s (best fixed) | 0.0 % | 1.0 % | 1.2 % | 0.75 % |

* The optimum now depends on complexity (3.0–3.7 m/s), not on SOH/SOC (battery check:
  2.75–3.0 m/s reward optimum, energy optimum 3.25–3.3 m/s for all SOH/SOC).
* Lifetime still follows energy (Xu fade ∝ DoD); with C-rate aging (k_I = 0.149) the
  lifetime optimum stays within 0.05 m/s of the energy optimum.
* Complexity adaptation is worth ~0.75 % over the best fixed speed and ~6 % over
  AS-MF (2.5 m/s). QoS-EPD's 20 % safety margin on the complexity estimate costs
  ~5 % at C = 2 (5.66 vs 5.36 kJ per km), part of which is needed under ±20 % jitter.

## Where could DARO beat the rule-based baseline? (scenario checks)

No training: policies are variants of the QoS-EPD optimiser (`scenario_utils.py`),
differing only in what they know. Gap between QoS-EPD and an oracle = the most a
learner could recover.

### 1. Model mismatch (`check_model_mismatch.py`, ~1 min)

True plant ≠ QoS-EPD's calibrated model (payload, terrain, compute hardware).
Oracle = better of (EPD with the true model, best QoS-compliant constant speed in hindsight). 2 km, ±20 % jitter, 3 seeds.

| true plant | QoS-EPD extra energy vs oracle (C = 1 / 2 / 3) |
|---|---|
| nominal | +0.5 / +2.5 / +3.8 % (cost of its 20 % complexity margin only) |
| payload (+50 % motor) | +1.1 / +3.6 / +2.8 % |
| rough terrain (v³ × 2) | +3.4 / +6.7 / +6.4 % |
| smooth floor (v³ × 0.7) | +1.4 / +2.2 / +5.4 % |
| extra sensors (compute × 1.3) | +0.9 / +1.8 / +4.3 % |
| efficient compute (× 0.8) | +0.7 / +3.4 / +3.5 % |

Mean +3.2 % (max +6.7 %) on mismatched plants, +2.3 % on the nominal plant.
A fixed 3.3 m/s is often as good as QoS-EPD; AS-MF is 7–15 % worse.

### 2. Complexity switching during the mission (`check_varying_complexity.py`, ~10 s)

Nominal level re-drawn from {1, 2, 3} every period, plus ±20 % jitter. 3 km, 6 seeds.

| switch period | QoS-EPD Wh/km, viol % | oracle-safe (knows level) | oracle-tight (no margin) | EPD vs oracle-safe |
|---|---|---|---|---|
| constant | 1.548, 0.0 % | 1.523, 0.0 % | 1.505, 12.7 % | +1.7 % |
| 120 s | 1.557, 0.7 % | 1.530, 0.0 % | 1.516, 12.5 % | +1.8 % |
| 30 s | 1.572, 2.9 % | 1.542, 0.0 % | 1.530, 13.3 % | +2.0 % |
| 10 s | 1.584, 8.5 % | 1.547, 0.0 % | 1.543, 13.6 % | +2.4 % |

Not knowing the level costs ~2 % energy and, with fast switching, QoS violations
(8.5 % of steps at 10 s). Dropping the safety margin saves only 0.3–1.2 % but costs ~13 % violations.

### 3. Battery as a constraint (`check_battery_constrained.py`, ~1 min)

Long missions, SOH 1.0 / 0.8, full charge. QoS-EPD (energy-optimal), fast (4 m/s),
battery-aware (fast if affordable → energy-optimal → keep as much QoS as the energy budget allows).

* Battery allows it (e.g. 20–35 km): battery-aware = fast, **19 % shorter mission**
  (83 vs 103 min) with full QoS and completion. Only valuable with a time objective
  (the reward now has one, see below).
* Near the range limit (C = 3: 45 km at SOH 1, 35 km at SOH 0.8): battery-aware
  **completes where both others fail**, but at 28–42 % QoS level: most compute
  power is static, so lowering f saves little energy.
* The window is narrow: range differs by only ~3 % between fast and energy-optimal
  driving (48.4 vs 49.8 km at C = 1, SOH 1).

### Summary

| scenario | achievable gain over QoS-EPD | battery state needed? |
|---|---|---|
| model mismatch | ~3 % energy (up to ~7 %) | no (power/current observations) |
| complexity switching | ~2 % energy + fewer QoS violations | no |
| battery-limited missions | ~19 % time when affordable; completion near the range limit | yes |

## Reward for "finish fast under charge and QoS constraints" (`check_reward_objective.py`, ~3.5 min)

Paper claim: complete the mission as fast as possible subject to the battery charge
and QoS, adapting to model mismatch, complexity changes and battery degradation
(degradation changes the charge budget: 10 km at SOH 0.8 needs ~0.26 SOC vs 0.20 fresh).

Reward per step (all terms normalised per mission, `T_ref = L / v_ref`, `v_ref = 3.3 m/s`):

    r = U_t - E_d - D_d - T_d - Q_d (+ R_end at episode end)
    U_t = w_utility * dd / L                        progress (sum = 2 if completed)
    E_d = w_energy * E / L                          energy (sum ~ 3 at the energy optimum)
    D_d = w_stress * stress / S_ref                 w_stress = 0 (uncalibrated proxy; ablation 'with_stress')
    T_d = w_time * dt / T_ref                       w_time = 1
    Q_d = w_qos * dt / T_ref * (1[viol > 0] + viol) w_qos = 3
    R_end = +2 if completed else -10;  - 5 * clip((0.05 - SOC_end) / 0.05, 0, 1)

Old reward: QoS only as a cubic factor on progress, no time / terminal terms;
its optimum at C = 3 violated QoS in 24 % of steps (v 2.7 m/s, f margin 1.1).

A. Enough charge (1 km, constant speed, CPU at the QoS frequency for c * m):

| C | SOH | reward-optimal v | energy-optimal v | f margin m | QoS violation |
|---|---|---|---|---|---|
| 1 | 1.0 / 0.8 | 3.9 / 3.8 | 3.3 / 3.2 | 1.2 | 0 % |
| 2 | 1.0 / 0.8 | 3.6 / 3.5 | 3.0 / 3.0 | 1.2 | 0 % |
| 3 | 1.0 / 0.8 | 3.7 / 3.7 (QoS limit) | 3.5 / 3.5 | 1.2 | 0 % |

The margin 1.2 equals the ±20 % complexity jitter: the optimum keeps full QoS.

C. Returns (3 seeds); battery-aware = `scenario_utils.BatteryAware` (v_fast 3.9, priority
completion > QoS > reserve):

| case | QoS-EPD | fast 3.9 | battery-aware |
|---|---|---|---|
| 1 km, SOH 1, SOC 1, C 1 | -0.052 | 0.056 | **0.056** (4.3 vs 5.2 min) |
| 1 km, SOH 1, SOC 1, C 3 | -0.605 | -0.720 (14.5 % viol) | **-0.603** |
| 10 km, SOH 0.8, SOC 0.33 | -0.070 | 0.032 | **0.032** (43 vs 52 min) |
| 10 km, SOH 0.8, SOC 0.31 | -0.311 | -0.890 | **-0.311** (drives energy-optimal) |
| 10 km, SOH 0.8, SOC 0.29 | -2.607 | -14.8 (fails) | -2.771 |
| 10 km, SOH 1, SOC 0.23 | **-2.988** | -14.9 (fails) | -6.623 (drops QoS) |
| 35 km, SOH 0.8, SOC 1, C 3 | -15.3 (fails) | -15.3 (fails) | **-5.8** (completes, QoS dropped) |

* Battery-aware >= QoS-EPD except at the knife edge (SOC 0.23: QoS-EPD finishes at
  SOC 0.021, 0.001 above the cutoff; the rule's 1 % safety makes it drop QoS).
* Fast-if-affordable saves ~17 % time; near the range limit it turns a failure into
  a completion. Both need SOC; knowing SOH explicitly matters only in a narrow band near
  the range limit (closed-loop replanning corrects most of an SOH error).
