"""DARO mobile robot environment, revised for independent aging validation.

Training reward uses a dimensionless *battery stress proxy*, never a capacity
fade prediction. SOH remains fixed within a mission. Actual/model-predicted
aging must be assessed by an independently calibrated evaluator on exported
trajectories, including recharge/rest for lifetime studies.

Dependencies: numpy. Gymnasium is optional (spaces/interface if installed).
All motor/power/OCV/thermal coefficients are illustrative, NOT calibrated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import json
import math
from pathlib import Path
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:
    gym = None
    spaces = None
    _Base = object


DEFAULT_FIT_PATH = Path(__file__).resolve().parent / 'lipo_fit_parameters.json'
# Time-domain 2RC (battery_model/fit_pulse_rc.py: 1C pulses, slow polarisation, data-fitted ageing).
PULSE_FIT_PATH = Path(__file__).resolve().parent / 'pulse_rc_fit.json'
ECM_MODEL = 'timedomain'   # 'timedomain' (default) | 'eis' (ms-scale EIS 2RC, legacy)
_ECM_KEYS = ('R0', 'R1', 'R2', 'C1', 'C2')


@dataclass
class ECMParams:
    """Pack-level 2RC ECM parameters.

    By default the SOC-dependent single-cell fit in lipo_fit_parameters.json is
    loaded and scaled to the pack (same convention as battery_model.py):
        Ns = V_nom / cell V,  Np = Q_n[Ah] / cell Ah,
        R_pack = R_cell * Ns / Np,  C_pack = C_cell * Np / Ns.
    Each component is exp(polyval(coeffs, SOC)). The constants Rs/R1/C1/R2/C2
    below are only used as a fallback when no fit file is available.
    """
    Rs: float = 0.035
    R1: float = 0.22
    C1: float = 120.0
    R2: float = 0.019
    C2: float = 1500.0
    eta: float = 1.0  # discharge coulombic efficiency convention
    voltage_min: float = 12.0  # 4S pack, illustrative cutoff
    # Pack: 5000 mAh 14.8 V 4S1P LiPo, 60 C continuous discharge (300 A), EC5
    # connector, recommended charge 1-3 C. The EC5 connector/wiring may limit
    # current below 300 A; the robot draws < 3 A, so this limit never binds.
    current_max: float = 300.0
    # A resistance multiplier can be calibrated independently of capacity SOH.
    resistance_aging_factor: float = 7.5
    # SOC-dependent fit: {'R0': [coeffs], ...} for log(parameter), cell level.
    fit_coeffs: Optional[dict] = None
    r_scale: float = 1.0  # Ns / Np, applied to R (C is divided by it)
    soc_clip: tuple = (0.0, 1.0)  # SOC range used when evaluating the fit
    # Time-domain model: R_j(SOC, SOH) = exp(poly(SOC)) * r_scale * (1 + k_j (1 - SOH)), fixed taus
    # (C_j = tau_j / R_j). k_R0 = resistance_aging_factor. EIS model: R1, R2 do not age.
    model: str = 'eis'
    tau1_s: float = 0.0
    tau2_s: float = 0.0
    k_r1: float = 0.0
    k_r2: float = 0.0
    source: str = 'constant_fallback'

    @classmethod
    def from_fit_json(cls, path=DEFAULT_FIT_PATH, pack_v_nom: float = 14.8,
                      pack_capacity_ah: float = 5.0, **overrides) -> 'ECMParams':
        with open(path) as fh:
            payload = json.load(fh)
        ecm = payload['ecm']
        coeffs = {k: [float(c) for c in ecm[k]['coefficients']] for k in _ECM_KEYS}
        cell = payload.get('cell', {})
        n_s = pack_v_nom / float(cell.get('nominal_voltage_v', 3.7))
        n_p = pack_capacity_ah / float(cell.get('capacity_ah', 1.0))
        r_scale = n_s / n_p
        ref = {k: math.exp(np.polyval(coeffs[k], 0.5)) for k in _ECM_KEYS}
        kw = dict(Rs=ref['R0'] * r_scale, R1=ref['R1'] * r_scale,
                  C1=ref['C1'] / r_scale, R2=ref['R2'] * r_scale,
                  C2=ref['C2'] / r_scale, fit_coeffs=coeffs, r_scale=r_scale,
                  source=f'lipo_fit:{Path(path).name}')
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def from_pulse_json(cls, path=PULSE_FIT_PATH, pack_v_nom: float = 14.8,
                        pack_capacity_ah: float = 5.0, cell_json=DEFAULT_FIT_PATH,
                        **overrides) -> 'ECMParams':
        """Time-domain 2RC identified from the 1C pulses (fit_pulse_rc.py): R_i (resistance at
        1 s, incl. the ms EIS arcs), two slow branches with fixed taus, SOH factors per branch."""
        with open(path) as fh:
            m = json.load(fh)['model']
        cell = {}
        if Path(cell_json).exists():
            with open(cell_json) as fh:
                cell = json.load(fh).get('cell', {})
        n_s = pack_v_nom / float(cell.get('nominal_voltage_v', 3.7))
        n_p = pack_capacity_ah / float(cell.get('capacity_ah', 1.0))
        r_scale = n_s / n_p
        coeffs = {'R0': m['R_i']['coefficients'], 'R1': m['R_1']['coefficients'],
                  'R2': m['R_2']['coefficients']}
        ref = {k: math.exp(np.polyval(c, 0.5)) * r_scale for k, c in coeffs.items()}
        kw = dict(Rs=ref['R0'], R1=ref['R1'], C1=m['tau1'] / ref['R1'], R2=ref['R2'],
                  C2=m['tau2'] / ref['R2'], fit_coeffs=coeffs, r_scale=r_scale,
                  soc_clip=(0.15, 1.0),        # range covered by the pulse data
                  model='timedomain', tau1_s=float(m['tau1']), tau2_s=float(m['tau2']),
                  resistance_aging_factor=float(m['R_i']['k_soh']),
                  k_r1=float(m['R_1']['k_soh']), k_r2=float(m['R_2']['k_soh']),
                  source=f'pulse_fit:{Path(path).name}')
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def default(cls, robot: Optional['RobotParams'] = None, **overrides) -> 'ECMParams':
        """ECM_MODEL fit when available ('timedomain' pulse fit or 'eis' fit), else constant fallback."""
        r = robot or RobotParams()
        kw = dict(pack_v_nom=r.V_nom, pack_capacity_ah=r.Q_n / 3600.0)
        if ECM_MODEL == 'timedomain' and PULSE_FIT_PATH.exists():
            return cls.from_pulse_json(PULSE_FIT_PATH, **kw, **overrides)
        if DEFAULT_FIT_PATH.exists():
            return cls.from_fit_json(DEFAULT_FIT_PATH, **kw, **overrides)
        return cls(**overrides)

    def values(self, soc: float) -> tuple[float, float, float, float, float]:
        """Pack (R0, R1, C1, R2, C2) at the given SOC, before R0 aging."""
        if not self.fit_coeffs:
            return self.Rs, self.R1, self.C1, self.R2, self.C2
        s = float(np.clip(soc, *self.soc_clip))
        k = self.r_scale
        if self.model == 'timedomain':
            r0, r1, r2 = (math.exp(np.polyval(self.fit_coeffs[n], s)) * k for n in ('R0', 'R1', 'R2'))
            return r0, r1, self.tau1_s / r1, r2, self.tau2_s / r2
        v = {n: math.exp(np.polyval(self.fit_coeffs[n], s)) for n in _ECM_KEYS}
        return v['R0'] * k, v['R1'] * k, v['C1'] / k, v['R2'] * k, v['C2'] / k

    def aged(self, soc: float, soh: float) -> tuple[float, float, float, float, float]:
        """(R0, R1, C1, R2, C2) at SOC and SOH: R_j x (1 + k_j (1 - SOH)), time constants kept."""
        r0, r1, c1, r2, c2 = self.values(soc)
        g0 = 1.0 + self.resistance_aging_factor * (1.0 - soh)
        g1 = 1.0 + self.k_r1 * (1.0 - soh)
        g2 = 1.0 + self.k_r2 * (1.0 - soh)
        return r0 * g0, r1 * g1, c1 / g1, r2 * g2, c2 / g2


@dataclass
class StressParams:
    """Dimensionless per-step proxy, NOT ΔSOH or physical degradation."""
    current_weight: float = 1.0
    temperature_weight: float = 1.0
    soc_weight: float = 0.0
    c_rate_ref: float = 1.0
    temp_threshold_c: float = 35.0
    temp_scale_c: float = 10.0
    soc_threshold: float = 0.85
    # [unit_test] C-rate relative to the ACTUAL capacity Q_n * SOH (an aged cell
    # sees a higher C-rate for the same current). False = legacy (rated Q_n).
    c_rate_actual_capacity: bool = True
    # [unit_test] Low-SOC amplification of the current stress: high current at
    # low SOC (steep OCV, high polarisation) is more damaging.
    #   g_i *= 1 + low_soc_gain * ((low_soc_threshold - soc) / low_soc_threshold)^2
    # 0 disables (legacy). Illustrative, not calibrated.
    low_soc_gain: float = 3.0
    low_soc_threshold: float = 0.5


# Discrete CPU DVFS levels (cpufreq scaling_available_frequencies, kHz). The action's
# frequency component is a level INDEX 0..N_FREQ_LEVELS-1; observations report GHz.
CPU_CLOCK_KHZ = (499200, 652800, 806400, 960000, 1113600, 1267200, 1420800,
                 1574400, 1728000, 1881600, 2035200)
CPU_FREQ_GHZ = np.array(CPU_CLOCK_KHZ, dtype=float) / 1e6
N_FREQ_LEVELS = len(CPU_CLOCK_KHZ)


def freq_level(freq_ghz):
    """Index of the lowest DVFS level >= freq_ghz (clipped to the table). Rounding UP
    keeps a frequency that meets QoS meeting it. Accepts scalars or arrays."""
    f = np.asarray(freq_ghz, dtype=float)
    k = np.clip(np.searchsorted(CPU_FREQ_GHZ, f - 1e-9, side='left'), 0, N_FREQ_LEVELS - 1)
    return int(k) if np.ndim(k) == 0 else k


def quantize_freq(freq_ghz):
    """Frequency [GHz] of freq_level(freq_ghz), i.e. the level the CPU actually runs at."""
    out = CPU_FREQ_GHZ[freq_level(freq_ghz)]
    return float(out) if np.ndim(out) == 0 else out


def freq_action(speed: float, freq_ghz: float) -> np.ndarray:
    """Env action [speed, level index] for a desired (speed, frequency in GHz)."""
    return np.array([speed, freq_level(freq_ghz)], dtype=np.float32)


@dataclass
class RobotParams:
    Q_n: float = 5 * 3600.0  # [As], pack capacity in series remains 5 Ah
    V_nom: float = 14.8
    SOH_EOL: float = 0.80
    v_min: float = 0.5
    v_max: float = 5.0
    a_max: float = 0.75  # m/s², illustrative acceleration limit
    f_min: float = float(CPU_FREQ_GHZ[0])    # lowest / highest DVFS level [GHz]
    f_max: float = float(CPU_FREQ_GHZ[-1])
    dt: float = 1.0
    T_amb: float = 25.0  # Celsius
    mission_distance_m: float = 500.0
    max_steps: int = 1200
    n_apps: int = 2
    # QoS target [batch/s] for the fitted throughput model: with the 1.05 margin at the worst-case
    # workload (1.2 c) the QoS frequency spans the DVFS range (C=1: 0.65-1.27 GHz over 1-5 m/s,
    # C=2: 0.81-1.73, C=3: 0.96-1.88 GHz up to 4.3 m/s). Was 15 with the old 1/(1+0.1 v c) model.
    qos_ref: list[float] = field(default_factory=lambda: [25.0, 25.0])
    # Reward (see RobotEnv.reward_terms). E_d is energy per mission metre:
    #   E_d = w_energy * E_t / L   [J/m], so sum(E_d) = w_energy * (mission J/m).
    # w_energy [1/(J/m)] = 3 / e_ref, with e_ref = 5.17 J/m (= 1.44 Wh/km, minimum
    # cell-side energy per metre at C = 1 with the 'measured' compute model, 3.3 m/s;
    # was 3.96 J/m with the legacy model). Then sum(E_d) ~ 3 at the energy-optimal
    # speed for ANY L, and the per-second (myopic) optimum equals the per-mission
    # energy optimum: d/dv [w_U v - w_E P(v)] = 0  <=>  P'(v) = e_ref = P/v.
    # Stress normalised per mission: S_ref = stress_ref_s_per_km * L / 1000.
    # ~15.5 s/km is the stress proxy at the energy-optimal speed (3.3 m/s, C=1,
    # SOH 1, 'measured' compute model; was 8 s/km with the legacy model),
    # so sum(D_d) ~ w_stress per mission there (same scale as E_d / w_energy).
    stress_ref_s_per_km: float = 15.5
    stress_ref_s: float = 1000.0   # legacy (unused by reward_terms)
    # Progress utility: sum(U_t) = w_utility for a completed mission (pure progress,
    # no QoS factor; QoS is penalised separately by Q_d).
    w_utility: float = 2.0
    w_energy: float = 3.0 / 5.17  # [1/(J/m)] = 3 / e_ref  (~0.580; was 3 / 3.96)
    # Battery-stress proxy D_d: OFF in the main reward. It is not calibrated (low-SOC
    # gain is illustrative; C-rate aging is negligible at this pack's ~0.15 C) and it
    # would make the policy slow down at low SOC / SOH even when the charge suffices.
    # Kept as the 'with_stress' ablation (obs_variants.REWARD_VARIANTS).
    w_stress: float = 0.0
    # Time: T_d = w_time * dt / T_ref, T_ref = L / v_ref, so sum(T_d) = w_time * T / T_ref
    # (= w_time at v_ref for any L); sum(E_d) ~ 3 at the energy optimum. Calibrated
    # (unit_test/check_reward_objective.py): with enough charge the reward-optimal speed
    # is 3.6-3.9 m/s for any SOC / SOH (energy optimum 3.0-3.3 m/s; at C = 3 the QoS
    # limit ~3.7 m/s binds). Battery state then matters only through the charge budget
    # (SOC reserve and failure terms).
    w_time: float = 1.0
    v_ref_mps: float = 3.3   # reference speed for T_ref (energy-optimal speed at C = 1)
    w_smooth: float = 0.2  # unused (kept for config compatibility)
    # QoS: Q_d = w_qos * dt / T_ref * (1[violation > 0] + violation). The indicator
    # makes any violation cost >> what it saves in energy or time, so the reward
    # optimum keeps a full margin against the complexity jitter (violation ~ 0).
    w_qos: float = 3.0
    # Battery reserve, terminal: -w_reserve * clip((soc_reserve - SOC_end) / soc_reserve, 0, 1)
    # at the end of every episode (completed or not).
    soc_reserve: float = 0.05
    w_reserve: float = 5.0
    # [unit_test] True: energy = OCV * I * dt (cell side, incl. internal losses).
    # False: legacy load-side V_terminal * I * dt.
    energy_at_cell: bool = True
    # Terminal: +completion_bonus on completion, -failure_penalty otherwise (battery
    # limit / SOC <= 2 % / timeout). failure_penalty > max mission QoS penalty, so
    # finishing with degraded QoS beats not finishing.
    completion_bonus: float = 2.0
    failure_penalty: float = 10.0
    noise_std_frac: float = 0.02  # observation only
    # Battery-state observation:
    #   'true' : obs SOC / SOH are the true values (legacy).
    #   'bms'  : realistic BMS. SOC0 is exact at the start (OCV at rest), then
    #            SOC_bms = SOC0 - integral(I dt) / (eta * Q_n) with the RATED capacity, so it
    #            drifts above the true SOC as the cell ages; SOH is unknown (obs = 1.0).
    #            SOH is only inferable from voltage / current (OCV(SOC, SOH), R0(SOH)).
    battery_obs: str = 'true'
    # Battery realism (all off by default; envs.test_env.make_robot sets them when
    # config['battery_realism'] is true). Sensor errors apply in train AND test mode.
    meas_voltage_noise_v: float = 0.0       # terminal-voltage sensor noise (std) [V]
    meas_current_noise_frac: float = 0.0    # current sensor noise (std, fraction of |I|)
    bms_current_gain_err: float = 0.0       # BMS coulomb-counter gain error ~ U(+-x) per episode
    # Per-episode true-cell variation (controllers only know the nominal ECM / OCV):
    cell_variation: bool = False
    cell_r0_scale: tuple = (0.8, 1.3)       # R0 multiplier ~ U
    cell_ocv_offset_v: float = 0.02         # OCV offset ~ U(+-x) [V]
    cell_aging_shape: tuple = (0.7, 1.3)    # OCV-shape ageing = (1 - SOH) x U, so the OCV
                                            # curve no longer maps one-to-one onto capacity
    # Motor (mechanical) power, steady-speed fit to robot measurements:
    #   P(v) = 1.1625 + 1.14005 v + 0.10893 v^3  [W]
    motor_base_w: float = 1.1625
    motor_linear_w_per_mps: float = 1.14005
    motor_quadratic_w_per_mps2: float = 0.0
    motor_cubic_w_per_mps3: float = 0.10893
    # Acceleration term is NOT part of the fit (illustrative); 0 disables it.
    motor_accel_w_per_mps2: float = 1.5
    # Non-motor power model (see non_motor_power):
    #   'measured' : P_comp(v, c) from the energy-modelling tool (COMP_TABLE_*), which
    #                covers the whole compute board incl. auxiliaries (aux_power_w unused),
    #                plus a DVFS correction for f away from the QoS frequency.
    #   'legacy'   : compute_base_w + compute_dynamic_w (f/2)^3 + aux_power_w.
    compute_model: str = 'measured'
    compute_scale: float = 1.0        # multiplies the measured table (hardware/sensor variation)
    compute_base_w: float = 2.0       # legacy only
    compute_dynamic_w: float = 1.5    # legacy, and DVFS slope in 'measured'
    aux_power_w: float = 2.0          # legacy only
    # Motor winding thermal model (illustrative, NOT calibrated): the hotter the motor,
    # the higher its copper resistance and losses, so fast driving costs extra energy.
    #   P_motor = P_poly(v, a) * (1 + beta * dT),   dT = winding temperature above ambient
    #   C_m d(dT)/dt = h * P_poly + (P_motor - P_poly) - G_m * dT
    # Steady state dT_ss = h P_poly / (G_m - beta P_poly); the extra loss grows ~ P_poly^2
    # (~ v^6): +4 % at 3 m/s, +8 % at 4 m/s, +16 % at 5 m/s (dT_ss 17 / 32 / 62 K).
    # Time constant C_m / G_m = 300 s. motor_thermal_loss = False restores the cold-motor fit.
    motor_thermal_loss: bool = True
    motor_heat_frac: float = 0.3                     # h: share of P_poly dissipated as heat
    motor_temp_coeff_per_k: float = 0.0025           # beta: loss increase per kelvin
    motor_thermal_capacity_j_per_k: float = 45.0     # C_m
    motor_thermal_conductance_w_per_k: float = 0.15  # G_m
    # Simple lumped thermal model; calibration required.
    thermal_capacity_j_per_k: float = 400.0
    thermal_conductance_w_per_k: float = 0.8


# RobotParams fields that reset(options={'plant': ...}) may override (true plant only;
# rule-based baselines keep the nominal calibration).
PLANT_KEYS = ('motor_base_w', 'motor_linear_w_per_mps', 'motor_cubic_w_per_mps3', 'compute_scale')


class BatteryLimitError(ValueError):
    pass


# 4S discharge-voltage fit (battery_model/real_battery_discharge/
# calibrate_battery_voltage_model.py):  V(x) = v0 - a1 x - a3 x^3 - a8 x^8 - a25 x^25,
# x = 1 - SOC. Coefficients [v0, a1, a3, a8, a25].
VOC_FRESH_COEFF = (15.061061, 0.0, 0.995533, 1.18033, 0.881677)
VOC_AGED_COEFF = (15.008879, 0.077644, 1.31415, 0.976887, 0.855835)
VOC_SOH_AGED_REF = 0.85     # SOH assigned to the aged curve (demonstration mapping)
# The fit is a loaded terminal voltage, not a rested OCV: it already contains
# I_fit * R_pack at the (unknown) test current. Set this to I_fit * R_pack [V]
# once the test load is known, so the ECM's own I*R drop is not counted twice.
VOC_LOAD_DROP_V = 0.0


def _voc_curve(soc: float, coeff) -> float:
    x = 1.0 - float(np.clip(soc, 0.0, 1.0))
    v0, a1, a3, a8, a25 = coeff
    return v0 - a1 * x - a3 * x ** 3 - a8 * x ** 8 - a25 * x ** 25


def _voc(soc: float, soh: float = 1.0, V_nom: float = 14.8) -> float:
    """4S open-circuit voltage OCV(SOC, SOH) from the fitted discharge curves.

    SOH interpolates linearly between the fresh (SOH = 1) and aged
    (SOH = VOC_SOH_AGED_REF) curves, clipped outside that range.
    V_fresh: 15.06 V at SOC 1, 14.52 V at SOC 0.25, 12.00 V at SOC 0.
    V_nom is kept for API compatibility (curves are absolute 4S volts).
    """
    w = float(np.clip((1.0 - soh) / (1.0 - VOC_SOH_AGED_REF), 0.0, 1.0))
    v = (1.0 - w) * _voc_curve(soc, VOC_FRESH_COEFF) + w * _voc_curve(soc, VOC_AGED_COEFF)
    return v + VOC_LOAD_DROP_V


def motor_power_cold(speed: float, acceleration: float = 0.0,
                     params: Optional[RobotParams] = None) -> float:
    """Motor power [W] of the steady-speed fit (motor at ambient temperature):
    P = b0 + b1 v + b2 v^2 + b3 v^3 + k_a |a| v.  Accepts scalars or arrays.
    """
    p = params or RobotParams()
    v = np.maximum(0.0, speed)
    out = (p.motor_base_w + p.motor_linear_w_per_mps * v
           + p.motor_quadratic_w_per_mps2 * v ** 2
           + p.motor_cubic_w_per_mps3 * v ** 3
           + p.motor_accel_w_per_mps2 * np.abs(acceleration) * v)
    return float(out) if np.ndim(out) == 0 else out


def motor_temp_rise_ss(p_cold, params: Optional[RobotParams] = None):
    """Steady-state winding temperature above ambient [K] at cold-fit power p_cold [W]."""
    p = params or RobotParams()
    pc = np.maximum(0.0, np.asarray(p_cold, dtype=float))
    den = np.maximum(p.motor_thermal_conductance_w_per_k - p.motor_temp_coeff_per_k * pc,
                     0.2 * p.motor_thermal_conductance_w_per_k)     # no thermal runaway
    out = p.motor_heat_frac * pc / den
    return float(out) if np.ndim(out) == 0 else out


def motor_power(speed: float, acceleration: float = 0.0,
                params: Optional[RobotParams] = None, temp_rise: Optional[float] = None) -> float:
    """Motor electrical power [W] incl. the temperature-dependent loss:
    P = P_cold(v, a) * (1 + beta * dT). temp_rise = winding temperature above ambient
    [K]; None = steady state for this speed (what planners / steady analyses use).
    Accepts scalars or arrays."""
    p = params or RobotParams()
    pc = motor_power_cold(speed, acceleration, p)
    if not p.motor_thermal_loss:
        return pc
    dT = motor_temp_rise_ss(pc, p) if temp_rise is None else temp_rise
    out = np.asarray(pc) * (1.0 + p.motor_temp_coeff_per_k * np.asarray(dT))
    return float(out) if np.ndim(out) == 0 else out


def motor_thermal_factor(speed, params: Optional[RobotParams] = None):
    """Steady-state P_motor / P_cold at constant speed (1 if the thermal loss is off)."""
    p = params or RobotParams()
    pc = motor_power_cold(speed, 0.0, p)
    return np.asarray(motor_power(speed, 0.0, p)) / np.maximum(pc, 1e-9)


def compute_cpu_power(freq_ghz, base_power: float = 4.0, alpha_cpu: float = 1.5):
    """CPU power [W] = base + alpha (f / 2 GHz)^3. Accepts scalars or arrays."""
    out = base_power + alpha_cpu * (np.maximum(0.0, freq_ghz) / 2.0) ** 3
    return float(out) if np.ndim(out) == 0 else out


# Throughput model (single source of truth; baselines use the inverses below), fitted to
# the measured high-complexity curves of the energy-modelling tool
# (figures/data/throughput_high_complexity.csv: DVFS levels 499.2-2035.2 MHz at 1 and 4 m/s,
# endpoints at 2/3/5 m/s; RMSE 0.43 batch/s, max error 0.85 batch/s):
#   s   = v * c / THR_C_REF          workload load: speed x complexity, 1 = 1 m/s at high complexity
#   thr = A - B s^P + (G + D s^R) (f - THR_F_REF),   clipped at THR_MIN
# Linear in f (closed-form inverse), strictly decreasing in s for f <= THR_F_REF (inverse by
# bisection). Throughput saturates in f (0.5 -> 2.04 GHz gains 20 % at 1 m/s, 70 % at 5 m/s,
# high complexity); speed costs throughput mostly at low frequency.
THR_A, THR_B, THR_P = 29.972693795717916, 0.0012670335270440827, 4.862540015188641
THR_G, THR_D, THR_R = 2.2952112821311688, 0.8116839357565258, 1.1262643686746012
THR_F_REF = float(CPU_FREQ_GHZ[-1])   # [GHz] top DVFS level (fit reference)
THR_C_REF = 3.0                       # complexity of the measured curves ('high')
THR_MIN = 0.1                         # [batch/s] floor (model extrapolated far beyond the data)
THR_S_MAX = 20.0                      # bisection bracket for s (thr(s_max) is at the floor)


# Measured computing power [W] at the QoS-limited CPU frequency, ESTIMATED from the
# energy-modelling tool figure (energy per metre vs speed, low/moderate/high
# complexity): P_comp = v * (E_total/d - P_motor(v)/v), with P_motor the steady
# motor fit above (it reproduces the tool's "Mechanical" bars). Read off a plot:
# replace with the tool's raw data when available. Steps (e.g. 8.3 -> 9.0 -> 10.0 W)
# look like DVFS levels.
COMP_TABLE_V = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
COMP_TABLE_C = np.array([1.0, 2.0, 3.0])
COMP_TABLE_W = np.array([
    [7.40, 7.64, 7.65, 7.65, 7.86, 7.87, 8.06, 8.31, 9.03, 9.02],     # low complexity
    [7.65, 7.65, 7.86, 8.09, 8.09, 8.32, 9.04, 9.99, 10.74, 11.52],   # moderate
    [9.54, 9.57, 9.57, 9.59, 10.09, 10.78, 10.79, 10.83, 11.68, 11.57],  # high
])


def measured_compute_power(speed, freq_ghz, complexity, p: 'RobotParams'):
    """Compute-board power [W]: table P_comp(v, c) at the QoS frequency f_q(v, c),
    plus compute_dynamic_w * ((f/2)^3 - (f_q/2)^3) for running above/below f_q.

    v and c are clipped to the table range (no extrapolation). Accepts arrays in
    speed / freq_ghz (complexity scalar).
    """
    v = np.clip(np.asarray(speed, dtype=float), COMP_TABLE_V[0], COMP_TABLE_V[-1])
    c = float(np.clip(complexity, COMP_TABLE_C[0], COMP_TABLE_C[-1]))
    rows = np.array([np.interp(v, COMP_TABLE_V, r) for r in COMP_TABLE_W])
    k = min(int(np.searchsorted(COMP_TABLE_C, c, side='right')) - 1, len(COMP_TABLE_C) - 2)
    w = (c - COMP_TABLE_C[k]) / (COMP_TABLE_C[k + 1] - COMP_TABLE_C[k])
    p_tab = (1.0 - w) * rows[k] + w * rows[k + 1]
    f_q = quantize_freq(throughput_freq_required(np.asarray(speed, dtype=float), max(p.qos_ref), c))
    f = np.maximum(0.0, np.asarray(freq_ghz, dtype=float))
    out = p.compute_scale * p_tab + p.compute_dynamic_w * ((f / 2.0) ** 3 - (f_q / 2.0) ** 3)
    return float(out) if np.ndim(out) == 0 else out


def non_motor_power(speed, freq_ghz, complexity, p: 'RobotParams'):
    """All non-motor electrical power [W] (compute + auxiliaries) for p.compute_model."""
    if p.compute_model == 'measured':
        return measured_compute_power(speed, freq_ghz, complexity, p)
    if p.compute_model == 'legacy':
        return compute_cpu_power(freq_ghz, p.compute_base_w, p.compute_dynamic_w) + p.aux_power_w
    raise ValueError(f'unknown compute_model {p.compute_model!r}')


def _thr_raw(s, f):
    """Unclipped throughput at workload load s and frequency f (f capped at THR_F_REF)."""
    s = np.maximum(0.0, s)
    df = np.minimum(f, THR_F_REF) - THR_F_REF
    return THR_A - THR_B * s ** THR_P + (THR_G + THR_D * s ** THR_R) * df


def _thr_load(freq_ghz, thr):
    """Workload load s >= 0 at which frequency f gives throughput thr (bisection; the model
    is strictly decreasing in s). 0 if thr >= thr(s=0), THR_S_MAX if below the bracket."""
    f, q = np.broadcast_arrays(np.asarray(freq_ghz, dtype=float), np.asarray(thr, dtype=float))
    lo, hi = np.zeros(f.shape), np.full(f.shape, THR_S_MAX)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        above = _thr_raw(mid, f) >= q
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    return 0.5 * (lo + hi)


def compute_throughput(speed, freq_ghz, complexity=1.0):
    s = np.asarray(speed, dtype=float) * complexity / THR_C_REF
    out = np.maximum(THR_MIN, _thr_raw(s, np.asarray(freq_ghz, dtype=float)))
    return float(out) if np.ndim(out) == 0 else out


def throughput_freq_required(speed, thr_target, complexity=1.0):
    """CPU frequency [GHz] giving throughput thr_target (inverse in f; unclipped, may exceed
    f_max when the target is unreachable)."""
    s = np.maximum(0.0, np.asarray(speed, dtype=float) * complexity / THR_C_REF)
    out = THR_F_REF + (thr_target - THR_A + THR_B * s ** THR_P) / (THR_G + THR_D * s ** THR_R)
    return float(out) if np.ndim(out) == 0 else out


def throughput_max_speed(freq_ghz: float, thr_target: float, complexity: float = 1.0) -> float:
    """Largest speed [m/s] at which freq_ghz still reaches thr_target.

    Returns 0 if thr_target is unreachable even at standstill, inf if c <= 0.
    """
    if _thr_raw(0.0, freq_ghz) < thr_target:
        return 0.0
    if complexity <= 0.0:
        return float('inf')
    return float(_thr_load(freq_ghz, thr_target)) * THR_C_REF / complexity


def throughput_complexity(speed: float, freq_ghz: float, thr: float) -> Optional[float]:
    """Complexity implied by an observed throughput (inverse in c).

    None if not identifiable (speed ~ 0, or throughput at the floor).
    """
    if speed <= 1e-6 or thr <= THR_MIN + 1e-9:
        return None
    return float(_thr_load(freq_ghz, thr)) * THR_C_REF / speed

class ECMBattery:
    """Constant-power 2RC ECM with exact first-order RC transitions."""
    def __init__(self, ecm: ECMParams, robot: RobotParams,
                 soc_init: float = 1.0, soh_init: float = 1.0):
        self.ecm, self.robot = ecm, robot
        self.soc, self.soh = float(soc_init), float(soh_init)
        self.V1, self.V2 = 0.0, 0.0
        self.temperature_c = float(robot.T_amb)
        self.Q_act = self.robot.Q_n * self.soh
        self.current_a = 0.0
        # True-cell deviations from the nominal model (set_cell; identity by default).
        self.r0_scale, self.ocv_offset_v, self.aging_shape = 1.0, 0.0, 1.0
        self.voltage_v = self._ocv()
        self.ocv_v = self.voltage_v

    def set_cell(self, r0_scale: float = 1.0, ocv_offset_v: float = 0.0, aging_shape: float = 1.0):
        self.r0_scale, self.ocv_offset_v = float(r0_scale), float(ocv_offset_v)
        self.aging_shape = float(aging_shape)
        self.voltage_v = self.ocv_v = self._ocv()

    def _ocv(self) -> float:
        soh_shape = 1.0 - self.aging_shape * (1.0 - self.soh)
        return _voc(self.soc, soh_shape, self.robot.V_nom) + self.ocv_offset_v

    def update_soh(self, soh: float):
        self.soh = float(soh)
        self.Q_act = self.robot.Q_n * self.soh

    def _resistance(self, r0_base: Optional[float] = None) -> float:
        r0 = self.ecm.values(self.soc)[0] if r0_base is None else r0_base
        return self.r0_scale * r0 * (1.0 + self.ecm.resistance_aging_factor * (1.0 - self.soh))

    def step(self, power_w: float, dt: Optional[float] = None,
             **_legacy_kwargs) -> tuple[float, float, float]:
        dt = self.robot.dt if dt is None else float(dt)
        if power_w < 0 or dt <= 0:
            raise ValueError('This environment supports positive discharge power and dt > 0.')
        e = self.ecm
        r0_base, r1, c1, r2, c2 = e.values(self.soc)
        r0 = self._resistance(r0_base)
        if e.k_r1 or e.k_r2:         # time-domain model: the slow branches age too (tau kept)
            _, r1, c1, r2, c2 = e.aged(self.soc, self.soh)
        ocv = self._ocv()
        u = ocv - self.V1 - self.V2
        disc = u * u - 4.0 * r0 * power_w
        if disc < 0:
            raise BatteryLimitError('Requested power violates the constant-power ECM limit.')
        # Rationalized expression avoids cancellation for small loads.
        current = 2.0 * power_w / (u + math.sqrt(disc)) if power_w else 0.0
        v_terminal = u - current * r0
        if current > e.current_max or v_terminal < e.voltage_min:
            raise BatteryLimitError('Current or voltage safety limit exceeded.')
        if self.soc - current * dt / (e.eta * self.Q_act) < 0:
            raise BatteryLimitError('Insufficient remaining charge.')
        def branch(vold, r, c):
            if r <= 0 or c <= 0:
                raise ValueError('RC branch R and C must be positive.')
            a = math.exp(-dt / (r * c))
            return a * vold + (1.0 - a) * r * current
        new_v1 = branch(self.V1, r1, c1)
        new_v2 = branch(self.V2, r2, c2)
        new_soc = max(0.0, self.soc - current * dt / (e.eta * self.Q_act))
        # Lumped Joule heating; RC dissipation is positive, not double counted.
        heat_w = (current ** 2 * r0 + new_v1 ** 2 / r1 + new_v2 ** 2 / r2)
        self.temperature_c += dt * (heat_w - self.robot.thermal_conductance_w_per_k
                                  * (self.temperature_c - self.robot.T_amb)) / self.robot.thermal_capacity_j_per_k
        self.V1, self.V2, self.soc = new_v1, new_v2, new_soc
        self.current_a, self.voltage_v = current, v_terminal
        self.ocv_v = ocv   # [unit_test] OCV used this step (cell-side energy)
        return current, v_terminal, new_soc

class BatteryStress:
    """Instantaneous stress proxy for training, not an aging law."""
    def __init__(self, p: StressParams, robot: RobotParams):
        self.p, self.robot = p, robot

    def step(self, current_a: float, temperature_c: float, soc: float,
             dt: float, soh: float = 1.0) -> float:
        p = self.p
        q_ah = self.robot.Q_n / 3600.0
        if p.c_rate_actual_capacity:
            q_ah *= max(float(soh), 1e-3)
        c_rate = abs(current_a) / q_ah
        g_i = (c_rate / p.c_rate_ref) ** 2
        if p.low_soc_gain > 0.0 and soc < p.low_soc_threshold:
            depth = (p.low_soc_threshold - soc) / max(p.low_soc_threshold, 1e-6)
            g_i *= 1.0 + p.low_soc_gain * depth ** 2
        g_t = (max(0.0, temperature_c - p.temp_threshold_c) / p.temp_scale_c) ** 2
        g_s = (max(0.0, soc - p.soc_threshold) / max(1e-6, 1.0 - p.soc_threshold)) ** 2
        return dt * (p.current_weight * g_i + p.temperature_weight * g_t
                     + p.soc_weight * g_s)


class RobotEnv(_Base):
    """Training/test mission env. Evaluation SOH never uses training proxy.

    API: Gymnasium reset -> (obs, info), step -> (obs,reward,terminated,truncated,info).
    For old trainers call reset_legacy() and step_legacy().
    """
    metadata = {'render_modes': []}

    def __init__(self, robot: Optional[RobotParams] = None,
                 ecm: Optional[ECMParams] = None,
                 deg=None,  # old parameter accepted but intentionally not used
                 stress: Optional[StressParams] = None,
                 soc_init: float = 1.0, soh_init: float = 1.0,
                 complexity: float = 1.0, mode: str = 'train',
                 max_steps: Optional[int] = None,
                 complexity_jitter: Optional[float] = None):
        super().__init__()
        self.robot = robot or RobotParams()
        self.ecm_params = ecm or ECMParams.default(self.robot)
        self.stress_params = stress or StressParams()
        self.soc_init, self.soh_init = float(soc_init), float(soh_init)
        self.complexity, self.mode = float(complexity), mode
        # Per-step multiplicative complexity disturbance U(1-j, 1+j); seeded by
        # reset(seed=...). Default: 0.2 in 'train', 0 otherwise.
        self.complexity_jitter = (0.05 if mode == 'train' else 0.0) \
            if complexity_jitter is None else float(complexity_jitter)
        if not 0.0 <= self.complexity_jitter < 1.0:
            raise ValueError('complexity_jitter must be in [0, 1).')
        if max_steps is not None:
            self.robot.max_steps = int(max_steps)
        if self.robot.dt <= 0 or self.robot.mission_distance_m <= 0:
            raise ValueError('dt and mission_distance_m must be positive.')
        if len(self.robot.qos_ref) != self.robot.n_apps:
            raise ValueError('qos_ref length must equal n_apps.')
        self.n_apps = self.robot.n_apps
        # Action [speed (m/s), CPU DVFS level index 0..N_FREQ_LEVELS-1]. Continuous agents
        # (SAC) use action_low/high and the index is rounded in step().
        self.action_low = np.array([self.robot.v_min, 0.0], dtype=np.float32)
        self.action_high = np.array([self.robot.v_max, N_FREQ_LEVELS - 1.0], dtype=np.float32)
        # [v, f, SOC, SOH, temperature, terminal voltage, distance remaining,
        #  last current, app throughputs..., app QoS targets...,
        #  mission length L, fraction of the mission remaining]
        # L is needed because the reward is normalised per mission (progress 1/L per
        # metre, time 1/T_ref per second, T_ref = L / v_ref); mission features are
        # appended last so the indices used by the rule baselines are unchanged.
        self.obs_dim = 10 + 2 * self.n_apps
        self.act_dim = 2
        if spaces is not None:
            self.action_space = spaces.Tuple((
                spaces.Box(self.action_low[:1], self.action_high[:1], dtype=np.float32),
                spaces.Discrete(N_FREQ_LEVELS)))
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(self.obs_dim,), dtype=np.float32)
        self.np_random = np.random.default_rng()
        # Plant parameters restored by reset(options={'plant': {...}}) (training-time
        # model-mismatch randomisation); unspecified keys go back to these values.
        self._nominal_plant = {k: getattr(self.robot, k) for k in PLANT_KEYS}
        # Mid-mission complexity switching (reset option): every step, with probability
        # dt / period, the nominal complexity is re-drawn from complexity_levels.
        self.complexity_switch_period_s = 0.0
        self.complexity_levels: tuple = ()
        # Deterministic schedule (reset option 'complexity_schedule': (period_s, [levels])):
        # complexity = levels[k % len(levels)] during [k period, (k + 1) period).
        self.complexity_schedule: tuple = ()
        self.complexity_schedule_period_s = 0.0
        self._reset_internal()

    def _reset_internal(self):
        self.battery = ECMBattery(self.ecm_params, self.robot, self.soc_init, self.soh_init)
        if getattr(self, '_cell', None):
            self.battery.set_cell(**self._cell)
        self.stress_model = BatteryStress(self.stress_params, self.robot)
        self.v_obs = 0.0
        self.f_obs = self.robot.f_min
        self.motor_temp_rise_k = 0.0     # winding temperature above ambient (cold start)
        self.distance_m = 0.0
        self.step_count = 0
        self.thr = [compute_throughput(0.0, self.f_obs, self.complexity)] * self.n_apps
        self.total_energy_j = 0.0
        self.total_stress = 0.0
        self.charge_drawn_as = 0.0   # BMS coulomb counter [As]
        self.trajectory: list[dict] = []
        self._finished = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,
              soc_init: Optional[float] = None, soh_init: Optional[float] = None,
              complexity: Optional[float] = None):
        if gym is not None:
            super().reset(seed=seed)
        elif seed is not None:
            self.np_random = np.random.default_rng(seed)
        p, r = self.robot, self.np_random
        self._cell = {}
        if p.cell_variation:   # drawn from the seeded stream: same cell for every policy
            self._cell = dict(r0_scale=float(r.uniform(*p.cell_r0_scale)),
                              ocv_offset_v=float(r.uniform(-p.cell_ocv_offset_v, p.cell_ocv_offset_v)),
                              aging_shape=float(r.uniform(*p.cell_aging_shape)))
        self._i_gain = (1.0 + float(r.uniform(-p.bms_current_gain_err, p.bms_current_gain_err))
                        if p.bms_current_gain_err > 0 else 1.0)
        if options and 'cell' in options:
            self._cell = dict(options['cell'] or {})
        if options:
            soc_init = options.get('soc_init', soc_init)
            soh_init = options.get('soh_init', soh_init)
            complexity = options.get('complexity', complexity)
            if 'plant' in options:
                plant = dict(options['plant'] or {})
                unknown = set(plant) - set(PLANT_KEYS)
                if unknown:
                    raise ValueError(f'unknown plant keys {sorted(unknown)}; use {PLANT_KEYS}')
                for k, v0 in self._nominal_plant.items():
                    setattr(self.robot, k, float(plant.get(k, v0)))
            if 'complexity_switch_period_s' in options:
                period = options['complexity_switch_period_s']
                self.complexity_switch_period_s = float(period or 0.0)
                self.complexity_levels = tuple(float(x) for x in options.get('complexity_levels', ()))
                if self.complexity_switch_period_s > 0 and not self.complexity_levels:
                    raise ValueError('complexity_switch_period_s needs complexity_levels.')
            if 'complexity_schedule' in options:
                sched = options['complexity_schedule']
                if sched:
                    self.complexity_schedule_period_s = float(sched[0])
                    self.complexity_schedule = tuple(float(x) for x in sched[1])
                    complexity = self.complexity_schedule[0]
                else:
                    self.complexity_schedule, self.complexity_schedule_period_s = (), 0.0
            if 'mission_distance_m' in options:
                d = float(options['mission_distance_m'])
                if not (math.isfinite(d) and d > 0):
                    raise ValueError('mission_distance_m must be positive.')
                self.robot.mission_distance_m = d
        if soc_init is not None:
            self.soc_init = float(soc_init)
        if soh_init is not None:
            self.soh_init = float(soh_init)
        if complexity is not None:
            self.complexity = float(complexity)
        if not (0.0 < self.soc_init <= 1.0 and 0.0 < self.soh_init <= 1.0):
            raise ValueError('Initial SOC/SOH must be in (0,1].')
        self._reset_internal()
        return self._get_obs(), {'initial_soh': self.soh_init}

    def _get_obs(self):
        vals = [self.v_obs, self.f_obs, self.battery.soc,
                self.battery.soh, self.battery.temperature_c,
                self.battery.voltage_v,
                max(0.0, self.robot.mission_distance_m - self.distance_m),
                self.battery.current_a] + list(self.thr) + list(self.robot.qos_ref)
        L = max(float(self.robot.mission_distance_m), 1e-9)
        vals += [L, max(0.0, L - self.distance_m) / L]
        if self.robot.battery_obs == 'bms':
            vals[2] = float(np.clip(self.soc_init - self.charge_drawn_as
                                    / (self.ecm_params.eta * self.robot.Q_n), 0.0, 1.0))
            vals[3] = 1.0   # SOH unknown: nominal
        elif self.robot.battery_obs != 'true':
            raise ValueError(f"battery_obs must be 'true' or 'bms', got {self.robot.battery_obs!r}")
        if self.robot.meas_voltage_noise_v > 0:
            vals[5] += float(self.np_random.normal(0.0, self.robot.meas_voltage_noise_v))
        if self.robot.meas_current_noise_frac > 0:
            vals[7] += float(self.np_random.normal(0.0, self.robot.meas_current_noise_frac
                                                   * max(abs(vals[7]), 0.1)))
        arr = np.asarray(vals, dtype=np.float32)
        if self.mode == 'train' and self.robot.noise_std_frac > 0:
            # Sensor noise affects only agent observation, never dynamics/reward.
            std = self.robot.noise_std_frac
            arr[2] = np.clip(arr[2] + self.np_random.normal(0, std * 0.1), 0, 1)
            arr[5] += self.np_random.normal(0, std * max(1.0, arr[5]))
            arr[7] += self.np_random.normal(0, std * max(1.0, abs(arr[7])))
        return arr

    def obs_normalization(self, max_distance_m: Optional[float] = None):
        """[unit_test] Fixed (offset, scale) so (obs - offset) / scale is O(1).

        Physical ranges, not running statistics: deterministic, identical for
        every seed/variant, and stored with the checkpoint.
        """
        p = self.robot
        d = float(max_distance_m or p.mission_distance_m)
        q = np.asarray(p.qos_ref, dtype=np.float32)
        offset = [0.0, 0.0, 0.5, 0.9, p.T_amb, 13.5, 0.0, 0.0] + list(q) + list(q) + [0.0, 0.0]
        scale = [p.v_max, p.f_max, 0.5, 0.1, 10.0, 1.5, d, 2.0] + list(q) + list(q) + [d, 1.0]
        return (np.asarray(offset, dtype=np.float32), np.asarray(scale, dtype=np.float32))

    def reward_terms(self, energy_j: float, stress: float, delta_distance_m: float,
                     qos_violation: float, dt_s: Optional[float] = None) -> dict:
        """Per-step reward terms; reward = U_t - E_d - D_d - T_d - Q_d (+ terminal R_end).

        All terms are normalised per mission (length L, reference time T_ref = L / v_ref),
        so their mission sums do not depend on L.
        """
        p = self.robot
        L = p.mission_distance_m
        dt_s = p.dt if dt_s is None else float(dt_s)
        T_ref = L / p.v_ref_mps

        # E_d: energy per mission metre [J/m], weighted by w_energy [1/(J/m)]
        E_d = p.w_energy * energy_j / L

        # D_d: Normalized battery stress
        S_ref = p.stress_ref_s_per_km * L / 1000.0
        D_d = p.w_stress * stress / max(S_ref, 1e-9)

        # U_t: physical progress
        U_t = p.w_utility * delta_distance_m / L

        # T_d: elapsed time
        T_d = p.w_time * dt_s / T_ref

        # Q_d: QoS shortfall (indicator + magnitude) per elapsed time
        viol = float(np.clip(qos_violation, 0.0, 1.0))
        Q_d = p.w_qos * dt_s / T_ref * ((1.0 if viol > 0.0 else 0.0) + viol)

        return dict(E_d=float(E_d), D_d=float(D_d), U_t=float(U_t), T_d=float(T_d), Q_d=float(Q_d))

    def terminal_reward(self, completed: bool, soc_end: float) -> float:
        """R_end: completion bonus / failure penalty and the SOC-reserve penalty."""
        p = self.robot
        r = p.completion_bonus if completed else -p.failure_penalty
        if p.soc_reserve > 0.0:
            short = (p.soc_reserve - soc_end) / p.soc_reserve
            r -= p.w_reserve * float(np.clip(short, 0.0, 1.0))
        return float(r)

    def step(self, action: Sequence[float], ext_Pe: Optional[float] = None):
        if self._finished:
            raise RuntimeError('Episode finished; call reset() before step().')
        if isinstance(action, tuple) and len(action) == 2:      # action_space.sample()
            action = [float(np.asarray(action[0]).reshape(-1)[0]), float(action[1])]
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError('Action must be a finite pair [speed, CPU DVFS level index].')
        p = self.robot
        v_cmd = float(np.clip(action[0], p.v_min, p.v_max))
        f_cmd = float(CPU_FREQ_GHZ[int(np.clip(np.rint(action[1]), 0, N_FREQ_LEVELS - 1))])
        delta_v = np.clip(v_cmd - self.v_obs, -p.a_max * p.dt, p.a_max * p.dt)
        v_new = max(0.0, self.v_obs + float(delta_v))
        accel = float(delta_v) / p.dt
        remaining = p.mission_distance_m - self.distance_m
        delta_distance = min(remaining, v_new * p.dt)
        active_dt = (delta_distance / v_new) if v_new > 0 else p.dt
        # Workload complexity this step (drawn first: drives compute power and throughput).
        if (self.complexity_switch_period_s > 0
                and self.np_random.random() < p.dt / self.complexity_switch_period_s):
            self.complexity = float(self.np_random.choice(self.complexity_levels))
        if self.complexity_schedule:
            k = int(self.step_count * p.dt // self.complexity_schedule_period_s)
            self.complexity = self.complexity_schedule[k % len(self.complexity_schedule)]
        c = self.complexity
        if self.complexity_jitter > 0:
            j = self.complexity_jitter
            c *= float(self.np_random.uniform(1.0 - j, 1.0 + j))
        motor_cold_w = motor_power_cold(v_new, accel, p)
        motor_w = motor_power(v_new, accel, p, temp_rise=self.motor_temp_rise_k)
        # compute_w = all non-motor power (compute board + auxiliaries for 'measured').
        compute_w = non_motor_power(v_new, f_cmd, c, p)
        aux_w = p.aux_power_w if p.compute_model == 'legacy' else 0.0
        if ext_Pe is not None:
            if not math.isfinite(ext_Pe) or ext_Pe < 0:
                raise ValueError('External compute electrical power must be finite and nonnegative.')
            compute_w = float(ext_Pe) + aux_w  # external compute/board power, not total pack power
        total_w = motor_w + compute_w
        reason = None
        try:
            current, voltage, soc = self.battery.step(total_w, dt=active_dt)
            if p.energy_at_cell:
                # [unit_test] Chemical energy drawn from the cell: OCV * I * dt.
                # Includes R0 (SOH-aged) and RC losses, which the load-side
                # V_terminal * I * dt misses, so SOC/SOH now affect energy.
                energy_j = current * self.battery.ocv_v * active_dt
            else:
                energy_j = current * voltage * active_dt
            stress = self.stress_model.step(current, self.battery.temperature_c, soc,
                                            active_dt, soh=self.battery.soh)
            self.distance_m += delta_distance
            self.total_energy_j += energy_j
            self.total_stress += stress
            self.charge_drawn_as += current * getattr(self, '_i_gain', 1.0) * active_dt
        except BatteryLimitError as exc:
            reason = str(exc)
            current, voltage, soc = self.battery.current_a, self.battery.voltage_v, self.battery.soc
            energy_j, stress, delta_distance = 0.0, 0.0, 0.0
        if p.motor_thermal_loss and reason is None:   # winding heats with the losses
            heat_w = p.motor_heat_frac * motor_cold_w + (motor_w - motor_cold_w)
            self.motor_temp_rise_k += active_dt * (
                heat_w - p.motor_thermal_conductance_w_per_k * self.motor_temp_rise_k
            ) / p.motor_thermal_capacity_j_per_k
        self.v_obs, self.f_obs = v_new, f_cmd
        self.thr = [compute_throughput(v_new, f_cmd, c)] * self.n_apps
        qos_ratio = [self.thr[i] / p.qos_ref[i] for i in range(self.n_apps)]
        qos_violation = max(0.0, 1.0 - min(qos_ratio))
        self.step_count += 1
        completed = self.distance_m >= p.mission_distance_m - 1e-9
        low_soc = self.battery.soc <= 0.02
        terminated = bool(completed or low_soc or reason)
        truncated = bool(self.step_count >= p.max_steps and not terminated)
        failed = bool((terminated and not completed) or truncated)

        # Compute reward terms and return the Gymnasium API tuple.
        terms = self.reward_terms(energy_j, stress, delta_distance, qos_violation, active_dt)
        R_end = self.terminal_reward(completed, self.battery.soc) if (terminated or truncated) else 0.0
        reward = (terms['U_t'] - terms['E_d'] - terms['D_d'] - terms['T_d'] - terms['Q_d']
                  + R_end)

        self._finished = terminated or truncated
        info = dict(t_s=self.step_count * p.dt, dt_s=active_dt, speed_mps=v_new, acceleration_mps2=accel,
                    cpu_ghz=f_cmd, distance_m=self.distance_m, delta_distance_m=delta_distance,
                    motor_power_w=motor_w, compute_power_w=compute_w - aux_w,
                    aux_power_w=aux_w, power_w=total_w, current_a=current,
                    voltage_v=voltage, soc=soc, soh=self.battery.soh,
                    temperature_c=self.battery.temperature_c,
                    motor_temp_c=p.T_amb + self.motor_temp_rise_k,
                    energy_j=energy_j, total_energy_j=self.total_energy_j,
                    stress_proxy=stress, total_stress_proxy=self.total_stress,
                    throughput=list(self.thr), complexity=c, qos_violation=qos_violation,
                    **terms, R_end=R_end, reward=float(reward),
                    completed=completed, failed=failed, battery_limit=reason,
                    terminated=terminated, truncated=truncated)
        # Save actual physical trajectories (not noisy agent observations).
        self.trajectory.append(info.copy())
        return self._get_obs(), float(reward), terminated, truncated, info


def tight_soc_init(plant: dict, soh: float, c: float, distance_m: float, k: float,
                   soc_reserve: Optional[float] = None, loss_factor: float = 1.05) -> float:
    """SOC0 whose usable energy above the SOC reserve is k * the energy-optimal
    (QoS-feasible, jitter-safe) mission energy of the given plant
    (dict of PLANT_KEYS overrides, {} = nominal). Used for charge-limited
    training episodes and test scenarios. loss_factor covers what the load-side
    model misses (cell losses, QoS / jitter margin, start-up), calibrated so that
    QoS-EPD with k = 1 ends at ~ the reserve."""
    robot = RobotParams()
    for key, val in plant.items():
        setattr(robot, key, val)
    v = np.linspace(1.0, robot.v_max, 81)
    f = quantize_freq(throughput_freq_required(v, max(robot.qos_ref), 1.2 * c))
    ok = throughput_freq_required(v, max(robot.qos_ref), 1.05 * c) <= robot.f_max
    e = (motor_power(v, 0.0, robot) + non_motor_power(v, f, c, robot)) / v
    e_need = loss_factor * k * distance_m * float(np.min(np.where(ok, e, np.inf)))
    floor = robot.soc_reserve if soc_reserve is None else float(soc_reserve)
    s = np.linspace(floor, 1.0, 501)
    ocv = np.array([_voc(x, soh) for x in s])
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (ocv[1:] + ocv[:-1]) * np.diff(s))]) * robot.Q_n * soh
    return float(np.clip(np.interp(e_need, cum, s), 0.03, 1.0))
