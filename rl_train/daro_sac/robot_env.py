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
    current_max: float = 30.0  # illustrative; replace with datasheet/measurement
    # A resistance multiplier can be calibrated independently of capacity SOH.
    resistance_aging_factor: float = 0.0
    # SOC-dependent fit: {'R0': [coeffs], ...} for log(parameter), cell level.
    fit_coeffs: Optional[dict] = None
    r_scale: float = 1.0  # Ns / Np, applied to R (C is divided by it)
    soc_clip: tuple = (0.0, 1.0)  # SOC range used when evaluating the fit
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
    def default(cls, robot: Optional['RobotParams'] = None, **overrides) -> 'ECMParams':
        """Fit-file parameters when available, otherwise constant fallback."""
        if DEFAULT_FIT_PATH.exists():
            r = robot or RobotParams()
            return cls.from_fit_json(DEFAULT_FIT_PATH, pack_v_nom=r.V_nom,
                                     pack_capacity_ah=r.Q_n / 3600.0, **overrides)
        return cls(**overrides)

    def values(self, soc: float) -> tuple[float, float, float, float, float]:
        """Pack (R0, R1, C1, R2, C2) at the given SOC, before R0 aging."""
        if not self.fit_coeffs:
            return self.Rs, self.R1, self.C1, self.R2, self.C2
        s = float(np.clip(soc, *self.soc_clip))
        v = {k: math.exp(np.polyval(self.fit_coeffs[k], s)) for k in _ECM_KEYS}
        k = self.r_scale
        return v['R0'] * k, v['R1'] * k, v['C1'] / k, v['R2'] * k, v['C2'] / k


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


@dataclass
class RobotParams:
    Q_n: float = 5 * 3600.0  # [As], pack capacity in series remains 5 Ah
    V_nom: float = 14.8
    SOH_EOL: float = 0.80
    v_min: float = 0.5
    v_max: float = 5.0
    a_max: float = 0.75  # m/s², illustrative acceleration limit
    f_min: float = 0.499
    f_max: float = 2.0
    dt: float = 1.0
    T_amb: float = 25.0  # Celsius
    mission_distance_m: float = 500.0
    max_steps: int = 1200
    n_apps: int = 2
    qos_ref: list[float] = field(default_factory=lambda: [15.0, 15.0])
    # Mission energy objective: scale each step by constant reference energy.
    energy_ref_j: float = 20000.0
    stress_ref_s: float = 1000.0
    w_energy: float = 1.0
    w_stress: float = 1.0
    w_time: float = 0.15  # avoid trivial crawl policies
    w_qos: float = 1.0
    completion_bonus: float = 2.0
    failure_penalty: float = 3.0
    noise_std_frac: float = 0.02  # observation only
    # Illustrative power model coefficients; fit to robot measurements.
    motor_base_w: float = 2.0
    motor_linear_w_per_mps: float = 2.5
    motor_quadratic_w_per_mps2: float = 1.8
    motor_accel_w_per_mps2: float = 1.5
    compute_base_w: float = 2.0
    compute_dynamic_w: float = 1.5
    aux_power_w: float = 2.0
    # Simple lumped thermal model; calibration required.
    thermal_capacity_j_per_k: float = 400.0
    thermal_conductance_w_per_k: float = 0.8


class BatteryLimitError(ValueError):
    pass


def _voc(soc: float, soh: float = 1.0, V_nom: float = 14.8) -> float:
    """Illustrative 4S OCV(SOC), independent of SOH by default.

    Replace with measured rested OCV-SOC curve. No artificial SOH scaling.
    """
    s = float(np.clip(soc, 0.0, 1.0))
    return 12.8 + 3.8 * s + 0.2 * s ** 8


def motor_power(speed: float, acceleration: float = 0.0,
                params: Optional[RobotParams] = None) -> float:
    p = params or RobotParams()
    v = max(0.0, speed)
    return (p.motor_base_w + p.motor_linear_w_per_mps * v
            + p.motor_quadratic_w_per_mps2 * v * v
            + p.motor_accel_w_per_mps2 * abs(acceleration) * v)


def compute_cpu_power(freq_ghz: float, base_power: float = 2.0,
                      alpha_cpu: float = 1.5) -> float:
    return base_power + alpha_cpu * (max(0.0, freq_ghz) / 2.0) ** 3


def compute_throughput(speed: float, freq_ghz: float,
                       complexity: float = 1.0) -> float:
    return 25.0 * (freq_ghz / 2.0) / (1.0 + 0.1 * speed * complexity)

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
        self.voltage_v = _voc(self.soc, self.soh, robot.V_nom)

    def update_soh(self, soh: float):
        self.soh = float(soh)
        self.Q_act = self.robot.Q_n * self.soh

    def _resistance(self, r0_base: Optional[float] = None) -> float:
        r0 = self.ecm.values(self.soc)[0] if r0_base is None else r0_base
        return r0 * (1.0 + self.ecm.resistance_aging_factor * (1.0 - self.soh))

    def step(self, power_w: float, dt: Optional[float] = None,
             **_legacy_kwargs) -> tuple[float, float, float]:
        dt = self.robot.dt if dt is None else float(dt)
        if power_w < 0 or dt <= 0:
            raise ValueError('This environment supports positive discharge power and dt > 0.')
        e = self.ecm
        r0_base, r1, c1, r2, c2 = e.values(self.soc)
        r0 = self._resistance(r0_base)
        u = _voc(self.soc, self.soh, self.robot.V_nom) - self.V1 - self.V2
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
        return current, v_terminal, new_soc

class BatteryStress:
    """Instantaneous stress proxy for training, not an aging law."""
    def __init__(self, p: StressParams, robot: RobotParams):
        self.p, self.robot = p, robot

    def step(self, current_a: float, temperature_c: float, soc: float,
             dt: float) -> float:
        p = self.p
        c_rate = abs(current_a) / (self.robot.Q_n / 3600.0)
        g_i = (c_rate / p.c_rate_ref) ** 2
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
        self.complexity_jitter = (0.2 if mode == 'train' else 0.0) \
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
        self.action_low = np.array([self.robot.v_min, self.robot.f_min], dtype=np.float32)
        self.action_high = np.array([self.robot.v_max, self.robot.f_max], dtype=np.float32)
        # [v, f, SOC, SOH, temperature, terminal voltage, distance remaining,
        #  last current, app throughputs..., app QoS targets...]
        self.obs_dim = 8 + 2 * self.n_apps
        self.act_dim = 2
        if spaces is not None:
            self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(self.obs_dim,), dtype=np.float32)
        self.np_random = np.random.default_rng()
        self._reset_internal()

    def _reset_internal(self):
        self.battery = ECMBattery(self.ecm_params, self.robot, self.soc_init, self.soh_init)
        self.stress_model = BatteryStress(self.stress_params, self.robot)
        self.v_obs = 0.0
        self.f_obs = self.robot.f_min
        self.distance_m = 0.0
        self.step_count = 0
        self.thr = [compute_throughput(0.0, self.f_obs, self.complexity)] * self.n_apps
        self.total_energy_j = 0.0
        self.total_stress = 0.0
        self.trajectory: list[dict] = []
        self._finished = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,
              soc_init: Optional[float] = None, soh_init: Optional[float] = None,
              complexity: Optional[float] = None):
        if gym is not None:
            super().reset(seed=seed)
        elif seed is not None:
            self.np_random = np.random.default_rng(seed)
        if options:
            soc_init = options.get('soc_init', soc_init)
            soh_init = options.get('soh_init', soh_init)
            complexity = options.get('complexity', complexity)
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
        arr = np.asarray(vals, dtype=np.float32)
        if self.mode == 'train' and self.robot.noise_std_frac > 0:
            # Sensor noise affects only agent observation, never dynamics/reward.
            std = self.robot.noise_std_frac
            arr[2] = np.clip(arr[2] + self.np_random.normal(0, std * 0.1), 0, 1)
            arr[5] += self.np_random.normal(0, std * max(1.0, arr[5]))
            arr[7] += self.np_random.normal(0, std * max(1.0, abs(arr[7])))
        return arr

    def reward_terms(self, energy_j: float, stress: float, delta_distance_m: float,
                     dt_s: float, qos_violation: float, failed: bool = False):
        """Per-step reward components; reward r_t = -E_d - D_d + U_t.

        E_d  energy cost    w_E * E_t / E_ref
             E_t = I*V*dt [J] drawn from the pack this step.
        D_d  degradation    w_D * S_t / S_ref
             S_t = dt * (w_I (C-rate/C_ref)^2 + w_T ((T-T_th)^+/T_s)^2
                         + w_S ((SOC-SOC_th)^+/(1-SOC_th))^2)
             = BatteryStress proxy [s]; dimensionless shaping, NOT dSOH.
        U_t  task utility   B * dx_t/L                      progress (sums to B)
                            - w_T * dt / (L / v_max)        time pressure
                            - w_Q * q_t * dt / dt_ctrl      QoS penalty
                            - P_fail * 1[failed/truncated]  terminal
             q_t = (max(0, 1 - min_i thr_i/QoS_ref_i))^2.

        L = mission distance. Returns (E_d, D_d, U_t) as floats.
        """
        p = self.robot
        E_d = p.w_energy * energy_j / p.energy_ref_j
        D_d = p.w_stress * stress / p.stress_ref_s
        U_t = (p.completion_bonus * delta_distance_m / p.mission_distance_m
               - p.w_time * dt_s / (p.mission_distance_m / p.v_max)
               - p.w_qos * qos_violation * dt_s / p.dt
               - (p.failure_penalty if failed else 0.0))
        return float(E_d), float(D_d), float(U_t)

    def step(self, action: Sequence[float], ext_Pe: Optional[float] = None):
        if self._finished:
            raise RuntimeError('Episode finished; call reset() before step().')
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError('Action must be a finite pair [speed, CPU frequency].')
        p = self.robot
        v_cmd = float(np.clip(action[0], p.v_min, p.v_max))
        f_cmd = float(np.clip(action[1], p.f_min, p.f_max))
        delta_v = np.clip(v_cmd - self.v_obs, -p.a_max * p.dt, p.a_max * p.dt)
        v_new = max(0.0, self.v_obs + float(delta_v))
        accel = float(delta_v) / p.dt
        remaining = p.mission_distance_m - self.distance_m
        delta_distance = min(remaining, v_new * p.dt)
        active_dt = (delta_distance / v_new) if v_new > 0 else p.dt
        motor_w = motor_power(v_new, accel, p)
        compute_w = compute_cpu_power(f_cmd, p.compute_base_w, p.compute_dynamic_w)
        if ext_Pe is not None:
            if not math.isfinite(ext_Pe) or ext_Pe < 0:
                raise ValueError('External compute electrical power must be finite and nonnegative.')
            compute_w = float(ext_Pe)  # external compute/board power, not total pack power
        total_w = motor_w + compute_w + p.aux_power_w
        reason = None
        try:
            current, voltage, soc = self.battery.step(total_w, dt=active_dt)
            energy_j = current * voltage * active_dt
            stress = self.stress_model.step(current, self.battery.temperature_c, soc, active_dt)
            self.distance_m += delta_distance
            self.total_energy_j += energy_j
            self.total_stress += stress
        except BatteryLimitError as exc:
            reason = str(exc)
            current, voltage, soc = self.battery.current_a, self.battery.voltage_v, self.battery.soc
            energy_j, stress, delta_distance = 0.0, 0.0, 0.0
        self.v_obs, self.f_obs = v_new, f_cmd
        c = self.complexity
        if self.complexity_jitter > 0:
            j = self.complexity_jitter
            c *= float(self.np_random.uniform(1.0 - j, 1.0 + j))
        self.thr = [compute_throughput(v_new, f_cmd, c)] * self.n_apps
        qos_ratio = [self.thr[i] / p.qos_ref[i] for i in range(self.n_apps)]
        qos_violation = max(0.0, 1.0 - min(qos_ratio)) ** 2
        self.step_count += 1
        completed = self.distance_m >= p.mission_distance_m - 1e-9
        low_soc = self.battery.soc <= 0.02
        terminated = bool(completed or low_soc or reason)
        truncated = bool(self.step_count >= p.max_steps and not terminated)
        failed = bool((terminated and not completed) or truncated)

        # Compute reward terms and return the Gymnasium API tuple.
        E_d, D_d, U_t = self.reward_terms(energy_j, stress, delta_distance,
                                          active_dt, qos_violation, failed)
        reward = -E_d - D_d + U_t

        self._finished = terminated or truncated
        info = dict(t_s=self.step_count * p.dt, dt_s=active_dt, speed_mps=v_new, acceleration_mps2=accel,
                    cpu_ghz=f_cmd, distance_m=self.distance_m, delta_distance_m=delta_distance,
                    motor_power_w=motor_w, compute_power_w=compute_w,
                    aux_power_w=p.aux_power_w, power_w=total_w, current_a=current,
                    voltage_v=voltage, soc=soc, soh=self.battery.soh,
                    temperature_c=self.battery.temperature_c,
                    energy_j=energy_j, total_energy_j=self.total_energy_j,
                    stress_proxy=stress, total_stress_proxy=self.total_stress,
                    throughput=list(self.thr), qos_violation=qos_violation,
                    E_d=E_d, D_d=D_d, U_t=U_t, reward=float(reward),
                    completed=completed, battery_limit=reason,
                    terminated=terminated, truncated=truncated)
        # Save actual physical trajectories (not noisy agent observations).
        self.trajectory.append(info.copy())
        return self._get_obs(), float(reward), terminated, truncated, info
