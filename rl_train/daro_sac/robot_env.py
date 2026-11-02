"""
robot_env.py
------------
DARO robot environment compatible with the OpenAI Gym interface.

Implements the full working scenario from §II and the battery models from
§IV-A, including:
  - 2nd-order Equivalent Circuit Model (ECM) for short-term SOC (Eqs. 1-5)
  - Throughput-based fade model for long-term SOH (Eqs. 6-8)
  - Reward function: r(t) = -E_d(t) - D_d(t) + U(t)  (Eq. 10)
  - Rainflow-Counting stub for cycle-based SOH updates
  - Gaussian sensor noise added during training (§V-A)
  - EKF state estimation hook (interface only; full EKF in estimation.py)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Battery / Robot constants (Table I)
# ---------------------------------------------------------------------------

@dataclass
class ECMParams:
    """2nd-order RC equivalent circuit model parameters (Table I)."""
    Rs: float = 0.035    # Internal resistance [Ω]
    R1: float = 0.22     # RC branch 1 resistance [Ω]
    C1: float = 120.0    # RC branch 1 capacitance [F]
    R2: float = 0.019    # RC branch 2 resistance [Ω]
    C2: float = 1500.0   # RC branch 2 capacitance [F]
    eta: float = 0.98    # Coulombic efficiency factor


@dataclass
class DegradationParams:
    """Semi-empirical LiPo degradation model coefficients (Table I, Eqs. 8)."""
    k_sigma: float = 1.04       # Average-SOC stress coefficient
    k_T:     float = 6.93e-2    # Temperature stress coefficient
    k_d1:    float = 1.40e5     # DOD stress coefficient (amplitude)
    k_d2:    float = -5.01e-1   # DOD stress exponent
    k_d3:    float = -1.23e5    # DOD stress offset
    sigma_ref: float = 0.5      # Reference SOC
    T_ref:     float = 298.15   # Reference temperature [K]


@dataclass
class RobotParams:
    """Rover platform and software stack parameters (§II-A)."""
    # Battery
    Q_n: float = 5.0 * 3600     # Rated capacity [As] (5 Ah)
    V_nom: float = 14.8         # Nominal battery voltage [V]
    SOH_EOL: float = 0.80       # End-of-life SOH threshold
    # Motors
    v_min: float = 0.5          # Min motor speed [m/s]
    v_max: float = 5.0          # Max motor speed [m/s]
    # CPU DVFS
    f_min: float = 0.499        # Min CPU frequency [GHz]
    f_max: float = 2.0          # Max CPU frequency [GHz]
    # Control loop
    dt: float = 1.0             # Control period [s]
    T_amb: float = 25       # Ambient temperature [C]
    # Applications (N applications)
    n_apps: int = 2
    qos_ref: List[float] = field(default_factory=lambda: [25.0, 25.0])
    # Reward shaping
    alpha: float = 550.0        # Energy reward weight (§IV-B-2)
    kappa: float = 500.0        # Degradation penalty weight (§IV-B-2)
    gamma_qos: float = 3.0      # QoS utility exponent (Eq. 13)
    # Noise (§V-A)
    noise_std_frac: float = 0.02  # Fraction of power reading as noise std


# ---------------------------------------------------------------------------
# Open-circuit voltage lookup (simplified polynomial fit)
# Voc depends on SOC and SOH: at lower SOH, internal resistance increases,
# which lowers the effective Voc seen by the circuit.
# ---------------------------------------------------------------------------

def _voc(soc: float, soh: float, V_nom: float = 14.8) -> float:
    """
    Estimate open-circuit voltage from SOC and SOH.

    Uses a simple polynomial model calibrated to a 4S LiPo pack.
    The SOH scaling captures the voltage drop due to increased internal
    impedance at lower battery health states (§II-B, Fig. 4).
    """
    # Polynomial fit to Voc-SOC curve of a 4S LiPo (4 × 3.7 V nominal)
    coeffs = [2.031, -4.923, 5.565, -1.599, 13.97]
    voc_fresh = np.polyval(coeffs, np.clip(soc, 0.0, 1.0))
    # Slight reduction in Voc as SOH degrades (internal resistance effect)
    soh_factor = 0.95 + 0.05 * soh   # ranges 0.95–1.0 over SOH ∈ [0,1]
    return voc_fresh * soh_factor


def _internal_resistance(soc: float, soh: float,
                          ecm: ECMParams) -> float:
    """
    Effective internal resistance Rs(SOC, SOH).

    Rs increases as the battery ages (higher SOH loss ↔ higher impedance).
    Approximated from the ECM base value scaled by SOH⁻¹.
    """
    # Rs grows as SOH decreases
    return ecm.Rs / max(soh, 0.01)


# ---------------------------------------------------------------------------
# SOC ECM update  (Eqs. 1-2)
# ---------------------------------------------------------------------------

class ECMBattery:
    """
    2nd-order RC ECM for real-time SOC estimation (§IV-A-1).

    State: [V1, V2, SOC]
    Input: power demand Pe [W]
    """

    def __init__(self, ecm: ECMParams, robot: RobotParams,
                 soc_init: float = 1.0, soh_init: float = 1.0):
        self.ecm = ecm
        self.robot = robot
        self.soc = soc_init
        self.soh = soh_init
        # RC network voltages
        self.V1 = 0.0
        self.V2 = 0.0
        # Actual capacity accounts for aging: Q_act = SOH * Q_n
        self._update_capacity()

    def _update_capacity(self):
        self.Q_act = self.soh * self.robot.Q_n  # Eq. (2) denominator

    def update_soh(self, soh: float):
        self.soh = soh
        self._update_capacity()

    def step(self, Pe: float, add_noise: bool = False,
             noise_std_frac: float = 0.02) -> Tuple[float, float, float]:
        """
        Advance ECM by one timestep dt.

        Parameters
        ----------
        Pe : total electrical power demand [W]
        add_noise : inject Gaussian measurement noise (training only)

        Returns
        -------
        Ib   : battery current [A]
        Vb   : terminal voltage [V]
        soc  : updated SOC
        """
        dt = self.robot.dt
        ecm = self.ecm
        soc = self.soc
        soh = self.soh

        Voc = _voc(soc, soh, self.robot.V_nom)
        Rs  = _internal_resistance(soc, soh, ecm)

        # Noisy power measurement during training (§V-A)
        Pe_obs = Pe
        if add_noise:
            std = noise_std_frac * abs(Pe) + 1e-3
            Pe_obs = Pe + np.random.normal(0.0, std)

        # Battery current from ECM (Eq. 4)
        discriminant = (Voc - self.V1 - self.V2) ** 2 - 4 * Rs * Pe_obs
        discriminant = max(discriminant, 0.0)
        Ib = (Voc - self.V1 - self.V2 - math.sqrt(discriminant)) / (2 * Rs)

        # RC branch dynamics (Eqs. 1a, 1b)
        V1_new = self.V1 * (1 - dt / (ecm.R1 * ecm.C1)) + Ib * dt / ecm.C1
        V2_new = self.V2 * (1 - dt / (ecm.R2 * ecm.C2)) + Ib * dt / ecm.C2

        # SOC update (Eq. 2)
        soc_new = soc - ecm.eta * Ib * dt / self.Q_act
        soc_new = float(np.clip(soc_new, 0.0, 1.0))

        self.V1 = V1_new
        self.V2 = V2_new
        self.soc = soc_new

        Vb = Voc - Ib * Rs - self.V1 - self.V2
        return Ib, Vb, soc_new


# ---------------------------------------------------------------------------
# SOH fade model using Rainflow cycle counting  (Eqs. 6-8)
# ---------------------------------------------------------------------------

class SOHModel:
    """
    Throughput-based capacity fade model (§IV-A-2).

    Accumulates degradation across charge/discharge cycles identified
    by Rainflow counting applied to the SOC history.
    EOL is defined as SOH ≤ SOH_EOL (default 0.80).
    """

    def __init__(self, deg: DegradationParams, robot: RobotParams,
                 soh_init: float = 1.0):
        self.deg = deg
        self.robot = robot
        self.soh = soh_init
        self._fd_accum = -math.log(soh_init) if soh_init < 1.0 else 0.0
        self._soc_history: List[float] = []

    # ------------------------------------------------------------------
    # Stress factors (Eqs. 8a-8c)
    # ------------------------------------------------------------------

    def _S_sigma(self, sigma: float) -> float:
        """Average-SOC stress (Eq. 8a)."""
        return math.exp(self.deg.k_sigma * (sigma - self.deg.sigma_ref))

    def _S_delta(self, delta: float) -> float:
        """DOD stress (Eq. 8b)."""
        d = max(delta, 1e-6)
        return (self.deg.k_d1 * d ** self.deg.k_d2 + self.deg.k_d3) ** -1

    def _S_T(self, T: float) -> float:
        """Temperature stress (Eq. 8c)."""
        T_ref = self.deg.T_ref
        return math.exp(self.deg.k_T * (T - T_ref) / T_ref * T_ref / T)

    # ------------------------------------------------------------------
    # Rainflow cycle counting (simplified 3-point method)
    # ------------------------------------------------------------------

    @staticmethod
    def _rainflow_simple(peaks: List[float]) -> List[Tuple[float, float, float]]:
        """
        Simplified Rainflow counting on a list of SOC turning points.

        Returns a list of (DOD δ, mean_SOC σ, count n) tuples where
        count is 0.5 for half-cycles and 1.0 for full cycles.
        Reference: [23] Xu et al., IEEE Trans. Smart Grid, 2016.
        """
        if len(peaks) < 2:
            return []
        cycles = []
        stack = list(peaks)
        while len(stack) >= 3:
            a, b, c = stack[-3], stack[-2], stack[-1]
            range_ab = abs(b - a)
            range_bc = abs(c - b)
            if range_bc >= range_ab:
                # Count inner half-cycle (a→b)
                dod = range_ab
                sigma = (a + b) / 2.0
                cycles.append((dod, sigma, 0.5))
                stack.pop(-2)
            else:
                break
        # Remaining stack entries are residual half-cycles
        for i in range(len(stack) - 1):
            dod = abs(stack[i + 1] - stack[i])
            sigma = (stack[i] + stack[i + 1]) / 2.0
            cycles.append((dod, sigma, 0.5))
        return cycles

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record_soc(self, soc: float):
        """Append current SOC to history for later Rainflow analysis."""
        self._soc_history.append(soc)

    def update_soh(self, T: float = 298.15) -> float:
        """
        Compute cumulative degradation from SOC history and update SOH.

        Parameters
        ----------
        T : average temperature for the current mission segment [K]

        Returns
        -------
        Updated SOH value.
        """
        if len(self._soc_history) < 2:
            return self.soh

        # Extract turning points for Rainflow
        peaks = self._extract_turning_points(self._soc_history)
        cycles = self._rainflow_simple(peaks)

        # Accumulate degradation (Eq. 7)
        for (delta, sigma, n_i) in cycles:
            s_sigma = self._S_sigma(sigma)
            s_delta = self._S_delta(delta)
            s_T = self._S_T(T)
            self._fd_accum += n_i * s_delta * s_sigma * s_T

        # SOH from Eq. (6): SOH = e^{-f_d}
        self.soh = float(np.clip(math.exp(-self._fd_accum), 0.0, 1.0))
        self._soc_history.clear()
        return self.soh

    @staticmethod
    def _extract_turning_points(series: List[float]) -> List[float]:
        """Extract local minima/maxima (reversal points) from a time series."""
        if len(series) < 2:
            return series
        peaks = [series[0]]
        for i in range(1, len(series) - 1):
            if (series[i] > series[i - 1] and series[i] > series[i + 1]) or \
               (series[i] < series[i - 1] and series[i] < series[i + 1]):
                peaks.append(series[i])
        peaks.append(series[-1])
        return peaks

    @property
    def is_eol(self) -> bool:
        return self.soh <= self.robot.SOH_EOL


# ---------------------------------------------------------------------------
# Power models
# ---------------------------------------------------------------------------

def motor_power(speed: float) -> float:
    """
    Mechanical power demand of the brushless DC motor [W].

    Fitted as a quadratic in speed, calibrated to the rover's 3000KV motor.
    P_m ≈ k1 * v + k2 * v^2  (drag + inertia dominated)
    """
    k1, k2 = 2.5, 1.8    # empirical coefficients [W·s/m, W·s²/m²]
    return k1 * speed + k2 * speed ** 2


def compute_cpu_power(freq_ghz: float, base_power: float = 2.0,
                      alpha_cpu: float = 1.5) -> float:
    """
    CPU+camera electrical power [W] at a given DVFS frequency [GHz].

    Simple cubic model: P_c ∝ f^3 (DVFS dynamic power scaling).
    The camera is powered by the Jetson TX2 and included here (§II-A).
    """
    f_norm = freq_ghz / 2.0                  # normalise to max frequency
    return base_power + alpha_cpu * (f_norm ** 3)


def compute_throughput(speed: float, freq_ghz: float,
                        complexity: float = 1.0) -> float:
    """
    Application throughput (event batches/s) as a function of robot speed
    and CPU frequency (§II-A).

    - CPU frequency linearly scales processing throughput.
    - Higher speed generates more events per batch (larger batch sizes),
      indirectly reducing throughput available for QoS satisfaction.

    complexity : environmental complexity multiplier (1=low, 2=medium, 3=high)
    """
    thr_cpu = 25.0 * (freq_ghz / 2.0)        # linear in CPU freq
    event_load = 1.0 + 0.1 * speed * complexity  # more events at speed
    return thr_cpu / event_load


# ---------------------------------------------------------------------------
# Reward components  (Eqs. 10-13)
# ---------------------------------------------------------------------------

def energy_per_distance(delta_soc: float, speed: float,
                         Q_act: float, V_nom: float,
                         alpha: float) -> float:
    """
    E_d(t): energy consumed per metre of travel (Eq. 11).

    delta_soc : fractional SOC drop in one timestep (positive value)
    speed     : observed motor speed [m/s]
    Q_act     : actual battery capacity [As]
    V_nom     : nominal voltage [V] (converts As → Ws)
    alpha     : reward scaling parameter (550 in paper)
    """
    if speed < 1e-6:
        return 0.0
    delta_soc_pct = delta_soc * 100.0          # percentage drop
    Ed = alpha * delta_soc_pct / speed
    return Ed


def degradation_penalty(soh_prev: float, soh_curr: float,
                         speed: float, kappa: float) -> float:
    """
    D_d(t): battery degradation penalty per metre (Eq. 12).

    Only true SOH decreases contribute to the penalty; spurious increases
    (sensor noise / model recalibration) are suppressed via max(0, ...).
    """
    if speed < 1e-6:
        return 0.0
    delta_soh_pct = max(0.0, soh_prev - soh_curr) * 100.0
    Dd = (math.exp(kappa * delta_soh_pct) - 1.0) / speed
    return Dd


def qos_utility(thr: float, qos_ref: float, gamma: float = 3.0) -> float:
    """
    U(t): per-application QoS utility (Eq. 13).

    Returns 1 when thr ≥ qos_ref (full compliance), and
    (thr/qos_ref)^gamma < 1 otherwise (sharp penalty).
    """
    if thr >= qos_ref:
        return 1.0
    return (thr / qos_ref) ** gamma


# ---------------------------------------------------------------------------
# Main Gym-compatible environment
# ---------------------------------------------------------------------------

class DARORobotEnv:
    """
    DARO robot environment (§IV, observe–decide–act loop).

    Observation / state vector (§IV-B-2, Fig. 7):
        [v_obs, f_obs, SOC, SOH, thr_1, …, thr_N, QoS_ref_1, …, QoS_ref_N]
    dimension: 4 + 2*N

    Action vector:
        [v_cmd, f_cmd]  — desired speed [m/s] and CPU frequency [GHz]
    dimension: 2

    The environment can run in three modes:
        'train'  : simulated, noise injected, SOH updated each episode
        'hil'    : actual computing board power used, simulated battery
        'real'   : full hardware (real power readings fed externally)
    """

    def __init__(self,
                 robot: RobotParams = None,
                 ecm: ECMParams = None,
                 deg: DegradationParams = None,
                 soc_init: float = 1.0,
                 soh_init: float = 1.0,
                 complexity: float = 1.0,
                 mode: str = 'train',
                 max_steps: int = 1000):

        self.robot = robot or RobotParams()
        self.ecm_params = ecm or ECMParams()
        self.deg_params = deg or DegradationParams()

        self.soc_init = soc_init
        self.soh_init = soh_init
        self.complexity = complexity   # environmental complexity level
        self.mode = mode
        self.max_steps = max_steps

        self.n_apps = self.robot.n_apps
        self.obs_dim = 4 + 2 * self.n_apps
        self.act_dim = 2

        # Action space bounds
        self.action_low  = np.array([self.robot.v_min, self.robot.f_min],
                                    dtype=np.float32)
        self.action_high = np.array([self.robot.v_max, self.robot.f_max],
                                    dtype=np.float32)

        self._reset_internal()

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, soc_init: Optional[float] = None,
              soh_init: Optional[float] = None,
              complexity: Optional[float] = None) -> np.ndarray:
        """Reset environment to initial conditions. Returns initial obs."""
        if soc_init is not None:
            self.soc_init = soc_init
        if soh_init is not None:
            self.soh_init = soh_init
        if complexity is not None:
            self.complexity = complexity
        self._reset_internal()
        return self._get_obs()

    def step(self, action: np.ndarray,
             ext_Pe: Optional[float] = None
             ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one control step (observe–decide–act, §IV-B-4).

        Parameters
        ----------
        action  : [v_cmd, f_cmd] — clipped to valid ranges internally
        ext_Pe  : externally measured power [W] (HIL/real modes)

        Returns
        -------
        obs     : next state vector
        reward  : scalar reward r(t) = -E_d - D_d + U(t)
        done    : True if SOC depleted or max_steps reached
        info    : diagnostic dictionary
        """
        v_cmd = float(np.clip(action[0], self.robot.v_min, self.robot.v_max))
        f_cmd = float(np.clip(action[1], self.robot.f_min, self.robot.f_max))

        soh_prev = self.soh_model.soh
        soc_prev = self.battery.soc

        # --- Power demand ---
        Pe_mech = motor_power(v_cmd)
        Pe_comp = compute_cpu_power(f_cmd)
        Pe_total = Pe_mech + Pe_comp

        if ext_Pe is not None:
            # HIL / real mode: override computing power with measurement
            Pe_total = ext_Pe + Pe_mech

        # --- ECM step (SOC update) ---
        add_noise = (self.mode == 'train')
        Ib, Vb, soc_new = self.battery.step(
            Pe_total, add_noise=add_noise,
            noise_std_frac=self.robot.noise_std_frac
        )

        # --- SOH model: record SOC sample ---
        self.soh_model.record_soc(soc_new)

        # --- Throughput (per app) with workload fluctuation in training ---
        thrp = []
        for i in range(self.n_apps):
            c = self.complexity
            if self.mode == 'train':
                # Randomise complexity level per step to improve robustness
                c_fluct = c * np.random.uniform(0.8, 1.2)
            else:
                c_fluct = c
            thr_i = compute_throughput(v_cmd, f_cmd, complexity=c_fluct)
            thrp.append(thr_i)

        # --- Reward components ---
        delta_soc = max(soc_prev - soc_new, 0.0)
        Ed = energy_per_distance(delta_soc, v_cmd,
                                  self.battery.Q_act,
                                  self.robot.V_nom,
                                  self.robot.alpha)
        Dd = degradation_penalty(soh_prev, self.soh_model.soh,
                                  v_cmd, self.robot.kappa)

        # QoS utility: use the most-critical application (highest load)
        U_vals = [qos_utility(thrp[i], self.robot.qos_ref[i],
                               self.robot.gamma_qos)
                  for i in range(self.n_apps)]
        U = min(U_vals)   # most critical (bottleneck) application

        reward = -Ed - Dd + U

        # --- State transitions ---
        self.v_obs = v_cmd
        self.f_obs = f_cmd
        self.thr   = thrp
        self.step_count += 1

        # --- Termination ---
        soc_depleted = soc_new <= 0.02        # ≈ empty
        timeout      = self.step_count >= self.max_steps
        done = soc_depleted or timeout

        # --- End-of-mission SOH update (Rainflow) ---
        if done:
            self.soh_model.update_soh(T=self.robot.T_amb)
            self.battery.update_soh(self.soh_model.soh)

        info = {
            'v_cmd': v_cmd, 'f_cmd': f_cmd,
            'Pe_total': Pe_total, 'Ib': Ib, 'Vb': Vb,
            'SOC': soc_new, 'SOH': self.soh_model.soh,
            'Ed': Ed, 'Dd': Dd, 'U': U,
            'throughput': thrp,
            'is_eol': self.soh_model.is_eol,
        }
        return self._get_obs(), reward, done, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_internal(self):
        self.battery   = ECMBattery(self.ecm_params, self.robot,
                                     self.soc_init, self.soh_init)
        self.soh_model = SOHModel(self.deg_params, self.robot, self.soh_init)
        self.v_obs     = (self.robot.v_min + self.robot.v_max) / 2.0
        self.f_obs     = (self.robot.f_min + self.robot.f_max) / 2.0
        self.thr       = list(self.robot.qos_ref)
        self.step_count = 0

    def _get_obs(self) -> np.ndarray:
        """
        Assemble state vector s(t) (§IV-B-2):
        [v_obs, f_obs, SOC, SOH, thr_1,...,thr_N, QoS_ref_1,...,QoS_ref_N]
        """
        obs = np.array(
            [self.v_obs, self.f_obs,
             self.battery.soc, self.soh_model.soh]
            + list(self.thr)
            + list(self.robot.qos_ref),
            dtype=np.float32
        )
        return obs

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        return self.obs_dim

    @property
    def action_dim_prop(self) -> int:
        return self.act_dim

    def is_battery_eol(self) -> bool:
        return self.soh_model.is_eol

    def current_soh(self) -> float:
        return self.soh_model.soh

    def current_soc(self) -> float:
        return self.battery.soc