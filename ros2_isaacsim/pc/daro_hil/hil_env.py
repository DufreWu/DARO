"""HIL plant: DARO's RobotEnv with measured motion, throughput and compute power.

Everything that is simulated in the paper's evaluation stays identical (2RC ECM,
thermal model, BMS coulomb counter, sensor noise, per-episode cell variation,
reward terms, trajectory for the Xu capacity-fade metric). What changes per step:

  speed / distance   measured in Isaac Sim (odometry), instead of the commanded
                     speed with the acceleration limit;
  motor power        DARO motor model evaluated on the measured speed profile
                     (mean over the odometry samples of the step), including the
                     motor thermal state and the plant overrides (Mismatch family);
  compute power      measured on the TX2 (INA3221 VDD_IN) x plant compute_scale;
  throughput         measured on the TX2 per application;
  complexity         not simulated: produced by the Isaac Sim scene (logged as the
                     zone label for reference only).

The controllers see exactly the observation vector of RobotEnv._get_obs().
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# ROS 2 Jazzy ships numpy 1.26; DARO uses np.trapezoid (numpy >= 2.0), same as np.trapz.
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

DARO_ROOT = Path(__file__).resolve().parents[3]
if str(DARO_ROOT) not in sys.path:
    sys.path.insert(0, str(DARO_ROOT))
import envs  # noqa: E402,F401  (adds battery_model to sys.path)
from envs.robot_env import (RobotEnv, BatteryLimitError, CPU_FREQ_GHZ, N_FREQ_LEVELS,  # noqa: E402
                            motor_power, motor_power_cold)


class HilRobotEnv(RobotEnv):

    def step(self, action, ext_Pe=None):
        raise RuntimeError('HilRobotEnv is driven by step_measured()')

    @staticmethod
    def level_to_ghz(level: float) -> float:
        return float(CPU_FREQ_GHZ[int(min(max(round(float(level)), 0), N_FREQ_LEVELS - 1))])

    def motor_power_sample(self, speed: float, accel: float) -> tuple[float, float]:
        """(cold-fit, thermal-corrected) motor power [W] at the current motor temperature."""
        p = self.robot
        return (float(motor_power_cold(speed, accel, p)),
                float(motor_power(speed, accel, p, temp_rise=self.motor_temp_rise_k)))

    def step_measured(self, freq_ghz: float, speed_mps: float, accel_mps2: float,
                      distance_m: float, dt_s: float, thr: Sequence[float], compute_w: float,
                      motor_w: Optional[float] = None, motor_cold_w: Optional[float] = None,
                      complexity: float = float('nan'), extra: Optional[dict] = None):
        if self._finished:
            raise RuntimeError('Episode finished; call reset() before step_measured().')
        p = self.robot
        if not (dt_s > 0 and math.isfinite(dt_s)):
            raise ValueError('dt_s must be positive')
        v_new = max(0.0, float(speed_mps))
        accel = float(accel_mps2)
        remaining = p.mission_distance_m - self.distance_m
        travelled = max(0.0, float(distance_m))
        delta_distance = min(remaining, travelled)
        active_dt = dt_s if travelled <= remaining or travelled <= 0 else dt_s * remaining / travelled
        if motor_w is None or motor_cold_w is None:
            motor_cold_w, motor_w = self.motor_power_sample(v_new, accel)
        compute_w = float(compute_w) * float(getattr(p, 'compute_scale', 1.0))
        total_w = float(motor_w) + compute_w
        reason = None
        try:
            current, voltage, soc = self.battery.step(total_w, dt=active_dt)
            if p.energy_at_cell:
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
        if p.motor_thermal_loss and reason is None:
            heat_w = p.motor_heat_frac * motor_cold_w + (motor_w - motor_cold_w)
            self.motor_temp_rise_k += active_dt * (
                heat_w - p.motor_thermal_conductance_w_per_k * self.motor_temp_rise_k
            ) / p.motor_thermal_capacity_j_per_k
        self.v_obs, self.f_obs = v_new, float(freq_ghz)
        thr = [float(x) for x in thr]
        if len(thr) != self.n_apps:
            raise ValueError(f'expected {self.n_apps} throughputs, got {len(thr)}')
        self.thr = thr
        qos_ratio = [self.thr[i] / p.qos_ref[i] for i in range(self.n_apps)]
        qos_violation = max(0.0, 1.0 - min(qos_ratio))
        self.step_count += 1
        completed = self.distance_m >= p.mission_distance_m - 1e-9
        low_soc = self.battery.soc <= 0.02
        terminated = bool(completed or low_soc or reason)
        truncated = bool(self.step_count >= p.max_steps and not terminated)
        failed = bool((terminated and not completed) or truncated)

        terms = self.reward_terms(energy_j, stress, delta_distance, qos_violation, active_dt)
        R_end = self.terminal_reward(completed, self.battery.soc) if (terminated or truncated) else 0.0
        reward = (terms['U_t'] - terms['E_d'] - terms['D_d'] - terms['T_d'] - terms['Q_d'] + R_end)

        self._finished = terminated or truncated
        t_prev = self.trajectory[-1]['t_s'] if self.trajectory else 0.0
        info = dict(t_s=t_prev + dt_s, dt_s=active_dt, speed_mps=v_new, acceleration_mps2=accel,
                    cpu_ghz=float(freq_ghz), distance_m=self.distance_m,
                    delta_distance_m=delta_distance, motor_power_w=float(motor_w),
                    compute_power_w=compute_w, aux_power_w=0.0, power_w=total_w,
                    current_a=current, voltage_v=voltage, soc=soc, soh=self.battery.soh,
                    temperature_c=self.battery.temperature_c,
                    motor_temp_c=p.T_amb + self.motor_temp_rise_k,
                    energy_j=energy_j, total_energy_j=self.total_energy_j,
                    stress_proxy=stress, total_stress_proxy=self.total_stress,
                    throughput=list(self.thr), complexity=complexity, qos_violation=qos_violation,
                    **terms, R_end=R_end, reward=float(reward),
                    completed=completed, failed=failed, battery_limit=reason,
                    terminated=terminated, truncated=truncated, **(extra or {}))
        self.trajectory.append(info.copy())
        return self._get_obs(), float(reward), terminated, truncated, info
