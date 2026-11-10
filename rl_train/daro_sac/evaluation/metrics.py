"""Episode runner and metrics.

Capacity fade uses the Xu half-cycle + calendar model from battery_model.py
(LiPo-fitted k_cyc) applied to the env's own SOC / temperature trajectory.
The training stress proxy is reported separately and never used for SOH.

Per mission ("cycle" = discharge mission + recharge to the initial SOC):
  fd_discharge = Xu half-cycle(DoD, avg SOC, mean T) + calendar during mission
  fd_charge    = Xu half-cycle(DoD, avg SOC, T_amb) + calendar during charge
  dSOH         = SOH * (1 - exp(-(fd_discharge + fd_charge)))
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from robot_env import RobotEnv
from battery_model import _DEFAULT_AGING_PARAMS, _xu_fd_halfcycle, _xu_calendar_fd


def xu_mission_fd(env: RobotEnv, charge_c_rate: float) -> dict:
    ap = _DEFAULT_AGING_PARAMS
    traj = env.trajectory
    soc0, soh0 = env.soc_init, env.soh_init
    if not traj:
        return {'fd_discharge': 0.0, 'fd_charge': 0.0, 'fd_cycle': 0.0, 'delta_soh': 0.0}
    soc_end = float(traj[-1]['soc'])
    dod = max(0.0, soc0 - soc_end)
    avg = 0.5 * (soc0 + soc_end)
    dts = np.array([s['dt_s'] for s in traj], dtype=float)
    temps_k = np.array([s['temperature_c'] for s in traj], dtype=float) + 273.15
    t_mean = float(np.sum(dts * temps_k) / max(np.sum(dts), 1e-9))
    fd_dis = _xu_fd_halfcycle(dod, avg, t_mean, **ap)
    soc_prev = soc0
    for s, dt, tk in zip(traj, dts, temps_k):
        fd_dis += _xu_calendar_fd(soc_prev, float(s['soc']), float(dt), float(tk), **ap)
        soc_prev = float(s['soc'])
    t_amb_k = env.robot.T_amb + 273.15
    t_charge_s = dod * 3600.0 / charge_c_rate  # CC charge of dod at c-rate (of actual capacity)
    fd_chg = (_xu_fd_halfcycle(dod, avg, t_amb_k, **ap)
              + _xu_calendar_fd(soc_end, soc0, t_charge_s, t_amb_k, **ap))
    fd = fd_dis + fd_chg
    return {'fd_discharge': fd_dis, 'fd_charge': fd_chg, 'fd_cycle': fd,
            'delta_soh': soh0 * (1.0 - math.exp(-fd))}


def run_episode(env: RobotEnv, policy, seed: Optional[int] = None,
                deterministic: bool = True, charge_c_rate: float = 0.5) -> dict:
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, 'reset'):
        policy.reset()
    ret, viol_steps, steps, info, done = 0.0, 0, 0, {}, False
    while not done:
        action = policy.select_action(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ret += reward
        viol_steps += int(info['qos_violation'] > 0.0)
        steps += 1
    dist_km = max(env.distance_m, 1e-9) / 1000.0
    fade = xu_mission_fd(env, charge_c_rate)
    return {
        'return':         ret,
        'energy_kj':      env.total_energy_j / 1e3,
        'wh_per_km':      env.total_energy_j / 3600.0 / dist_km,
        'mission_time_s': float(info.get('t_s', 0.0)),
        'avg_speed_mps':  env.distance_m / max(float(info.get('t_s', 0.0)), 1e-9),
        'completed':      float(bool(info.get('completed', False))),
        'qos_viol_pct':   100.0 * viol_steps / max(steps, 1),
        'stress_proxy':   env.total_stress,
        'delta_soc':      env.soc_init - float(env.battery.soc),
        'delta_soh_ppm':  1e6 * fade['delta_soh'],
        'fd_cycle':       fade['fd_cycle'],
        'peak_temp_c':    max((s['temperature_c'] for s in env.trajectory),
                              default=env.robot.T_amb),
        'steps':          steps,
    }


def aggregate(eps: List[dict]) -> dict:
    out = {}
    for k in eps[0]:
        vals = np.array([e[k] for e in eps], dtype=float)
        out[k] = float(np.mean(vals))
        out[k + '_std'] = float(np.std(vals))
    return out
