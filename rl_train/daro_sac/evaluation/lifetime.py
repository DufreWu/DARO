"""Missions / km to end of life, extrapolated from per-mission Xu fade.

SOH = exp(-F), where F accumulates fd_cycle(SOH) per mission. With fd_cycle
measured at the test SOH levels (full charge before each mission,
soc_init = 1), the number of missions until SOH_EOL is

    N_EOL = integral_{F(SOH=1)}^{F(SOH_EOL)} dF / fd_cycle(SOH = e^-F),

evaluated by trapezoid on the SOH grid (fd_cycle interpolated linearly).
This replaces a mission-by-mission simulation, which would need ~1e4 missions.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def missions_to_eol(soh_levels, fd_cycle, soh_eol: float = 0.8) -> float:
    soh = np.asarray(soh_levels, dtype=float)
    fd = np.asarray(fd_cycle, dtype=float)
    order = np.argsort(-soh)
    soh, fd = soh[order], fd[order]
    if soh[0] < 1.0 - 1e-9 or soh[-1] > soh_eol + 1e-9 or np.any(fd <= 0):
        return float('nan')
    grid = np.linspace(0.0, -math.log(soh_eol), 201)          # F grid
    fd_g = np.interp(np.exp(-grid), soh[::-1], fd[::-1])
    return float(np.trapezoid(1.0 / fd_g, grid))


def lifetime_from_results(results: dict, config: dict, soh_eol: float = 0.8) -> Dict:
    """results from evaluate_policies -> {policy: {complexity: {...}}}.

    Uses the soc_init = max(test_soc_levels) scenarios (full charge).
    """
    soc_full = max(config['test_soc_levels'])
    dist_km = config['mission_distance_m'] / 1000.0
    out = {}
    for pol, r in results.items():
        out[pol] = {}
        for c in config['test_complexity_levels']:
            rows = sorted((s for s in r['scenarios'].values()
                           if abs(s['soc_init'] - soc_full) < 1e-9
                           and abs(s['complexity'] - c) < 1e-9),
                          key=lambda s: -s['soh'])
            n = missions_to_eol([s['soh'] for s in rows], [s['fd_cycle'] for s in rows], soh_eol)
            out[pol][c] = {'missions_to_eol': n, 'km_to_eol': n * dist_km}
    return out
