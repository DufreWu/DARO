"""Measured perception workload (paper Table II): events per batch vs complexity and speed.

    complexity   1.0   2.0   3.0   4.0   5.0  m/s
    low (c=1)    12k   18k   24k   30k   36k
    medium (c=2) 16k   24k   32k   40k   48k
    high (c=3)   30k   40k   50k   60k   70k

Each row is linear in the speed, N = a_c + b_c v (a = 6k/8k/20k, b = 6k/8k/10k per m/s).
Between levels the coefficients are interpolated linearly in c.
"""

from __future__ import annotations

import numpy as np

SPEEDS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
EVENTS_PER_BATCH = {
    1.0: np.array([12e3, 18e3, 24e3, 30e3, 36e3]),
    2.0: np.array([16e3, 24e3, 32e3, 40e3, 48e3]),
    3.0: np.array([30e3, 40e3, 50e3, 60e3, 70e3]),
}
ZONE_C = {'low': 1.0, 'medium': 2.0, 'high': 3.0}

# Linear fit per level (exact for Table II): N = a + b v
_COEF = {c: np.polyfit(SPEEDS, n, 1)[::-1] for c, n in EVENTS_PER_BATCH.items()}   # (a, b)


def target_events_per_batch(speed: float, complexity: float) -> float:
    """Table II workload at any speed >= 0 and complexity in [1, 3] (clipped)."""
    levels = np.array(sorted(_COEF))
    c = float(np.clip(complexity, levels[0], levels[-1]))
    a = np.interp(c, levels, [_COEF[k][0] for k in levels])
    b = np.interp(c, levels, [_COEF[k][1] for k in levels])
    return float(max(0.0, a + b * max(0.0, float(speed))))


def resample_events(ev: np.ndarray, n_target: int, width: int, height: int,
                    batch_dt: float, rng: np.random.Generator) -> np.ndarray:
    """Return a batch with exactly n_target events, keeping the scene's event structure.

    More events than the target: uniform subsampling. Fewer: the scene events are
    replicated with a +-1 px / within-batch time jitter (the spatial and polarity
    distribution of the scene is kept). No scene events: uniform random events.
    """
    n = len(ev)
    n_target = int(max(0, n_target))
    if n_target == n:
        return ev
    if n > n_target:
        keep = np.sort(rng.choice(n, size=n_target, replace=False))
        return ev[keep]
    out = np.empty(n_target, dtype=ev.dtype)
    out[:n] = ev
    extra = n_target - n
    if n == 0:
        out['x'] = rng.integers(0, width, extra)
        out['y'] = rng.integers(0, height, extra)
        out['t'] = rng.random(extra) * batch_dt
        out['p'] = rng.choice(np.array([-1, 1], dtype=np.int8), extra)
    else:
        src = ev[rng.integers(0, n, extra)]
        out['x'][n:] = np.clip(src['x'].astype(np.int64) + rng.integers(-1, 2, extra), 0, width - 1)
        out['y'][n:] = np.clip(src['y'].astype(np.int64) + rng.integers(-1, 2, extra), 0, height - 1)
        out['t'][n:] = np.clip(src['t'] + rng.normal(0.0, 0.1 * batch_dt, extra), 0.0, batch_dt * 0.999)
        out['p'][n:] = src['p']
    return out[np.argsort(out['t'], kind='stable')]
