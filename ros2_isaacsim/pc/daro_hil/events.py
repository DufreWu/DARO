"""Event-camera emulation from rendered frames (ESIM-style) and time batching.

Each pixel keeps a reference log intensity L_ref. For a new frame L at time t,
a pixel fires n = floor(|L - L_ref| / C) events of polarity sign(L - L_ref), with
timestamps linearly interpolated between the previous and the current frame
(the k-th event at the k-th threshold crossing), and L_ref moves by n * C.
This is the frame-interpolation model of ESIM / v2e without noise; the number of
events per batch grows with the image motion (speed) and the scene texture
(complexity), which is the load the perception applications see.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:   # pragma: no cover - optional
    cv2 = None


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    rgb = frame[..., :3].astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def resize(gray: np.ndarray, width: int, height: int) -> np.ndarray:
    if gray.shape == (height, width):
        return gray
    if cv2 is not None:
        return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    ys = (np.arange(height) + 0.5) * gray.shape[0] / height
    xs = (np.arange(width) + 0.5) * gray.shape[1] / width
    return gray[ys.astype(int)[:, None], xs.astype(int)[None, :]]


class EventEmulator:
    def __init__(self, width: int = 346, height: int = 260, threshold: float = 0.2,
                 log_eps: float = 1e-3, max_events_per_pixel: int = 10):
        self.w, self.h = width, height
        self.C, self.eps, self.n_max = float(threshold), float(log_eps), int(max_events_per_pixel)
        self.ref: np.ndarray | None = None
        self.t_prev: float | None = None

    def reset(self):
        self.ref, self.t_prev = None, None

    def process(self, frame: np.ndarray, t: float):
        """Frame (HxW or HxWx3/4, uint8) at time t [s] -> (t, x, y, p) arrays sorted by t."""
        L = np.log(resize(to_gray(frame), self.w, self.h) / 255.0 + self.eps).astype(np.float32)
        if self.ref is None or self.t_prev is None or t <= self.t_prev:
            self.ref, self.t_prev = L, float(t)
            return _empty()
        diff = L - self.ref
        n = np.minimum(np.floor(np.abs(diff) / self.C), self.n_max).astype(np.int64)
        flat = np.flatnonzero(n)
        if flat.size == 0:
            self.t_prev = float(t)
            return _empty()
        counts = n.ravel()[flat]
        d = diff.ravel()[flat]
        pix = np.repeat(flat, counts)
        # k = 1..n within each pixel
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        k = np.arange(pix.size) - starts + 1
        frac = np.clip(k * self.C / np.abs(np.repeat(d, counts)), 0.0, 1.0)
        ts = self.t_prev + frac * (float(t) - self.t_prev)
        pol = np.sign(np.repeat(d, counts)).astype(np.int8)
        self.ref.ravel()[flat] += np.sign(d) * counts * self.C
        self.t_prev = float(t)
        order = np.argsort(ts, kind='stable')
        return (ts[order], (pix[order] % self.w).astype(np.uint16),
                (pix[order] // self.w).astype(np.uint16), pol[order])


def _empty():
    return (np.zeros(0), np.zeros(0, np.uint16), np.zeros(0, np.uint16), np.zeros(0, np.int8))


class Batcher:
    """Groups time-sorted events into fixed windows [t0 + k dt, t0 + (k + 1) dt).

    push() returns the list of windows completed by the new events (possibly empty
    windows), each as (t_start, events_structured) with t relative to t_start.
    """

    def __init__(self, batch_dt: float, dtype):
        self.dt, self.dtype = float(batch_dt), dtype
        self.t_start: float | None = None
        self.pending = []

    def push(self, ts, xs, ys, ps, t_now: float):
        if self.t_start is None:
            self.t_start = float(t_now)
            return []
        if len(ts):
            self.pending.append((ts, xs, ys, ps))
        out = []
        while self.t_start + self.dt <= t_now:
            t_end = self.t_start + self.dt
            parts, keep = [], []
            for chunk in self.pending:
                cut = np.searchsorted(chunk[0], t_end, side='left')
                if cut:
                    parts.append(tuple(a[:cut] for a in chunk))
                if cut < len(chunk[0]):
                    keep.append(tuple(a[cut:] for a in chunk))
            self.pending = keep
            n = sum(len(p[0]) for p in parts)
            ev = np.zeros(n, dtype=self.dtype)
            if n:
                ev['t'] = np.concatenate([p[0] for p in parts]) - self.t_start
                ev['x'] = np.concatenate([p[1] for p in parts])
                ev['y'] = np.concatenate([p[2] for p in parts])
                ev['p'] = np.concatenate([p[3] for p in parts])
            out.append((self.t_start, ev))
            self.t_start = t_end
        return out
