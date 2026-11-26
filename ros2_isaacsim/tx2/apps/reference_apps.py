"""PLACEHOLDER event applications for bring-up of the HIL loop.

These are NOT APP-DT / APP-OP. They only reproduce the qualitative load shape
(a fixed per-batch cost plus a cost proportional to the number of events) so the
link, DVFS and throughput reporting can be tested before the real applications
are plugged in. For the paper, replace them with the real APP-DT / APP-OP by
wrapping them in the same interface:

    class MyApp(object):
        def __init__(self, width, height): ...
        def process(self, events, t_start, batch_dt): ...

events is a numpy structured array with fields t (float32, s, relative to
t_start), x, y (uint16) and p (int8, +1 / -1). Python 3.6 / old numpy compatible.
"""

import numpy as np


class RefDetect(object):
    """Event-count image, box filter, threshold, blob statistics."""

    def __init__(self, width, height, radius=3, passes=3):
        self.w, self.h, self.r, self.passes = width, height, radius, passes

    def _box(self, img):
        r = self.r
        c = np.cumsum(np.cumsum(np.pad(img, ((r + 1, r), (r + 1, r)), mode='constant'), 0), 1)
        return c[2 * r + 1:, 2 * r + 1:] - c[:-2 * r - 1, 2 * r + 1:] - c[2 * r + 1:, :-2 * r - 1] \
            + c[:-2 * r - 1, :-2 * r - 1]

    def process(self, events, t_start, batch_dt):
        idx = events['y'].astype(np.int64) * self.w + events['x'].astype(np.int64)
        img = np.bincount(idx, minlength=self.w * self.h).astype(np.float32).reshape(self.h, self.w)
        for _ in range(self.passes):            # fixed per-batch cost
            img = self._box(img) / float((2 * self.r + 1) ** 2)
        mask = img > (img.mean() + 2.0 * img.std() + 1e-6)
        ys, xs = np.nonzero(mask)
        if len(xs):
            order = np.argsort(xs + ys * self.w)  # per-detection cost
            return float(xs[order].mean()), float(ys[order].mean())
        return None


class RefFlow(object):
    """Time surface and local plane-fit normal flow at every event."""

    def __init__(self, width, height):
        self.w, self.h = width, height
        self.ts = np.zeros((height, width), dtype=np.float32)

    def process(self, events, t_start, batch_dt):
        if len(events) == 0:
            return None
        x = events['x'].astype(np.int64)
        y = events['y'].astype(np.int64)
        t = events['t'] + np.float32(t_start % 1000.0)
        self.ts[y, x] = t
        xc = np.clip(x, 1, self.w - 2)
        yc = np.clip(y, 1, self.h - 2)
        gx = 0.5 * (self.ts[yc, xc + 1] - self.ts[yc, xc - 1])
        gy = 0.5 * (self.ts[yc + 1, xc] - self.ts[yc - 1, xc])
        g2 = gx * gx + gy * gy + 1e-9
        u, v = gx / g2, gy / g2
        ok = np.isfinite(u) & np.isfinite(v) & (np.abs(u) < 1e4) & (np.abs(v) < 1e4)
        if ok.any():
            return float(np.median(u[ok])), float(np.median(v[ok]))
        return None
