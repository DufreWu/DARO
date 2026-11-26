"""Jetson TX2 platform interface: CPU DVFS and board power (INA3221).

Python 3.6 compatible, stdlib + numpy only. Writing cpufreq needs root.

DVFS: every cpufreq policy (TX2: A57 cluster and Denver cluster) is pinned to
the requested frequency by writing scaling_min_freq / scaling_max_freq, which
works with any governor. The 11 DARO levels (0.4992-2.0352 GHz) are the A57
table; for each policy the nearest available frequency is used.

Power: the INA3221 rails are discovered by name. L4T 32.x (JetPack 4) exposes
them as iio devices of the ina3221x driver:
    /sys/bus/i2c/drivers/ina3221x/*/iio:device*/rail_name_<k>, in_power<k>_input [mW]
On the TX2 module the board input rail is VDD_IN. Newer kernels use the hwmon
ina3221 driver (in<k>_label, in<k>_input [mV], curr<k>_input [mA]); both are
supported.
"""

import glob
import os
import threading
import time

DARO_CPU_KHZ = (499200, 652800, 806400, 960000, 1113600, 1267200, 1420800,
                1574400, 1728000, 1881600, 2035200)


def _read(path):
    with open(path) as fh:
        return fh.read().strip()


def _write(path, value):
    with open(path, 'w') as fh:
        fh.write(str(value))


class CpuDvfs(object):
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.policies = sorted(glob.glob('/sys/devices/system/cpu/cpufreq/policy*'))
        if not self.policies and not dry_run:
            raise RuntimeError('no cpufreq policies found; run on the TX2 or use --dry-run')
        self.available = {}
        for pol in self.policies:
            path = os.path.join(pol, 'scaling_available_frequencies')
            if os.path.exists(path):
                self.available[pol] = sorted(int(x) for x in _read(path).split())
            else:
                lo = int(_read(os.path.join(pol, 'cpuinfo_min_freq')))
                hi = int(_read(os.path.join(pol, 'cpuinfo_max_freq')))
                self.available[pol] = [lo, hi]
        self._dry_khz = DARO_CPU_KHZ[-1]

    def _nearest(self, pol, khz):
        return min(self.available[pol], key=lambda f: abs(f - khz))

    def set_khz(self, khz):
        khz = int(khz)
        if self.dry_run:
            self._dry_khz = khz
            return
        for pol in self.policies:
            f = self._nearest(pol, khz)
            cur_min = int(_read(os.path.join(pol, 'scaling_min_freq')))
            # Keep min <= max at every intermediate step.
            if f >= cur_min:
                _write(os.path.join(pol, 'scaling_max_freq'), f)
                _write(os.path.join(pol, 'scaling_min_freq'), f)
            else:
                _write(os.path.join(pol, 'scaling_min_freq'), f)
                _write(os.path.join(pol, 'scaling_max_freq'), f)

    def current_khz(self):
        if self.dry_run:
            return [self._dry_khz]
        out = []
        for pol in self.policies:
            try:
                out.append(int(_read(os.path.join(pol, 'scaling_cur_freq'))))
            except (IOError, OSError, ValueError):
                out.append(-1)
        return out

    def restore(self):
        """Release the pin (min = cpuinfo_min, max = cpuinfo_max)."""
        if self.dry_run:
            return
        for pol in self.policies:
            _write(os.path.join(pol, 'scaling_min_freq'), _read(os.path.join(pol, 'cpuinfo_min_freq')))
            _write(os.path.join(pol, 'scaling_max_freq'), _read(os.path.join(pol, 'cpuinfo_max_freq')))


class Ina3221(object):
    """Reads one named rail (default VDD_IN) in watts."""

    def __init__(self, rail='VDD_IN', dry_run=False):
        self.rail, self.dry_run = rail, dry_run
        self._reader = None
        if dry_run:
            return
        # iio ina3221x (L4T 32.x)
        for name_path in glob.glob('/sys/bus/i2c/drivers/ina3221x/*/iio:device*/rail_name_*'):
            if _read(name_path).upper() == rail.upper():
                k = name_path.rsplit('_', 1)[-1]
                p_path = os.path.join(os.path.dirname(name_path), 'in_power%s_input' % k)
                self._reader = lambda p=p_path: float(_read(p)) / 1000.0
                self.source = p_path
                return
        # hwmon ina3221 (newer kernels)
        for label_path in glob.glob('/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label'):
            if _read(label_path).upper() == rail.upper():
                d = os.path.dirname(label_path)
                k = os.path.basename(label_path)[2:].split('_')[0]
                v_path = os.path.join(d, 'in%s_input' % k)
                i_path = os.path.join(d, 'curr%s_input' % k)
                self._reader = lambda v=v_path, i=i_path: float(_read(v)) * float(_read(i)) / 1e6
                self.source = d
                return
        raise RuntimeError('INA3221 rail %r not found' % rail)

    def read_w(self, dry_khz=None):
        if self.dry_run:
            f = (dry_khz or DARO_CPU_KHZ[-1]) / 1e6
            return 3.0 + 1.5 * (f / 2.0) ** 3   # placeholder for off-device tests only
        return self._reader()


class PowerSampler(threading.Thread):
    """Samples a rail at a fixed rate; window_mean() returns the mean since the last call."""

    def __init__(self, meter, dvfs, rate_hz=20.0):
        threading.Thread.__init__(self)
        self.daemon = True
        self.meter, self.dvfs, self.period = meter, dvfs, 1.0 / rate_hz
        self._lock = threading.Lock()
        self._sum, self._n, self._last = 0.0, 0, float('nan')
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                w = self.meter.read_w(self.dvfs.current_khz()[0] if self.meter.dry_run else None)
            except (IOError, OSError, ValueError):
                w = float('nan')
            with self._lock:
                if w == w:
                    self._sum += w
                    self._n += 1
                    self._last = w
            time.sleep(self.period)

    def window_mean(self):
        with self._lock:
            mean = self._sum / self._n if self._n else self._last
            self._sum, self._n = 0.0, 0
        return mean

    def stop(self):
        self._stop.set()
