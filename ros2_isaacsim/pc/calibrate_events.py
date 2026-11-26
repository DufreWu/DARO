#!/usr/bin/env python3
"""Calibrate the event emulation against the measured workload (Table II).

Run against daro_sim.py with one complexity on the whole route, e.g.
    bash simulator/launch_daro_sim.sh low --headless
    python3 pc/calibrate_events.py --out ../logs/hil/calib_low.json

The node drives the rover at each speed of --speeds (via /daro/speed_cmd), waits
--settle-s, and then, for --measure-s of simulation time, emulates events from the
camera for every contrast threshold in --thresholds in parallel, counting only
frames at the commanded speed (+-0.15 m/s). It reports the events per 40 ms batch
per threshold and speed next to Table II for the zone complexity, and the
threshold with the smallest mean |log(emulated / table)|.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from daro_hil.events import EventEmulator, to_gray, resize  # noqa: E402
from daro_hil.workload import ZONE_C, target_events_per_batch  # noqa: E402
from event_camera_node import image_to_numpy  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from std_msgs.msg import Float32, String  # noqa: E402


class Calibrator(Node):
    def __init__(self, args):
        super().__init__('daro_event_calibration')
        self.args = args
        self.emus = {c: EventEmulator(args.width, args.height, c) for c in args.thresholds}
        self.pub = self.create_publisher(Float32, '/daro/speed_cmd', 10)
        self.create_subscription(Image, args.image_topic, self._image, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self._odom, qos_profile_sensor_data)
        self.create_subscription(String, '/daro/scene_zone', self._zone, 10)
        self.create_timer(0.1, self._tick)
        self.speed, self.zone = 0.0, None
        self.k, self.t_phase = 0, None
        self.t_sim = None
        self.counts = {c: {v: 0 for v in args.speeds} for c in args.thresholds}
        self.dur = {v: 0.0 for v in args.speeds}
        self.t_prev_frame = None

    @property
    def v_cmd(self):
        return self.args.speeds[self.k] if self.k < len(self.args.speeds) else 0.0

    def _odom(self, msg):
        self.speed = float(np.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y))
        self.t_sim = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec

    def _zone(self, msg):
        self.zone = msg.data

    def _measuring(self):
        return (self.t_phase is not None and self.t_sim is not None
                and self.t_sim - self.t_phase >= self.args.settle_s)

    def _image(self, msg):
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        gray = resize(to_gray(image_to_numpy(msg)), self.args.width, self.args.height)
        ok = (self._measuring() and self.k < len(self.args.speeds)
              and abs(self.speed - self.v_cmd) < 0.15 and self.t_prev_frame is not None)
        for c, emu in self.emus.items():
            n = len(emu.process(gray, t)[0])
            if ok:
                self.counts[c][self.v_cmd] += n
        if ok:
            self.dur[self.v_cmd] += t - self.t_prev_frame
        self.t_prev_frame = t

    def _tick(self):
        self.pub.publish(Float32(data=float(self.v_cmd)))
        if self.t_sim is None or self.k >= len(self.args.speeds):
            return
        if self.t_phase is None:
            self.t_phase = self.t_sim
        if self.dur[self.v_cmd] >= self.args.measure_s:
            self.get_logger().info(f'v={self.v_cmd:.1f} m/s done (zone={self.zone})')
            self.k += 1
            self.t_phase = self.t_sim
            if self.k >= len(self.args.speeds):
                self.report()
                rclpy.shutdown()
        elif self.t_sim - self.t_phase > self.args.settle_s + 6 * self.args.measure_s:
            self.get_logger().warn(f'v={self.v_cmd:.1f}: speed not held long enough, skipping')
            self.k += 1
            self.t_phase = self.t_sim
            if self.k >= len(self.args.speeds):
                self.report()
                rclpy.shutdown()

    def report(self):
        a = self.args
        c_zone = ZONE_C.get(self.zone, float('nan'))
        bdt = a.batch_ms / 1000.0
        table = {v: target_events_per_batch(v, c_zone) for v in a.speeds} if self.zone else {}
        epb = {c: {v: (self.counts[c][v] * bdt / self.dur[v] if self.dur[v] > 0 else float('nan'))
                   for v in a.speeds} for c in a.thresholds}
        err = {}
        for c in a.thresholds:
            r = [np.log(epb[c][v] / table[v]) for v in a.speeds
                 if table and epb[c][v] > 0 and np.isfinite(epb[c][v])]
            err[c] = float(np.mean(np.abs(r))) if r else float('nan')
        best = min(err, key=lambda c: err[c] if np.isfinite(err[c]) else np.inf) if err else None
        lines = [f'zone={self.zone} (c={c_zone})  events per {a.batch_ms:g} ms batch',
                 'C       ' + ''.join(f'{v:>9.1f}' for v in a.speeds) + '   mean|log err|']
        if table:
            lines.append('Table II' + ''.join(f'{table[v]:>9.0f}' for v in a.speeds))
        for c in a.thresholds:
            lines.append(f'{c:<8.3f}' + ''.join(f'{epb[c][v]:>9.0f}' for v in a.speeds)
                         + f'   {err[c]:.3f}')
        lines.append(f'best threshold: {best}')
        print('\n'.join(lines), flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps({
                'zone': self.zone, 'c': c_zone, 'speeds': a.speeds, 'batch_ms': a.batch_ms,
                'table': {str(v): table.get(v) for v in a.speeds},
                'events_per_batch': {str(c): {str(v): epb[c][v] for v in a.speeds} for c in a.thresholds},
                'mean_abs_log_err': {str(c): err[c] for c in a.thresholds}, 'best_threshold': best,
                'measured_s': {str(v): self.dur[v] for v in a.speeds}}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--speeds', type=float, nargs='+', default=[1.0, 2.0, 3.0, 4.0, 5.0])
    ap.add_argument('--thresholds', type=float, nargs='+', default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    ap.add_argument('--settle-s', type=float, default=8.0)
    ap.add_argument('--measure-s', type=float, default=15.0)
    ap.add_argument('--batch-ms', type=float, default=40.0)
    ap.add_argument('--width', type=int, default=346)
    ap.add_argument('--height', type=int, default=260)
    ap.add_argument('--image-topic', default='/front_camera/rgb')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    rclpy.init()
    node = Calibrator(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
