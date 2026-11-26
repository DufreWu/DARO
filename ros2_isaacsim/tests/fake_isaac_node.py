#!/usr/bin/env python3
"""Stand-in for simulator/daro_sim.py to test the PC/TX2 pipeline without Isaac Sim.

Publishes the same topics as the Isaac Sim server: /odom (60 Hz, simulation-time
stamps, real time), /clock, /daro/scene_zone, /daro/route_progress,
/daro/wheel_power (NaN) and a synthetic front camera (/front_camera/rgb, 30 Hz):
a random-blob texture that scrolls with the travelled distance; the blob density
follows the zone (low/medium/high), so the event rate grows with speed and
complexity as in the rendered scene. The speed follows /daro/speed_cmd with the
DARO acceleration limit on a closed route of --lap-m metres.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

DENSITY = {'low': 0.002, 'medium': 0.005, 'high': 0.010}


def texture(density: float, width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((height, width), 90.0)
    n = int(density * width * height)
    yy, xx = np.mgrid[0:height, 0:width]
    for _ in range(n // 40 + 1):
        cx, cy, r = rng.uniform(0, width), rng.uniform(0, height), rng.uniform(3, 12)
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = rng.uniform(150, 255)
    return img


class FakeIsaac(Node):
    def __init__(self, args):
        super().__init__('fake_isaac')
        self.args = args
        self.odom_pub = self.create_publisher(Odometry, '/odom', qos_profile_sensor_data)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.zone_pub = self.create_publisher(String, '/daro/scene_zone', 10)
        self.progress_pub = self.create_publisher(Float32, '/daro/route_progress', 10)
        self.wheel_pub = self.create_publisher(Float32, '/daro/wheel_power', 10)
        self.img_pub = self.create_publisher(Image, '/front_camera/rgb', qos_profile_sensor_data)
        self.create_subscription(Float32, '/daro/speed_cmd', self._cmd, 10)
        self.zones = args.zone_order.split(',')
        self.tex = {z: texture(DENSITY[z], 4 * args.img_w, args.img_h, i)
                    for i, z in enumerate(DENSITY)}
        self.cmd, self.cmd_t = 0.0, -1e9
        self.v, self.s, self.t = 0.0, 0.0, 0.0
        self.t0 = time.monotonic()
        self.frame = 0
        self.create_timer(1.0 / 60.0, self._step)

    def _cmd(self, msg):
        self.cmd, self.cmd_t = float(msg.data), time.monotonic()

    def zone(self):
        prog = (self.s % self.args.lap_m) / self.args.lap_m
        return self.zones[min(int(prog * len(self.zones)), len(self.zones) - 1)], prog

    def _stamp(self, msg_header):
        msg_header.stamp.sec = int(self.t)
        msg_header.stamp.nanosec = int((self.t - int(self.t)) * 1e9)

    def _step(self):
        t_new = (time.monotonic() - self.t0) * self.args.rtf
        dt = t_new - self.t
        if dt <= 0:
            return
        target = self.cmd if time.monotonic() - self.cmd_t < 2.5 else 0.0
        dv = np.clip(target - self.v, -1.5 * dt, 0.75 * dt)
        self.v = max(0.0, self.v + dv)
        self.s += self.v * dt
        self.t = t_new
        ang = 2 * math.pi * self.s / self.args.lap_m
        rad = self.args.lap_m / (2 * math.pi)
        od = Odometry()
        self._stamp(od.header)
        od.header.frame_id, od.child_frame_id = 'odom', 'base_link'
        od.pose.pose.position.x, od.pose.pose.position.y = rad * math.cos(ang), rad * math.sin(ang)
        od.twist.twist.linear.x = -self.v * math.sin(ang)
        od.twist.twist.linear.y = self.v * math.cos(ang)
        self.odom_pub.publish(od)
        clk = Clock()
        clk.clock = od.header.stamp
        self.clock_pub.publish(clk)
        self.frame += 1
        zone, prog = self.zone()
        if self.frame % 6 == 0:
            self.progress_pub.publish(Float32(data=float(prog)))
            self.wheel_pub.publish(Float32(data=float('nan')))
        if self.frame % 60 == 0:
            self.zone_pub.publish(String(data=zone))
        if self.frame % 2 == 0 and not self.args.no_camera:
            tex = self.tex[zone]
            shift = int(self.s * self.args.px_per_m) % (tex.shape[1] - self.args.img_w)
            gray = tex[:, shift:shift + self.args.img_w].astype(np.uint8)
            img = Image()
            self._stamp(img.header)
            img.height, img.width, img.encoding, img.step = gray.shape[0], gray.shape[1], 'mono8', gray.shape[1]
            img.data = gray.tobytes()
            self.img_pub.publish(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lap-m', type=float, default=150.0)
    ap.add_argument('--zone-order', default='medium')
    ap.add_argument('--img-w', type=int, default=346)
    ap.add_argument('--img-h', type=int, default=260)
    ap.add_argument('--px-per-m', type=float, default=60.0)
    ap.add_argument('--rtf', type=float, default=1.0, help='simulation speed-up (no TX2 only)')
    ap.add_argument('--no-camera', action='store_true')
    args = ap.parse_args()
    rclpy.init()
    node = FakeIsaac(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
