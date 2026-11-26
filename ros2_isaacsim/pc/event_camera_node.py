#!/usr/bin/env python3
"""Event camera emulation (PC, ROS 2): Isaac Sim RGB -> events -> TX2.

Subscribes to the rendered front camera, converts every frame to events with the
ESIM-style model in daro_hil/events.py (simulation-time stamps from the image
header), groups them in fixed windows of --batch-ms (40 ms = 25 batch/s, the
throughput requirement) and streams each batch to the TX2 agent. Publishes the
number of events per batch on /daro/events_per_batch (as sent) and
/daro/events_per_batch_raw (as emulated from the scene).

With --match-table, every batch is resampled to the measured workload of the
paper (Table II, daro_hil/workload.py) at the current speed (odometry) and zone
complexity, so the TX2 load matches the profiling while the event structure
comes from the rendered scene.

    python3 pc/event_camera_node.py --tx2-host 192.168.1.20
    python3 pc/event_camera_node.py --tx2-host 192.168.1.20 --match-table
    python3 pc/event_camera_node.py --no-send       # calibration only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from daro_hil.events import EventEmulator, Batcher  # noqa: E402
from daro_hil.tx2_link import EventSender, proto  # noqa: E402
from daro_hil.workload import ZONE_C, target_events_per_batch, resample_events  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from std_msgs.msg import Float32, String  # noqa: E402

CHANNELS = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}


def image_to_numpy(msg: Image) -> np.ndarray:
    ch = CHANNELS.get(msg.encoding)
    if ch is None:
        raise ValueError(f'unsupported image encoding {msg.encoding}')
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = buf[:, :msg.width * ch].reshape(msg.height, msg.width, ch) if ch > 1 else buf[:, :msg.width]
    if msg.encoding.startswith('bgr'):
        img = img[..., [2, 1, 0]]
    return img


class EventCameraNode(Node):
    def __init__(self, args):
        super().__init__('daro_event_camera')
        self.emu = EventEmulator(args.width, args.height, args.threshold)
        self.batcher = Batcher(args.batch_ms / 1000.0, proto.EVENT_DTYPE)
        self.sender = None if args.no_send else EventSender(args.tx2_host, args.tx2_event_port)
        self.epb_pub = self.create_publisher(Float32, args.epb_topic, 10)
        self.epb_raw_pub = self.create_publisher(Float32, args.epb_topic + '_raw', 10)
        self.create_subscription(Image, args.image_topic, self._image_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, args.odom_topic, self._odom_cb, qos_profile_sensor_data)
        self.create_subscription(String, args.zone_topic, self._zone_cb, 10)
        self.batch_dt = args.batch_ms / 1000.0
        self.match = args.match_table
        self.width, self.height = args.width, args.height
        self.rng = np.random.default_rng(args.seed)
        self.speed, self.c = 0.0, None
        self._n_sum, self._n_raw, self._n_batches, self._t_last_pub = 0, 0, 0, None
        self.get_logger().info(
            f'{args.image_topic} -> events {args.width}x{args.height}, C={args.threshold}, '
            f'{args.batch_ms:g} ms batches -> {"(not sent)" if args.no_send else args.tx2_host}')

    def _odom_cb(self, msg: Odometry):
        self.speed = float(np.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y))

    def _zone_cb(self, msg: String):
        self.c = ZONE_C.get(msg.data)

    def _image_cb(self, msg: Image):
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        ts, xs, ys, ps = self.emu.process(image_to_numpy(msg), t)
        for t_start, ev in self.batcher.push(ts, xs, ys, ps, t):
            self._n_raw += len(ev)
            if self.match and self.c is not None:
                n = target_events_per_batch(self.speed, self.c)
                ev = resample_events(ev, int(round(n)), self.width, self.height, self.batch_dt, self.rng)
            if self.sender is not None:
                self.sender.submit(ev, t_start, self.batch_dt)
            self._n_sum += len(ev)
            self._n_batches += 1
        if self._t_last_pub is None:
            self._t_last_pub = t
        if t - self._t_last_pub >= 1.0 and self._n_batches:
            self.epb_pub.publish(Float32(data=float(self._n_sum / self._n_batches)))
            self.epb_raw_pub.publish(Float32(data=float(self._n_raw / self._n_batches)))
            self._n_sum, self._n_raw, self._n_batches, self._t_last_pub = 0, 0, 0, t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image-topic', default='/front_camera/rgb')
    ap.add_argument('--epb-topic', default='/daro/events_per_batch')
    ap.add_argument('--width', type=int, default=346)
    ap.add_argument('--height', type=int, default=260)
    ap.add_argument('--threshold', type=float, default=0.2, help='contrast threshold C (log intensity)')
    ap.add_argument('--batch-ms', type=float, default=40.0)
    ap.add_argument('--tx2-host', default='192.168.1.20')
    ap.add_argument('--tx2-event-port', type=int, default=proto.DEFAULT_EVENT_PORT)
    ap.add_argument('--no-send', action='store_true')
    ap.add_argument('--match-table', action='store_true',
                    help='resample every batch to the Table II workload (speed, zone complexity)')
    ap.add_argument('--odom-topic', default='/odom')
    ap.add_argument('--zone-topic', default='/daro/scene_zone')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rclpy.init()
    node = EventCameraNode(args)
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
