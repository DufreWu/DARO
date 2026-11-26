#!/usr/bin/env python3
"""Offline checks of the DARO HIL components (no Isaac Sim, no TX2, no ROS).

    python3 tests/test_offline.py
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pc'))
sys.path.insert(0, str(ROOT / 'common'))

import daro_protocol as proto  # noqa: E402
from daro_hil.events import EventEmulator, Batcher  # noqa: E402
from daro_hil.hil_env import HilRobotEnv  # noqa: E402
from daro_hil.tx2_link import Tx2Control, EventSender  # noqa: E402
from envs.robot_env import RobotEnv, ECMParams, StressParams  # noqa: E402
from envs.test_env import EVAL_CONFIG, make_robot  # noqa: E402
from methods import LagRLSEPD  # noqa: E402


def test_hil_env_matches_robot_env():
    """Fed with RobotEnv's own speed/power/throughput, HilRobotEnv reproduces it exactly."""
    cfg = dict(EVAL_CONFIG, battery_obs='bms', battery_realism=True, mission_distance_m=300.0)

    def make(cls):
        robot = make_robot(cfg)
        return cls(robot=robot, ecm=ECMParams.default(robot), stress=StressParams(), soc_init=0.9,
                   soh_init=0.85, complexity=2.0, mode='test', complexity_jitter=0.0)

    ref, hil = make(RobotEnv), make(HilRobotEnv)
    o1, _ = ref.reset(seed=7)
    o2, _ = hil.reset(seed=7)
    assert np.allclose(o1, o2)
    pol = LagRLSEPD(make_robot(cfg), cfg)
    pol.reset()
    done, n = False, 0
    while not done:
        a = pol.select_action(o1)
        motor_cold, motor_hot = hil.motor_power_sample(*_speed_accel(ref, a))
        o1, r1, t1, tr1, i1 = ref.step(a)
        o2, r2, t2, tr2, i2 = hil.step_measured(
            i1['cpu_ghz'], i1['speed_mps'], i1['acceleration_mps2'], i1['speed_mps'] * ref.robot.dt,
            ref.robot.dt, i1['throughput'], i1['compute_power_w'],
            motor_w=motor_hot, motor_cold_w=motor_cold)
        assert np.allclose(o1, o2, rtol=1e-6, atol=1e-6), (n, o1, o2)
        assert abs(r1 - r2) < 1e-9 and t1 == t2 and tr1 == tr2, (n, r1, r2)
        done, n = t1 or tr1, n + 1
    assert abs(ref.total_energy_j - hil.total_energy_j) < 1e-6
    print(f'  HilRobotEnv == RobotEnv over {n} steps (completed={i1["completed"]})')


def _speed_accel(env, action):
    p = env.robot
    v_cmd = float(np.clip(action[0], p.v_min, p.v_max))
    dv = np.clip(v_cmd - env.v_obs, -p.a_max * p.dt, p.a_max * p.dt)
    return max(0.0, env.v_obs + float(dv)), float(dv) / p.dt


def test_event_emulator():
    rng = np.random.default_rng(0)

    def blobs(n):   # smooth texture: n bright discs of radius 6-14 px
        img = np.full((260, 1200), 60.0)
        yy, xx = np.mgrid[0:260, 0:1200]
        for _ in range(n):
            cx, cy, r = rng.uniform(0, 1200), rng.uniform(0, 260), rng.uniform(6, 14)
            img[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = 220.0
        return img

    tex_lo, tex_hi = blobs(40), blobs(200)
    counts = {}
    for name, tex in (('lo', tex_lo), ('hi', tex_hi)):
        for px_per_frame in (2, 8):
            emu, n = EventEmulator(), 0
            for k in range(30):
                ts, xs, ys, ps = emu.process(tex[:, k * px_per_frame:k * px_per_frame + 346].astype(np.uint8), k / 30)
                assert np.all(np.diff(ts) >= 0) and np.all((xs < 346) & (ys < 260))
                n += len(ts)
            counts[(name, px_per_frame)] = n
    assert counts[('hi', 2)] > counts[('lo', 2)] and counts[('lo', 8)] > counts[('lo', 2)], counts
    print(f'  events: {counts}')

    b = Batcher(0.04, proto.EVENT_DTYPE)
    assert b.push(*[np.zeros(0)] * 4, 0.0) == []
    ts = np.linspace(0.0, 0.1, 101)
    out = b.push(ts, np.zeros(101, np.uint16), np.zeros(101, np.uint16), np.ones(101, np.int8), 0.1)
    assert len(out) == 2 and abs(out[1][0] - 0.04) < 1e-12
    assert all((ev['t'] >= 0).all() and (ev['t'] < 0.04 + 1e-6).all() for _, ev in out)
    assert sum(len(ev) for _, ev in out) == 80, [len(ev) for _, ev in out]


def test_workload_table():
    from daro_hil.workload import EVENTS_PER_BATCH, SPEEDS, target_events_per_batch, resample_events
    for c, row in EVENTS_PER_BATCH.items():
        for v, n in zip(SPEEDS, row):
            assert abs(target_events_per_batch(v, c) - n) < 1e-6, (c, v)
    assert abs(target_events_per_batch(3.0, 2.5) - 41e3) < 1e-6
    rng = np.random.default_rng(0)
    ev = np.zeros(1000, dtype=proto.EVENT_DTYPE)
    ev['t'] = np.sort(rng.random(1000) * 0.04)
    ev['x'], ev['y'], ev['p'] = rng.integers(0, 346, 1000), rng.integers(0, 260, 1000), 1
    for n in (0, 400, 1000, 5000):
        out = resample_events(ev, n, 346, 260, 0.04, rng)
        assert len(out) == n and np.all(np.diff(out['t']) >= 0)
        assert np.all(out['x'] < 346) and np.all(out['y'] < 260) and np.all(out['t'] < 0.04)
    assert len(resample_events(ev[:0], 300, 346, 260, 0.04, rng)) == 300


def test_protocol():
    a, b = socket.socketpair()
    ev = np.zeros(5, dtype=proto.EVENT_DTYPE)
    ev['x'] = np.arange(5)
    a.sendall(proto.pack_batch(ev, 1.5, 0.04, 9))
    got, t0, dt, seq = proto.recv_batch(b)
    assert (got['x'] == ev['x']).all() and t0 == 1.5 and dt == 0.04 and seq == 9
    proto.send_json(a, {'cmd': 'x', 'v': 1})
    assert proto.JsonLineReader(b).read() == {'cmd': 'x', 'v': 1}


def _free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_tx2_agent_loopback():
    ev_port, ctrl_port = _free_port(), _free_port()
    agent = subprocess.Popen([sys.executable, str(ROOT / 'tx2' / 'daro_tx2_agent.py'), '--dry-run',
                              '--event-port', str(ev_port), '--ctrl-port', str(ctrl_port)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        time.sleep(1.5)
        ctrl = Tx2Control('127.0.0.1', ctrl_port)
        sender = EventSender('127.0.0.1', ev_port)
        rng = np.random.default_rng(1)
        ctrl.set_freq_khz(806400)
        statuses = []
        t_next = time.monotonic()
        for k in range(125):                  # 5 s at 25 batch/s
            ev = np.zeros(20000, dtype=proto.EVENT_DTYPE)
            ev['x'], ev['y'] = rng.integers(0, 346, 20000), rng.integers(0, 260, 20000)
            ev['t'] = np.sort(rng.random(20000) * 0.04)
            ev['p'] = 1
            sender.submit(ev, 0.04 * k, 0.04)
            t_next += 0.04
            time.sleep(max(0.0, t_next - time.monotonic()))
            if k >= 50 and k % 25 == 0:
                statuses.append(ctrl.latest())
        st = max(statuses, key=lambda s: min(s['rate']))
        assert ctrl.hello is not None and ctrl.hello['apps'] == ['APP-DT', 'APP-OP'], ctrl.hello
        assert st is not None and st['freq_khz'] == [806400], st
        assert all(q > 0 for q in st['thr']) and all(r > 20 for r in st['rate']), st
        assert 15000 < st['events_per_batch'] <= 20000, st
        print(f"  TX2 agent (dry run): thr={[round(q) for q in st['thr']]} batch/s, "
              f"rate={[round(r, 1) for r in st['rate']]}, P={st['power_w']:.2f} W, sent={sender.sent}")
        ctrl.close()
    finally:
        agent.terminate()
        agent.wait(5)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            print(name)
            fn()
    print('all offline tests passed')
