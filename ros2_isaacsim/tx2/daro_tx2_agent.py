#!/usr/bin/env python3
"""DARO HIL agent for the Jetson TX2 (Python 3.6 compatible, stdlib + numpy).

Runs the perception applications on the event stream emulated from Isaac Sim,
applies the CPU DVFS level selected by the controller, and reports every
report period (default 1 s):

  thr[k]   throughput of application k [batch/s] = batches processed / busy time,
           i.e. the processing capacity, as in the offline profiling behind the
           throughput model q(v, f, c). q >= q_ref means the application keeps up
           with the 25 batch/s stream.
  rate[k]  batches actually processed per wall second (<= input rate).
  power_w  mean board input power (INA3221 VDD_IN) over the period.
  freq_khz current frequency of every cpufreq policy.

Each application runs in its own process (no GIL coupling). An application is a
class with ``__init__(self, width, height)`` and ``process(self, events, t_start, batch_dt)``,
given as NAME=module:Class (module importable from tx2/apps or PYTHONPATH). When an
application falls behind, the oldest queued batch is dropped (queue depth --queue).

    sudo python3 daro_tx2_agent.py \
        --app APP-DT=reference_apps:RefDetect --app APP-OP=reference_apps:RefFlow
    python3 daro_tx2_agent.py --dry-run      # off-device test (no DVFS, synthetic power)
"""

import argparse
import importlib
import multiprocessing as mp
import os
import queue
import socket
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, os.path.join(HERE, 'apps'))

import daro_protocol as proto  # noqa: E402
from tx2_platform import CpuDvfs, Ina3221, PowerSampler, DARO_CPU_KHZ  # noqa: E402


def _load_class(spec):
    module_name, cls_name = spec.split(':')
    return getattr(importlib.import_module(module_name), cls_name)


def app_worker(spec, width, height, q, stats, apps_dir, common_dir):
    sys.path.insert(0, apps_dir)
    sys.path.insert(0, common_dir)
    app = _load_class(spec)(width, height)
    while True:
        item = q.get()
        if item is None:
            break
        events, t_start, batch_dt = item
        t0 = time.perf_counter()
        app.process(events, t_start, batch_dt)
        dt = time.perf_counter() - t0
        with stats.get_lock():
            stats[0] += 1.0
            stats[1] += dt


class Agent(object):
    def __init__(self, args):
        self.args = args
        self.dvfs = CpuDvfs(dry_run=args.dry_run)
        self.power = PowerSampler(Ina3221(args.rail, dry_run=args.dry_run), self.dvfs)
        self.names, self.queues, self.stats, self.procs = [], [], [], []
        for item in args.app:
            name, spec = item.split('=', 1)
            q = mp.Queue(maxsize=args.queue)
            st = mp.Array('d', [0.0, 0.0])
            p = mp.Process(target=app_worker,
                           args=(spec, args.width, args.height, q, st,
                                 os.path.join(HERE, 'apps'), os.path.join(HERE, '..', 'common')))
            p.daemon = True
            self.names.append(name)
            self.queues.append(q)
            self.stats.append(st)
            self.procs.append(p)
        self._ev_lock = threading.Lock()
        self._ev_sum, self._ev_batches = 0, 0
        self._ctrl = None
        self._ctrl_lock = threading.Lock()
        self._stop = threading.Event()

    # ---------------- event channel ----------------
    def _dispatch(self, events, t_start, batch_dt):
        for q in self.queues:
            try:
                q.put_nowait((events, t_start, batch_dt))
            except queue.Full:
                try:
                    q.get_nowait()          # drop the oldest batch
                except queue.Empty:
                    pass
                try:
                    q.put_nowait((events, t_start, batch_dt))
                except queue.Full:
                    pass
        with self._ev_lock:
            self._ev_sum += len(events)
            self._ev_batches += 1

    def event_server(self):
        srv = proto.listen(self.args.event_port)
        print('[TX2] event channel on :%d' % self.args.event_port, flush=True)
        while not self._stop.is_set():
            conn, addr = srv.accept()
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
            print('[TX2] event stream from %s:%d' % addr, flush=True)
            try:
                while True:
                    item = proto.recv_batch(conn)
                    if item is None:
                        break
                    events, t_start, batch_dt, _seq = item
                    self._dispatch(events, t_start, batch_dt)
            except (OSError, ValueError) as exc:
                print('[TX2] event stream error: %s' % exc, flush=True)
            finally:
                conn.close()
                print('[TX2] event stream closed', flush=True)

    # ---------------- control channel ----------------
    def ctrl_server(self):
        srv = proto.listen(self.args.ctrl_port)
        print('[TX2] control channel on :%d' % self.args.ctrl_port, flush=True)
        while not self._stop.is_set():
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print('[TX2] controller connected from %s:%d' % addr, flush=True)
            with self._ctrl_lock:
                self._ctrl = conn
            reader = proto.JsonLineReader(conn)
            try:
                while True:
                    msg = reader.read()
                    if msg is None:
                        break
                    self._handle(msg)
            except (OSError, ValueError) as exc:
                print('[TX2] control error: %s' % exc, flush=True)
            finally:
                with self._ctrl_lock:
                    self._ctrl = None
                conn.close()
                print('[TX2] controller disconnected', flush=True)

    def _handle(self, msg):
        cmd = msg.get('cmd')
        if cmd == 'set_freq':
            self.dvfs.set_khz(int(msg['khz']))
        elif cmd == 'hello':
            self._send({'type': 'hello', 'apps': self.names, 'levels_khz': list(DARO_CPU_KHZ),
                        'dry_run': bool(self.args.dry_run)})
        else:
            print('[TX2] unknown command: %r' % (msg,), flush=True)

    def _send(self, obj):
        with self._ctrl_lock:
            conn = self._ctrl
            if conn is None:
                return
            try:
                proto.send_json(conn, obj)
            except OSError:
                pass

    # ---------------- main loop ----------------
    def run(self):
        for p in self.procs:
            p.start()
        self.power.start()
        self.dvfs.set_khz(self.args.init_khz)
        for target in (self.event_server, self.ctrl_server):
            th = threading.Thread(target=target)
            th.daemon = True
            th.start()
        prev = [(0.0, 0.0) for _ in self.stats]
        last_thr = [0.0 for _ in self.stats]
        t_prev = time.time()
        try:
            while True:
                time.sleep(self.args.report_period)
                now = time.time()
                wall = max(now - t_prev, 1e-6)
                t_prev = now
                thr, rate = [], []
                for k, st in enumerate(self.stats):
                    with st.get_lock():
                        cnt, busy = st[0], st[1]
                    dc, db = cnt - prev[k][0], busy - prev[k][1]
                    prev[k] = (cnt, busy)
                    if dc > 0 and db > 1e-6:
                        last_thr[k] = dc / db
                    thr.append(last_thr[k])        # hold the last capacity when idle
                    rate.append(dc / wall)
                with self._ev_lock:
                    epb = self._ev_sum / self._ev_batches if self._ev_batches else 0.0
                    self._ev_sum, self._ev_batches = 0, 0
                status = {'type': 'status', 't': now, 'freq_khz': self.dvfs.current_khz(),
                          'power_w': self.power.window_mean(), 'thr': thr, 'rate': rate,
                          'events_per_batch': epb, 'apps': self.names}
                self._send(status)
                if self.args.verbose:
                    print('[TX2] f=%s P=%.2fW thr=%s rate=%s ev/batch=%.0f' % (
                        status['freq_khz'], status['power_w'],
                        ['%.1f' % x for x in thr], ['%.1f' % x for x in rate], epb), flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            for q in self.queues:
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass
            if not self.args.keep_freq:
                self.dvfs.restore()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--event-port', type=int, default=proto.DEFAULT_EVENT_PORT)
    ap.add_argument('--ctrl-port', type=int, default=proto.DEFAULT_CTRL_PORT)
    ap.add_argument('--app', action='append', default=None,
                    help='NAME=module:Class (repeat per application, order = controller app order)')
    ap.add_argument('--width', type=int, default=346)
    ap.add_argument('--height', type=int, default=260)
    ap.add_argument('--queue', type=int, default=2, help='per-application batch queue depth')
    ap.add_argument('--report-period', type=float, default=1.0)
    ap.add_argument('--rail', default='VDD_IN')
    ap.add_argument('--init-khz', type=int, default=DARO_CPU_KHZ[-1])
    ap.add_argument('--keep-freq', action='store_true', help='do not release the DVFS pin on exit')
    ap.add_argument('--dry-run', action='store_true', help='no DVFS writes, synthetic power')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    if not args.app:
        args.app = ['APP-DT=reference_apps:RefDetect', 'APP-OP=reference_apps:RefFlow']
    if not args.dry_run and os.geteuid() != 0:
        sys.exit('DVFS needs root: run with sudo (or use --dry-run)')
    Agent(args).run()


if __name__ == '__main__':
    main()
