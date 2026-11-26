"""PC-side clients of the TX2 agent (control/status and event channels)."""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'common'))
import daro_protocol as proto  # noqa: E402


class Tx2Control:
    """Sends DVFS commands and keeps the latest status report from the TX2."""

    def __init__(self, host: str, port: int = proto.DEFAULT_CTRL_PORT, timeout: float = 10.0):
        self.sock = proto.connect(host, port, timeout=timeout)
        self._lock = threading.Lock()
        self.status: dict | None = None
        self.hello: dict | None = None
        self.n_status = 0
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        proto.send_json(self.sock, {'cmd': 'hello'})

    def _read_loop(self):
        reader = proto.JsonLineReader(self.sock)
        while True:
            try:
                msg = reader.read()
            except (OSError, ValueError):
                msg = None
            if msg is None:
                with self._lock:
                    self.status = None
                return
            with self._lock:
                if msg.get('type') == 'status':
                    msg['t_rx'] = time.monotonic()
                    self.status = msg
                    self.n_status += 1
                elif msg.get('type') == 'hello':
                    self.hello = msg

    def latest(self) -> dict | None:
        with self._lock:
            return None if self.status is None else dict(self.status)

    def wait_status(self, after: int, timeout: float) -> dict | None:
        """Block until a status newer than counter `after` arrives (or timeout)."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._lock:
                if self.n_status > after and self.status is not None:
                    return dict(self.status)
            time.sleep(0.01)
        return self.latest()

    def set_freq_khz(self, khz: int):
        proto.send_json(self.sock, {'cmd': 'set_freq', 'khz': int(khz)})

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class EventSender:
    """Background sender of event batches; drops the oldest batches if the link stalls."""

    def __init__(self, host: str, port: int = proto.DEFAULT_EVENT_PORT, max_queue: int = 50):
        self.host, self.port = host, port
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.sent = self.dropped = 0
        self._seq = 0
        self._sock = None
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, events, t_start: float, batch_dt: float):
        frame = proto.pack_batch(events, t_start, batch_dt, self._seq)
        self._seq += 1
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self.q.put_nowait(frame)

    def _loop(self):
        while True:
            frame = self.q.get()
            while True:
                try:
                    if self._sock is None:
                        self._sock = proto.connect(self.host, self.port)
                        self._sock.setsockopt(proto.socket.SOL_SOCKET, proto.socket.SO_SNDBUF, 8 << 20)
                    self._sock.sendall(frame)
                    self.sent += 1
                    break
                except OSError:
                    if self._sock is not None:
                        self._sock.close()
                    self._sock = None
                    time.sleep(1.0)
