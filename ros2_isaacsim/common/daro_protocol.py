"""Wire protocol between the PC (Isaac Sim side) and the Jetson TX2.

Kept dependency-free (stdlib + numpy) and Python 3.6 compatible, because the
TX2 runs JetPack 4.x (Ubuntu 18.04, Python 3.6). No ROS 2 is needed on the TX2.

Two TCP channels, both served by the TX2 agent:

* Event channel (PC -> TX2), binary frames, one frame per event batch:
      header  struct '<4sIddQ' = (MAGIC, n_events, t_start_s, batch_dt_s, seq)
      payload n_events records of EVENT_DTYPE (t relative to t_start, x, y, polarity)
* Control channel (PC <-> TX2), newline-delimited JSON:
      PC  -> TX2  {"cmd": "set_freq", "khz": 2035200}
                  {"cmd": "hello"}
      TX2 -> PC   {"type": "status", "t": ..., "freq_khz": [...], "power_w": ...,
                   "thr": [q_app1, q_app2], "rate": [...], "events_per_batch": ...}
"""

import json
import socket
import struct

import numpy as np

MAGIC = b'DEV1'
HEADER = struct.Struct('<4sIddQ')
EVENT_DTYPE = np.dtype([('t', '<f4'), ('x', '<u2'), ('y', '<u2'), ('p', 'i1')])

DEFAULT_EVENT_PORT = 5600
DEFAULT_CTRL_PORT = 5601


def recv_exact(sock, n):
    """Read exactly n bytes; returns None if the peer closed the connection."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def pack_batch(events, t_start_s, batch_dt_s, seq):
    """events: structured array with EVENT_DTYPE (t relative to t_start_s)."""
    ev = np.ascontiguousarray(events, dtype=EVENT_DTYPE)
    return HEADER.pack(MAGIC, len(ev), float(t_start_s), float(batch_dt_s), int(seq)) + ev.tobytes()


def recv_batch(sock):
    """Returns (events, t_start_s, batch_dt_s, seq) or None on disconnect."""
    head = recv_exact(sock, HEADER.size)
    if head is None:
        return None
    magic, n, t_start, batch_dt, seq = HEADER.unpack(head)
    if magic != MAGIC:
        raise ValueError('bad event frame magic: %r' % (magic,))
    payload = recv_exact(sock, n * EVENT_DTYPE.itemsize) if n else b''
    if payload is None:
        return None
    return np.frombuffer(payload, dtype=EVENT_DTYPE), t_start, batch_dt, seq


def send_json(sock, obj):
    sock.sendall((json.dumps(obj) + '\n').encode('utf-8'))


class JsonLineReader(object):
    """Incremental newline-delimited JSON reader for a blocking socket."""

    def __init__(self, sock):
        self.sock = sock
        self.buf = b''

    def read(self):
        """Next JSON object, or None if the peer closed the connection."""
        while b'\n' not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                return None
            self.buf += chunk
        line, self.buf = self.buf.split(b'\n', 1)
        line = line.strip()
        if not line:
            return self.read()
        return json.loads(line.decode('utf-8'))


def listen(port, host='0.0.0.0'):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    return srv


def connect(host, port, timeout=5.0):
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock
