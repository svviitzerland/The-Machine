"""The Machine's body: a head with two servos (see robot/head), reached over the network.

The head speaks one-line commands ("P90 T60"). `Head` keeps a connection to it and
reconnects on its own; `Aim` turns where the person is in the camera into where the
head should point. A real head and the Wokwi simulator look the same from here.
"""
from __future__ import annotations

import socket
import threading
import time

from .config import BodyConfig


class Head:
    def __init__(self, address: str, cfg: BodyConfig):
        host, _, port = address.rpartition(":")
        self.host, self.port, self.cfg = host or address, int(port or 7351), cfg
        self.connected = False
        self.pan = self.tilt = 90.0
        self._sock: socket.socket | None = None
        self._sent = (None, None)
        self._last_send = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._keep_connected, daemon=True, name="head").start()

    def _keep_connected(self) -> None:
        said_waiting = False
        while not self._stop.is_set():
            try:
                s = socket.create_connection((self.host, self.port), timeout=3)
                s.settimeout(None)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._lock:
                    self._sock, self.connected, self._sent = s, True, (None, None)
                print(f"[body] head connected at {self.host}:{self.port}")
                said_waiting = False
                for line in s.makefile("r", encoding="utf-8", errors="replace"):
                    if line.startswith("OK P"):
                        parts = dict((p[0], float(p[1:])) for p in line[3:].split() if p[1:].replace(".", "").isdigit())
                        self.pan, self.tilt = parts.get("P", self.pan), parts.get("T", self.tilt)
            except OSError:
                if not said_waiting:
                    print(f"[body] waiting for the head at {self.host}:{self.port}")
                    said_waiting = True
            with self._lock:
                if self.connected:
                    print("[body] head disconnected")
                self._sock, self.connected = None, False
            self._stop.wait(2)

    def look(self, pan: float, tilt: float) -> None:
        """Point the head; small or too frequent changes are skipped."""
        pan, tilt = round(pan), round(tilt)
        now = time.time()
        if (pan, tilt) == self._sent or now - self._last_send < 1 / self.cfg.send_hz:
            return
        with self._lock:
            if self._sock is None:
                return
            try:
                self._sock.sendall(f"P{pan} T{tilt}\n".encode())
                self._sent, self._last_send = (pan, tilt), now
            except OSError:
                pass                              # the reader notices and reconnects

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            if self._sock:
                self._sock.close()


class Aim:
    """Where the head should point to look at someone."""

    def __init__(self, cfg: BodyConfig):
        self.cfg = cfg
        self.pan = self.tilt = 90.0
        self.last_seen = 0.0

    def update(self, box, frame_w: int, frame_h: int, now: float) -> tuple[float, float]:
        c = self.cfg
        if box is not None:
            x, y, w, h = box
            dx = 0.5 - (x + w / 2) / frame_w                 # >0: they are to the camera's left
            dy = 0.5 - (y + h * 0.4) / frame_h               # >0: above the middle (aim at the eyes)
            if c.camera_on_head:                             # the camera turns with the head: nudge toward them
                self.pan += c.pan_sign * c.gain * dx * c.hfov
                self.tilt += c.tilt_sign * c.gain * dy * c.vfov
            else:                                            # the camera stays put: point straight at them
                self.pan += (90 + c.pan_sign * dx * c.hfov - self.pan) * c.gain
                self.tilt += (90 + c.tilt_sign * dy * c.vfov - self.tilt) * c.gain
            self.last_seen = now
        elif now - self.last_seen > c.idle_s:                # nobody here: slowly look ahead again
            self.pan += (90 - self.pan) * 0.05
            self.tilt += (90 - self.tilt) * 0.05
        self.pan = min(max(self.pan, 0.0), 180.0)
        self.tilt = min(max(self.tilt, 0.0), 180.0)
        return self.pan, self.tilt
