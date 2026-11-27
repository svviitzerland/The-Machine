"""The Machine's body: a head with two servos (see robot/head), reached over the network.

The head speaks one-line commands ("P90 T60"). `Head` keeps a connection to it and
reconnects on its own; `Aim` turns where the person is in the camera into where the
head should point. A real head and the Wokwi simulator look the same from here.
"""
from __future__ import annotations

import math
import random
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


# Where "left", "right" and so on are in the picture. Its own left is the camera's left,
# which is the left side of the (unmirrored) picture.
DIRECTIONS = {"left": (0.05, 0.5), "right": (0.95, 0.5), "up": (0.5, 0.05), "down": (0.5, 0.95),
              "ahead": (0.5, 0.5)}


class Aim:
    """Where the head should point: at the person it is talking to, with the small
    movements of something alive, a glance at anyone else now and then, and a slow
    look around when nobody is there."""

    def __init__(self, cfg: BodyConfig):
        self.cfg = cfg
        self.pan = self.tilt = 90.0
        self.last_seen = 0.0
        self.glance: tuple[int, float] | None = None      # (which face, until when)
        self.next_glance = 0.0
        self.point: tuple[float, float, float] | None = None   # (x, y, until): somewhere it was told to look

    def look_at(self, x: float, y: float, now: float, hold: float = 3.0) -> None:
        """Look at a spot in the picture (0..1 from the top left) for a few seconds."""
        self.point = (x, y, now + hold)

    def update(self, boxes, frame_w: int, frame_h: int, now: float) -> tuple[float, float]:
        c = self.cfg
        target = None
        if self.point and now < self.point[2]:
            target = self.point[:2]
        elif boxes:
            i = 0
            if len(boxes) > 1:
                if self.glance and now < self.glance[1]:
                    i = min(self.glance[0], len(boxes) - 1)
                elif now > self.next_glance:
                    self.glance = (random.randrange(1, len(boxes)), now + random.uniform(*c.glance_s))
                    self.next_glance = now + random.uniform(*c.glance_every_s)
            x, y, w, h = boxes[i]
            target = ((x + w / 2) / frame_w, (y + h * 0.4) / frame_h)   # aim at the eyes
            self.last_seen = now
        if target is not None:
            dx, dy = 0.5 - target[0], 0.5 - target[1]       # >0: left of / above the middle of the picture
            if c.camera_on_head:                             # the camera turns with the head: nudge toward it
                self.pan += c.pan_sign * c.gain * dx * c.hfov
                self.tilt += c.tilt_sign * c.gain * dy * c.vfov
            else:                                            # the camera stays put: point straight at it
                self.pan += (90 + c.pan_sign * dx * c.reach_pan - self.pan) * c.gain
                self.tilt += (90 + c.tilt_sign * dy * c.reach_tilt - self.tilt) * c.gain
        elif now - self.last_seen > c.idle_s:                # nobody here: look around slowly
            self.pan += (90 + c.search_deg * math.sin(now * 0.35) - self.pan) * 0.05
            self.tilt += (92 + 4 * math.sin(now * 0.23) - self.tilt) * 0.05
        # never quite still: a slow, uneven drift like breathing and small eye movements
        alive = c.alive_deg
        wob_p = alive * (math.sin(now * 0.7) + 0.5 * math.sin(now * 1.9 + 1.3))
        wob_t = alive * 0.6 * (math.sin(now * 0.9 + 0.4) + 0.4 * math.sin(now * 2.3))
        self.pan = min(max(self.pan, 0.0), 180.0)
        self.tilt = min(max(self.tilt, 0.0), 180.0)
        return self.pan + wob_p, self.tilt + wob_t
