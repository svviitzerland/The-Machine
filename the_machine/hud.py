"""The on-screen overlay: what the Machine sees, hears, thinks and says."""
from __future__ import annotations

import math
import textwrap
import time
from collections import deque

import cv2
import numpy as np


# BGR
YELLOW = (0, 215, 255)
WHITE = (235, 235, 235)
BLUE = (255, 170, 70)
RED = (70, 70, 235)
GREY = (150, 150, 150)
DARK = (14, 14, 14)
ROLE_COLOR = {"admin": YELLOW, "user": WHITE, "stranger": BLUE}
ROLE_TAG = {"admin": "ADMIN", "user": "KNOWN", "stranger": "UNIDENTIFIED"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
LOCK_S = 0.35            # how long the lock-on animation takes
TYPE_CPS = 45            # typing speed of names and replies, characters per second
SUB_S = 9.0              # how long a subtitle stays on screen


def ease(k: float) -> float:
    k = min(max(k, 0.0), 1.0)
    return 1 - (1 - k) ** 3


def text(img, s, org, scale=0.42, col=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, col, thick, cv2.LINE_AA)


def width(s, scale=0.42, thick=1) -> int:
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


def panel(img, x0, y0, x1, y1, alpha=0.6):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        roi[:] = (roi * (1 - alpha) + np.array(DARK) * alpha).astype(np.uint8)


def dashed_rect(img, x, y, w, h, col, dash=8, thick=1):
    for (ax, ay, bx, by) in ((x, y, x + w, y), (x + w, y, x + w, y + h), (x + w, y + h, x, y + h), (x, y + h, x, y)):
        n = max(1, int(math.hypot(bx - ax, by - ay) // dash))
        for i in range(0, n, 2):
            p = (int(ax + (bx - ax) * i / n), int(ay + (by - ay) * i / n))
            q = (int(ax + (bx - ax) * min(i + 1, n) / n), int(ay + (by - ay) * min(i + 1, n) / n))
            cv2.line(img, p, q, col, thick, cv2.LINE_AA)


def brackets(img, x, y, w, h, col, length, thick=2):
    for px, py, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(img, (px, py), (px + dx * length, py), col, thick, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * length), col, thick, cv2.LINE_AA)


def typed(s: str, since: float, now: float) -> str:
    return s[: int((now - since) * TYPE_CPS)]


class Hud:
    def __init__(self):
        self.locked: dict[int, float] = {}            # track id -> when we first drew it
        self.named: dict[tuple[int, int], float] = {}  # (track, concept) -> when it got that identity
        self.heard, self.heard_t = "", 0.0
        self.said, self.said_t = "", 0.0
        self.wave: deque[float] = deque([0.0] * 28, maxlen=28)
        self.bars = [0.0, 0.0, 0.0]
        self._shape = None
        self._vignette = None

    def hear(self, s: str) -> None:
        self.heard, self.heard_t = s, time.time()

    def _grade(self, frame: np.ndarray, now: float) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._shape != (h, w):
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
            v = np.clip(1.15 - 0.45 * r ** 2, 0.35, 1.0)
            v[::3] *= 0.88                                     # scanlines
            self._vignette = cv2.merge([(v * 255).astype(np.uint8)] * 3)
            self._shape = (h, w)
        grey = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(frame, 0.7, grey, 0.3, 0)
        out = cv2.multiply(out, self._vignette, scale=1 / 255)
        y = int((now * 90) % (h + 60)) - 30                    # slow sweep line
        if 0 <= y < h:
            out[y] = cv2.add(out[y], np.full_like(out[y], 28))
        return out

    def draw(self, frame: np.ndarray, mind, fps: float, ears=None, speaking: bool = False) -> np.ndarray:
        now = time.time()
        out = self._grade(frame, now)
        h, w = out.shape[:2]
        if mind.thought and mind.thought != "..." and mind.thought != self.said:
            self.said, self.said_t = mind.thought, now
        live = {t.id for t in mind.visible()}
        self.locked = {k: v for k, v in self.locked.items() if k in live}
        self.named = {k: v for k, v in self.named.items() if k[0] in live}
        for t in mind.visible():
            self._face(out, mind, t, now)
        if mind.pending and mind.pending.kind == "object":
            self._object(out, now)
        self._status(out, mind, ears, speaking, now)
        self._drives(out, mind, fps)
        self._subtitles(out, now)
        return out

    def _face(self, img, mind, t, now):
        x, y, bw, bh = [int(v) for v in t.box]
        c = mind.store.concept(t.concept_id) if t.known and t.concept_id else None
        role = c["role"] if c else None
        col = ROLE_COLOR.get(role, WHITE if c is None else BLUE)
        k = ease((now - self.locked.setdefault(t.id, now)) / LOCK_S)
        grow = int((1 - k) * bw * 0.5)
        x0, y0, ww, hh = x - grow, y - grow, bw + 2 * grow, bh + 2 * grow
        if c is None:
            dashed_rect(img, x, y, bw, bh, col)
            sy = y + int((math.sin(now * 3) * 0.5 + 0.5) * bh)
            cv2.line(img, (x + 2, sy), (x + bw - 2, sy), RED, 1, cv2.LINE_AA)
            brackets(img, x0, y0, ww, hh, RED, max(8, bw // 6))
            tag, name = "ANALYZING" + "." * (int(now * 3) % 4), ""
        else:
            cv2.rectangle(img, (x, y), (x + bw, y + bh), col, 1, cv2.LINE_AA)
            brackets(img, x0, y0, ww, hh, col, max(10, bw // 5), 2 if k < 1 else 3)
            if role == "admin":
                a = 0.5 + 0.5 * math.sin(now * 4)
                s = 4
                for px, py in ((x, y), (x + bw, y), (x, y + bh), (x + bw, y + bh)):
                    cv2.rectangle(img, (px - s, py - s), (px + s, py + s),
                                  tuple(int(v * (0.4 + 0.6 * a)) for v in col), -1)
            tag = ROLE_TAG.get(role, "KNOWN")
            full = (c["name"] or f"SUBJECT #{c['id']:04d}").upper()
            name = typed(full, self.named.setdefault((t.id, c["id"]), now), now)
        if k < 1:
            return
        # info card under the face, sliding out from the left edge of the box
        cw = max(bw, 170)
        top = y + bh + 6
        panel(img, x, top, x + cw, top + (40 if name else 22))
        cv2.rectangle(img, (x, top), (x + 3, top + (40 if name else 22)), col, -1)
        text(img, tag, (x + 9, top + 15), 0.38, col)
        if c is not None:
            conf = f"{t.sim:.2f}"
            text(img, conf, (x + cw - width(conf, 0.36) - 6, top + 15), 0.36, GREY)
        if name:
            text(img, name, (x + 9, top + 33), 0.5, WHITE)

    def _object(self, img, now):
        h, w = img.shape[:2]
        x, y, bw, bh = w // 4, h // 4, w // 2, h // 2
        dashed_rect(img, x, y, bw, bh, YELLOW, 12)
        p = int((math.sin(now * 2.5) * 0.5 + 0.5) * 10)
        brackets(img, x - p, y - p, bw + 2 * p, bh + 2 * p, YELLOW, 24)
        text(img, "LEARNING OBJECT", (x, y - 8), 0.42, YELLOW)

    def _status(self, img, mind, ears, speaking, now):
        h, w = img.shape[:2]
        x0, x1 = w - 210, w
        panel(img, x0, 0, x1, 52, 0.55)
        cx, cy = x0 + 26, 26
        level = min(1.0, getattr(ears, "level", 0.0) / 3) if ears else 0.0
        self.wave.append(level)
        if getattr(mind, "thinking", 0):
            label, col = "THINKING", YELLOW
            a = int(now * 360) % 360
            cv2.ellipse(img, (cx, cy), (12, 12), 0, a, a + 250, col, 2, cv2.LINE_AA)
        elif speaking:
            label, col = "SPEAKING", WHITE
            r = 6 + int(5 * abs(math.sin(now * 9)))
            cv2.circle(img, (cx, cy), r, col, 2, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 3, col, -1, cv2.LINE_AA)
        elif ears is not None and getattr(ears, "hearing", False):
            label, col = "HEARING", BLUE
            cv2.circle(img, (cx, cy), 9, col, -1, cv2.LINE_AA)
        elif ears is not None and hasattr(ears, "level"):
            label, col = "LISTENING", GREY
            cv2.circle(img, (cx, cy), 9, col, 1, cv2.LINE_AA)
        else:
            label, col = "TEXT ONLY", GREY
            cv2.circle(img, (cx, cy), 9, col, 1, cv2.LINE_AA)
        text(img, label, (x0 + 48, 22), 0.45, col)
        # mic waveform
        for i, v in enumerate(self.wave):
            bh = max(1, int(v * 14))
            cv2.line(img, (x0 + 48 + i * 5, 40 - bh), (x0 + 48 + i * 5, 40 + bh), col, 2)

    def _drives(self, img, mind, fps):
        h, w = img.shape[:2]
        d = mind.drives
        vals = [d.curiosity, d.social, d.energy]
        self.bars = [b + (v - b) * 0.15 for b, v in zip(self.bars, vals)]
        x0, y0 = w - 170, h - 92
        panel(img, x0 - 10, y0 - 18, w, h, 0.5)
        for i, (name, v) in enumerate(zip(("CURIOSITY", "SOCIAL", "ENERGY"), self.bars)):
            y = y0 + i * 20
            text(img, name, (x0, y), 0.32, GREY)
            cv2.rectangle(img, (x0 + 72, y - 7), (x0 + 152, y - 3), (60, 60, 60), -1)
            cv2.rectangle(img, (x0 + 72, y - 7), (x0 + 72 + int(80 * v), y - 3), YELLOW if i == 0 else WHITE, -1)
        text(img, f"{fps:4.1f} FPS  DET {mind.metrics['detect_ms']:.0f}MS  EMB {mind.metrics['embed_ms']:.0f}MS",
             (x0, h - 12), 0.3, GREY)

    def _subtitles(self, img, now):
        h, w = img.shape[:2]
        lines = []
        if self.heard and now - self.heard_t < SUB_S:
            lines += [("YOU", s, GREY) for s in textwrap.wrap(self.heard, 60)[:2]]
        if self.said and now - self.said_t < SUB_S:
            shown = typed(self.said, self.said_t, now)
            wrapped = textwrap.wrap(shown, 60)[-2:] or [""]
            if len(shown) < len(self.said) or int(now * 3) % 2:
                wrapped[-1] += "_"
            lines += [("MACHINE", s, WHITE) for s in wrapped]
        if not lines:
            return
        y = h - 18 - 22 * (len(lines) - 1)
        panel(img, 0, y - 22, w - 190, h, 0.5)
        prev = None
        for who, s, col in lines:
            if who != prev:
                text(img, who, (14, y), 0.36, YELLOW if who == "MACHINE" else GREY)
            text(img, s, (90, y), 0.5, col)
            prev, y = who, y + 22
