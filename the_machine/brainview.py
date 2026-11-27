"""The cortex's neural network, live: one column per layer, the strongest neurons of
each layer lit by their real activation for the token it is thinking about, and
the links between them. Each new token sends a wave from the first layer to the last.
"""
from __future__ import annotations

import json
import math
import time

import cv2
import numpy as np

PALETTE = [(60, 140, 255), (70, 70, 230), (100, 200, 120), (170, 215, 240), (80, 210, 240), (200, 190, 90)]
CYAN = (230, 210, 90)
RED = (80, 80, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class BrainView:
    def __init__(self, size: tuple[int, int] = (1100, 560), neurons: int = 36, links: int = 8):
        self.w, self.h = size
        self.k, self.m = neurons, links
        self.shown: np.ndarray | None = None
        self.last_tokens, self.wave_t, self.rate_t, self.rate_n, self.rate = 0, 0.0, time.time(), 0, 0.0
        self._bg = self._background()
        self._layout_for = None

    def _background(self) -> np.ndarray:
        y = np.linspace(0, 1, self.h, dtype=np.float32)[:, None]
        x = np.linspace(0, 1, self.w, dtype=np.float32)[None, :]
        glow = np.exp(-(((x - 0.45) / 0.7) ** 2 + ((y - 0.5) / 0.6) ** 2))
        return np.dstack([38 + 55 * glow, 16 + 22 * glow, 14 + 20 * glow + 20 * y]).astype(np.uint8)

    def _layout(self, layers: int) -> None:
        """A spindle like the old drawings of the Machine: a wide bulb near the input, then a long tube."""
        i = np.arange(layers, dtype=np.float32)
        h = 0.26 + 0.52 * np.exp(-((i - 1.2) / 2.0) ** 2) + 0.12 * np.exp(-((i - 4.5) / 1.6) ** 2)
        h[-1] *= 0.6
        gaps = 1.0 + 1.2 * np.exp(-i / 3.0)
        x = np.concatenate([[0], np.cumsum(gaps[:-1])])
        x = 0.07 + 0.80 * x / x[-1]
        self.cols = [(int(self.w * xx), int(self.h * 0.88 * hh)) for xx, hh in zip(x, h)]
        self.ys = [np.linspace(self.h / 2 - hh / 2, self.h / 2 + hh / 2, self.k).astype(int) for _, hh in self.cols]
        rng = np.random.default_rng(1)
        self.base = [(rng.integers(0, self.k, 14), rng.integers(0, self.k, 14)) for _ in range(layers - 1)]
        self._layout_for = layers

    def _activity(self, frame: np.ndarray) -> np.ndarray:
        """Pool each layer's neurons into k groups (strongest wins), scaled per layer."""
        layers, width = frame.shape
        g = width // self.k
        a = np.abs(frame[:, :g * self.k]).reshape(layers, self.k, g).max(-1)
        return np.clip(a / (np.percentile(a, 95, axis=1, keepdims=True) + 1e-6), 0, 1)

    def draw(self, mind) -> np.ndarray:
        now = time.time()
        backend = getattr(mind.cortex, "backend", None)
        tap = getattr(backend, "tap", None)
        out = self._bg.copy()
        if tap is None:
            cv2.putText(out, "NO CORTEX TO WATCH", (40, self.h // 2), FONT, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
            return out
        tap.on = True
        frame, tokens = tap.read()
        if self._layout_for != tap.layers:
            self._layout(tap.layers)
        target = self._activity(frame)
        if tokens != self.last_tokens:
            self.wave_t, self.last_tokens = now, tokens
            self.rate_n += 1
        if now - self.rate_t > 1:
            self.rate, self.rate_n, self.rate_t = self.rate_n * tap.every / (now - self.rate_t), 0, now
        busy = now - self.wave_t < 0.8
        self.shown = target if self.shown is None else self.shown * 0.7 + target * 0.3
        L = tap.layers
        sweep = (now - self.wave_t) / 0.35 * L                     # which layer the wave has reached
        col_on = np.clip(1.2 - np.abs(np.arange(L) - sweep) / 3, 0, 1) if busy else np.zeros(L)
        idle = 0.18 + 0.06 * math.sin(now * 1.3)
        level = self.shown * (idle + (1 - idle) * np.maximum(col_on, 0.35 if busy else 0))[:, None]

        glow = np.zeros_like(out)
        for i in range(L - 1):
            (x0, _), (x1, _) = self.cols[i], self.cols[i + 1]
            for a, b in zip(*self.base[i]):                        # the faint lattice that is always there
                cv2.line(glow, (x0, self.ys[i][a]), (x1, self.ys[i + 1][b]), (60, 50, 45), 1, cv2.LINE_AA)
            top_a = np.argsort(level[i])[-self.m:]
            top_b = np.argsort(level[i + 1])[-self.m:]
            for a in top_a:
                for b in top_b:
                    s = math.sqrt(level[i, a] * level[i + 1, b])
                    if s < 0.08:
                        continue
                    c = PALETTE[(a * 7 + b * 3 + i) % len(PALETTE)]
                    cv2.line(glow, (x0, self.ys[i][a]), (x1, self.ys[i + 1][b]),
                             tuple(int(v * min(1.0, s * 0.75)) for v in c), 1, cv2.LINE_AA)
        for i, (x, hh) in enumerate(self.cols):
            for j, y in enumerate(self.ys[i]):
                v = level[i, j]
                cv2.circle(glow, (x, int(y)), 2, tuple(int(c * (0.2 + 0.6 * v)) for c in (230, 240, 255)), -1,
                           cv2.LINE_AA)
            cv2.ellipse(glow, (x, self.h // 2), (int(12 + 10 * col_on[i]), hh // 2 + 6), 0, 0, 360,
                        tuple(int(c * (0.15 + 0.6 * col_on[i])) for c in CYAN), 1, cv2.LINE_AA)
        bloom = cv2.GaussianBlur(glow, (0, 0), 5)
        out = cv2.add(out, cv2.addWeighted(glow, 0.85, bloom, 0.9, 0))

        for i, (x, hh) in enumerate(self.cols):
            top = self.h // 2 - hh // 2 - 14
            cv2.putText(out, f"L{i + 1:02d}", (x - 12, top), FONT, 0.3, CYAN, 1, cv2.LINE_AA)
            if col_on[i] > 0.3:
                n = int(np.argmax(np.abs(frame[i])))                # the real neuron that fired hardest
                cv2.putText(out, f"N{n}", (x - 12, top - 12), FONT, 0.3, RED, 1, cv2.LINE_AA)
        cv2.putText(out, "INPUT", (self.cols[0][0] - 20, self.h // 2 + self.cols[0][1] // 2 + 26), FONT, 0.35,
                    (200, 200, 200), 1, cv2.LINE_AA)
        xo = self.cols[-1][0] + 22
        cv2.putText(out, "OUTPUT", (xo, self.h // 2 - 8), FONT, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        word = self._speaking(backend.recent)
        if word and busy:
            cv2.putText(out, word[-14:].upper(), (xo, self.h // 2 + 12), FONT, 0.4, RED, 1, cv2.LINE_AA)
        caption = (f"CORTEX {getattr(backend, 'name', '').upper()}   {L} LAYERS x {tap.width} NEURONS   "
                   f"{self.k} SHOWN PER LAYER   {self.rate:4.1f} TOKENS/S" if busy else
                   f"CORTEX {getattr(backend, 'name', '').upper()}   {L} LAYERS x {tap.width} NEURONS   RESTING")
        cv2.putText(out, caption, (16, self.h - 16), FONT, 0.36, (190, 190, 190), 1, cv2.LINE_AA)
        if busy:
            cv2.putText(out, self._speaking(backend.recent, full=True)[-90:], (16, 28), FONT, 0.42,
                        (235, 235, 235), 1, cv2.LINE_AA)
        cv2.rectangle(out, (0, 0), (self.w - 1, self.h - 1), CYAN, 1)
        return out

    @staticmethod
    def _speaking(raw: str, full: bool = False) -> str:
        """What it is saying, pulled out of the JSON it is writing."""
        i = raw.rfind('"reply"')
        if i < 0:
            return "" if not full else raw.replace("\n", " ")
        text = raw[i:].split(":", 1)[-1].strip().lstrip('"').rstrip('"}').replace('\\"', '"')
        return text if full else (text.split()[-1] if text.split() else "")

    def close(self, mind) -> None:
        tap = getattr(getattr(mind.cortex, "backend", None), "tap", None)
        if tap is not None:
            tap.on = False
