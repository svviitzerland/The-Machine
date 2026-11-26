"""Global workspace: posted signals compete each tick and the most salient one is broadcast.

A key that won recently is damped (habituation), so the Machine doesn't greet the same face over and over.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(order=True)
class Signal:
    salience: float
    kind: str = field(compare=False)
    key: str = field(compare=False, default="")          # identity for habituation
    payload: dict[str, Any] = field(compare=False, default_factory=dict)
    source: str = field(compare=False, default="")
    ts: float = field(compare=False, default_factory=time.time)
    ttl: float = field(compare=False, default=3.0)


class Workspace:
    def __init__(self, habituation_s: float = 20.0, ignition: float = 0.25):
        self.habituation_s = habituation_s
        self.ignition = ignition                      # minimum effective salience to win
        self._pending: list[Signal] = []
        self._last_win: dict[str, float] = {}
        self._subs: list[Callable[[Signal], None]] = []
        self.stream: deque[tuple[float, str, str, float]] = deque(maxlen=200)
        self.current: Signal | None = None

    def subscribe(self, fn: Callable[[Signal], None]) -> None:
        self._subs.append(fn)

    def post(self, sig: Signal) -> None:
        self._pending.append(sig)

    def effective(self, s: Signal, now: float) -> float:
        last = self._last_win.get(s.key)
        if last is None or not s.key:
            return s.salience
        return s.salience * (1.0 - math.exp(-(now - last) / self.habituation_s))

    def step(self) -> Signal | None:
        now = time.time()
        self._pending = [s for s in self._pending if now - s.ts <= s.ttl]
        if not self._pending:
            self.current = None
            return None
        scored = sorted(((self.effective(s, now), s) for s in self._pending), key=lambda t: -t[0])
        eff, win = scored[0]
        if eff < self.ignition:
            self.current = None
            return None
        self._pending.remove(win)
        if win.key:
            self._last_win[win.key] = now
        self.current = win
        self.stream.append((now, win.kind, win.key, round(eff, 3)))
        for fn in self._subs:
            fn(win)
        return win
