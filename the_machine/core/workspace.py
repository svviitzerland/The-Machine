"""Global Workspace: how the Machine "thinks".

Specialist processes (vision, hearing, memory, drives) run in parallel and
post *signals* (candidate contents of consciousness), each with a salience.
Every tick the workspace lets them compete; the single winner is *broadcast*
to all processes, which react to it (speak, learn, ask, sleep). Losing
signals fade. This serial bottleneck over parallel specialists is the core of
Global Workspace Theory (Baars; Dehaene) and gives the Machine a readable
"stream of consciousness" that we log and measure.

Habituation: content that won recently is suppressed, so the Machine does not
greet the same face forever, and attention moves on (inhibition of return).
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
        self.ignition = ignition                      # minimum salience to become conscious
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
