"""Drives that make the Machine act on its own.

Curiosity grows until something new happens, social grows while nobody is in view,
and energy drains while awake and is restored by sleep.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import MindConfig


def clamp(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


@dataclass
class Drives:
    cfg: MindConfig
    curiosity: float = 0.5
    social: float = 0.5
    energy: float = 1.0

    def tick(self, people_in_view: int) -> None:
        self.curiosity = clamp(self.curiosity + self.cfg.curiosity_gain)
        self.social = clamp(self.social + (self.cfg.social_gain if people_in_view == 0 else -self.cfg.social_gain))
        self.energy = clamp(self.energy - self.cfg.energy_drain)

    def novelty(self, amount: float) -> None:
        self.curiosity = clamp(self.curiosity - amount)

    def interacted(self, amount: float = 0.2) -> None:
        self.social = clamp(self.social - amount)

    def slept(self) -> None:
        self.energy = 1.0

    @property
    def sleepy(self) -> bool:
        return self.energy < self.cfg.sleep_energy

    def as_dict(self) -> dict[str, float]:
        return {"curiosity": round(self.curiosity, 3), "social": round(self.social, 3), "energy": round(self.energy, 3)}
