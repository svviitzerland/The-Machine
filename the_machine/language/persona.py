"""Personas: who the Machine sounds like.

A persona picks a voice, a base speaking rate and how each mood changes the
way it talks. Moods come from the drives and from what is being said, so the
same sentence sounds different when it is curious, happy or tired.
Add a persona by dropping a file in ``personas/`` that defines ``PERSONA``.
"""
from __future__ import annotations

import importlib
import random
from dataclasses import dataclass, field


@dataclass
class Mood:
    rate: float = 0.0                      # duration change: negative is faster, positive slower
    openers: list[str] = field(default_factory=list)
    chance: float = 0.0                    # how often an opener is used


@dataclass
class Persona:
    name: str
    voice: str
    description: str
    rate: float = 0.0
    moods: dict[str, Mood] = field(default_factory=dict)

    def style(self, text: str, mood: str, stage: int) -> tuple[str, float]:
        """Return the text with an optional mood opener, and the speaking rate."""
        m = self.moods.get(mood, Mood())
        # A newborn echoes words, so it gets no openers; it just speaks a little slower.
        if stage == 0:
            return text, self.rate + m.rate + 0.15
        if m.openers and random.random() < m.chance:
            opener = random.choice(m.openers)
            text = f"{opener} {text}"
        return text, self.rate + m.rate


def load_persona(name: str) -> Persona:
    return importlib.import_module(f"{__package__}.personas.{name}").PERSONA


# Which mood fits which phrase. Anything not listed is "calm".
PHRASE_MOODS = {
    "greet_known": "happy", "greet_admin": "warm", "first_admin": "warm", "learned_person": "happy",
    "learned_obj": "happy", "thanks": "happy", "fact_learned": "happy",
    "ask_name": "curious", "ask_stranger": "curious", "obj_unknown": "curious", "dont_know_you": "curious",
    "confused": "curious", "unknown_word": "curious", "confusion_q": "curious", "obj_maybe": "unsure",
    "you_maybe": "unsure", "sorry": "sorry", "sleepy": "tired", "sleep_start": "tired",
}


def mood_for(key: str, curiosity: float, energy: float) -> str:
    if energy < 0.25 and key not in ("thanks",):
        return "tired"
    mood = PHRASE_MOODS.get(key, "calm")
    if mood == "calm" and curiosity > 0.8:
        return "curious"
    return mood
