"""Language packs: intent patterns, phrases and stopwords for one language.

The core only sees intents and phrase keys, so adding a language means adding a
``packs/<code>.py`` that defines ``PACK``.
"""
from __future__ import annotations

import importlib
import random
import re
from dataclasses import dataclass, field

# Shared regex fragments for packs
NAME = r"(?P<x>[^\W\d_][\w'\-]{0,24}(?: [^\W\d_][\w'\-]{0,24})?)"
THING = r"(?P<x>[^\W\d_][\w'\- ]{0,40})"

# Intents the core understands. A pack must map (a subset of) these.
INTENT_KINDS = (
    "who_am_i", "who_are_you", "who_is", "what_is", "how_old", "how_are_you", "memory", "sleep", "bye",
    "forget", "name_me", "name_other", "correct", "positive", "negative", "name_object", "ask_fact", "fact",
)


@dataclass
class Intent:
    kind: str
    arg: str | None = None
    arg2: str | None = None
    text: str = ""


@dataclass
class LanguagePack:
    code: str
    intents: list[tuple[str, str]]                 # ordered (kind, regex); first match wins
    phrases: dict[str, list[str]]
    stopwords: set[str]
    asr_code: str                                  # language code given to the ASR model
    durations: dict[str, str] = field(default_factory=lambda: {
        "d": "{n} days", "h": "{n} hours", "m": "{n} minutes", "d1": "a day", "h1": "an hour", "m1": "a minute"})
    token: str = r"[^\W\d_]+"
    # What just happened, told to the language cortex so it can react in its own words.
    events: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        unknown = {k for k, _ in self.intents} - set(INTENT_KINDS)
        if unknown:
            raise ValueError(f"pack {self.code}: unknown intents {unknown}")
        self._compiled = [(k, re.compile(p, re.I)) for k, p in self.intents]
        self._tok = re.compile(self.token)

    def parse(self, text: str) -> Intent:
        t = re.sub(r"\s+", " ", text.strip().strip(".!")).strip()
        for kind, pat in self._compiled:
            m = pat.search(t)
            if m:
                d = m.groupdict()
                if kind == "fact":
                    return Intent(kind, d["s"].strip(), d["o"].strip(), t)
                x = d.get("x")
                return Intent(kind, x.strip().strip("?") if x else None, None, t)
        return Intent("chat", None, None, t)

    def content_words(self, text: str) -> list[str]:
        return [w for w in (t.lower() for t in self._tok.findall(text)) if len(w) > 1 and w not in self.stopwords]

    def say(self, key: str, stage: int = 2, **kw) -> str:
        text = random.choice(self.phrases[key]).format(**kw)
        if stage == 0 and key.startswith(("learned", "greet", "first")):
            # a newborn only echoes the key word
            w = next((str(v) for v in kw.values() if isinstance(v, str) and v), None)
            if w:
                return f"{w}... {w}!"
        return text

    def duration(self, seconds: float) -> str:
        for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
            n = int(seconds // size)
            if n >= 1 or unit == "m":
                n = max(1, n)
                return self.durations[unit + "1" if n == 1 else unit].format(n=n)
        return ""


def load_pack(code: str) -> LanguagePack:
    return importlib.import_module(f"{__package__}.packs.{code}").PACK


def stage_of(n_concepts: int, vocab: int) -> int:
    """Developmental stage: 0 = newborn, 1 = child, 2 = grown."""
    k = n_concepts + vocab
    return 0 if k < 3 else 1 if k < 25 else 2


STAGE_NAMES = {0: "newborn", 1: "child", 2: "grown"}
