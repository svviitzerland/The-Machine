"""Language cortex: understands what people say and decides what to do and say.

A small local LLM reads what the Machine currently sees and remembers, plus the
last few turns, and answers with JSON: a list of actions (learn a name, learn a
fact, ...) and a short spoken reply. The actions are carried out by the Mind's
own memory, so the learning stays in the Machine; the LLM only interprets and
speaks. When no LLM is available, RuleCortex falls back to the language pack.
"""
from __future__ import annotations

import json
import re
import threading
import queue
from dataclasses import dataclass, field
from typing import Callable, Protocol

ACTIONS: dict[str, tuple[str, list[str]]] = {
    "learn_my_name": ("The person speaking tells you their name or what to call them.", ["name"]),
    "learn_other_person": ("The speaker introduces someone else who is visible.", ["name"]),
    "learn_object": ("The speaker tells you what the object they are showing you is called.", ["name"]),
    "remember_fact": ("The speaker tells you a fact worth remembering.", ["subject", "fact"]),
    "confirm_last": ("The speaker says your last answer was right.", []),
    "correct_last": ("The speaker says your last answer was wrong and gives the right name.", ["name"]),
    "forget_person": ("Your maker asks you to forget someone completely.", ["name"]),
    "go_to_sleep": ("The speaker tells you to sleep or rest.", []),
}

# Arguments are flat optional strings: grammar-constrained decoding handles that
# far more reliably in small models than a free-form nested object.
ARG_FIELDS = sorted({a for _, args in ACTIONS.values() for a in args})
SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {"do": {"type": "string", "enum": list(ACTIONS)},
                               **{f: {"type": "string", "maxLength": 60} for f in ARG_FIELDS}},
                "required": ["do"],
            },
        },
        "reply": {"type": "string", "maxLength": 240},
    },
    "required": ["actions", "reply"],
}


@dataclass
class Action:
    name: str
    args: dict[str, str] = field(default_factory=dict)


@dataclass
class Thought:
    actions: list[Action]
    reply: str | None          # None means: let the Mind phrase it from its templates


class Backend(Protocol):
    def chat(self, messages: list[dict], schema: dict, max_tokens: int) -> str: ...


def instructions(situation: str) -> str:
    acts = "\n".join(f"- {k}({', '.join(a)}): {d}" for k, (d, a) in ACTIONS.items())
    return f"""{situation}

Decide what to do and what to say, then answer with JSON only:
{{"actions": [{{"do": "<action>", "<argument>": "..."}}], "reply": "..."}}

Actions you can take (zero or more):
{acts}

Rules:
- Only use names, objects and facts that appear above or that the speaker just told you. Never invent them.
- If you don't know something, say so honestly and ask to be taught.
- The reply is spoken aloud: one or two short sentences, natural and in character, no emojis, no lists, no markdown.
- Don't repeat the same words you used in your last reply."""


def parse(raw: str) -> Thought:
    m = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(m.group(0) if m else raw)
    actions = []
    for a in data.get("actions") or []:
        if isinstance(a, dict) and a.get("do") in ACTIONS:
            need = ACTIONS[a["do"]][1]
            args = {k: str(a[k]).strip() for k in need if str(a.get(k) or "").strip()}
            if len(args) == len(need):
                actions.append(Action(a["do"], args))
    reply = str(data.get("reply") or "").strip() or None
    return Thought(actions, reply)


class LLMCortex:
    """Runs the LLM on its own thread so the camera loop never waits for it."""

    def __init__(self, backend: Backend, max_tokens: int = 120):
        self.backend, self.max_tokens = backend, max_tokens
        self._jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True, name="cortex").start()

    def think(self, situation: str, history: list[tuple[str, str]], text: str,
              done: Callable[[Thought | None, str], None]) -> None:
        self._jobs.put((situation, history, text, done))

    def _loop(self) -> None:
        while True:
            situation, history, text, done = self._jobs.get()
            messages = [{"role": "system", "content": instructions(situation)}]
            for role, content in history:
                messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": text})
            try:
                done(parse(self.backend.chat(messages, SCHEMA, self.max_tokens)), text)
            except Exception:
                done(None, text)          # the Mind falls back to the language pack
