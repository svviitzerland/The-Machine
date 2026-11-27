"""The cortex: a small local LLM that is the Machine's inner voice.

It gets what the Machine senses and remembers, the last few turns, and either
what someone just said or something it noticed. It answers with JSON: actions
that change its own memory, then what it says out loud. The reply is streamed
sentence by sentence, so the voice starts before the model has finished.
"""
from __future__ import annotations

import json
import queue
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .senses import describe, event as describe_event

# name: (when to use it, required arguments, optional arguments)
ACTIONS: dict[str, tuple[str, list[str], list[str]]] = {
    "learn_my_name": ("the person talking to you says their name, in any words", ["name"], []),
    "learn_other_person": ("they say the name of another person in view", ["name"], ["which"]),
    "learn_object": ("they say what a thing in view is", ["name"], ["which"]),
    "remember_fact": ("they tell you something about a person, a thing or themselves", ["subject", "fact"], []),
    "confirm_last": ("they say your last guess was right", [], []),
    "correct_last": ("they say your last guess was wrong and give the right name", ["name"], ["which"]),
    "forget_person": ("the person who made you asks you to forget someone", ["name"], []),
    "go_to_sleep": ("you are told to rest, or you decide you need to", [], []),
    "look": ("turn your head. at=person or thing, with which, to look at someone or something in view; "
             "or at=left, right, up, down or ahead, as seen from you", ["at"], ["which"]),
}

GAZE = ("person", "thing", "left", "right", "up", "down", "ahead")
ARG_FIELDS = sorted({a for _, req, opt in ACTIONS.values() for a in req + opt})
SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {"do": {"type": "string", "enum": list(ACTIONS)},
                               **{f: {"type": "string", "maxLength": 60} for f in ARG_FIELDS
                                  if f not in ("which", "at")},
                               "at": {"type": "string", "enum": list(GAZE)},
                               "which": {"type": "string", "enum": [str(i) for i in range(1, 10)]}},
                "required": ["do"],
            },
        },
        "reply": {"type": "string"},
    },
    "required": ["actions", "reply"],
}

# A few moments that show the format. Other names and things than the ones it will meet.
EXAMPLES = """"oh hey, Rina here" -> {"actions": [{"do": "learn_my_name", "name": "Rina"}], "reply": "Rina! Hi. I'll remember your face."}
"and that's my cousin Dimas" -> {"actions": [{"do": "learn_other_person", "name": "Dimas", "which": "2"}], "reply": "Hello Dimas!"}
"check it out, it's a pen" -> {"actions": [{"do": "learn_object", "name": "pen", "which": "1"}], "reply": "A pen. What do you use it for?"}
"I really love tea" -> {"actions": [{"do": "remember_fact", "subject": "me", "fact": "loves tea"}], "reply": "You love tea. Hot or cold?"}
"no, that's a spoon" -> {"actions": [{"do": "correct_last", "name": "spoon"}], "reply": "Oh, a spoon. I'll fix that."}
"what can you do?" -> {"actions": [], "reply": "I can see you, hear you and remember what you teach me."}
(It has been quiet for 1 minute.) -> {"actions": [], "reply": ""}"""


# Only shown when it has a head to turn.
BODY_EXAMPLES = """
"look at Dimas" -> {"actions": [{"do": "look", "at": "person", "which": "2"}], "reply": "Hi Dimas."}
"look up there" -> {"actions": [{"do": "look", "at": "up"}], "reply": "Up here?"}"""


def prompt(senses: dict, examples: bool = True) -> str:
    body = senses.get("body")
    acts = "\n".join(f"- {k}({', '.join(r + [o + '?' for o in opt])}): {d}" for k, (d, r, opt) in ACTIONS.items()
                     if k != "look" or body)
    return f"""{describe(senses)}

Answer in JSON: {{"actions": [...], "reply": "..."}}.
Actions change your memory. Take one only when someone has just told you something:
{acts}
"which" is the number of the person or thing they mean.
"reply" is what you say out loud, in your own words, or empty to stay quiet.
A line in (parentheses) is something you noticed, not words said to you.""" + (
        f"\n\nFor example:\n{EXAMPLES}{BODY_EXAMPLES if body else ''}" if examples else "")


def messages(senses: dict, history: list[tuple[str, str]], said: str | None = None,
             noticed: dict | None = None) -> list[dict]:
    out = [{"role": "system", "content": prompt(senses)}]
    out += [{"role": r, "content": c} for r, c in history]
    out.append({"role": "user", "content": said if said is not None else describe_event(noticed)})
    return out


@dataclass
class Action:
    name: str
    args: dict[str, str] = field(default_factory=dict)


@dataclass
class Thought:
    actions: list[Action]
    reply: str


_EMPTY = {"none", "none given", "unknown", "n/a", "null", "?"}
_NOT_NAMES = {"i", "me", "you", "he", "she", "they", "it", "this", "that", "my", "your"}
# Names it would store must have been said out loud, so a guess never becomes a memory.
# Going to sleep on something someone said needs them to have said something about sleep.
_SLEEP_WORDS = re.compile(r"\b(sleep\w*|rest\w*|nap\w*|bed\w*|tired|night|lie down)\b", re.I)
_GROUNDED = {"learn_my_name", "learn_other_person", "learn_object", "correct_last", "forget_person"}


def _said(value: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(value.lower()) + r"\b", text.lower()) is not None


def actions_from(data: dict, said: str) -> list[Action]:
    out = []
    for a in data.get("actions") or []:
        if not isinstance(a, dict) or a.get("do") not in ACTIONS:
            continue
        _, need, opt = ACTIONS[a["do"]]
        args = {k: str(a[k]).strip() for k in need + opt
                if str(a.get(k) or "").strip() and str(a[k]).strip().lower() not in _EMPTY}
        if any(k not in args for k in need):
            continue
        if a["do"] in _GROUNDED and (args["name"].lower() in _NOT_NAMES or not _said(args["name"], said)):
            continue
        if a["do"] == "go_to_sleep" and said and not _SLEEP_WORDS.search(said):
            continue
        out.append(Action(a["do"], args))
    return out


def parse(raw: str, said: str = "") -> Thought:
    m = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(m.group(0) if m else raw)
    return Thought(actions_from(data, said), str(data.get("reply") or "").strip())


class ReplyStream:
    """Reads the reply out of the JSON while it is still being written."""

    KEY = re.compile(r'"reply"\s*:\s*"')
    END = re.compile(r'[.!?]+["\')\]]*\s')
    ESC = {"n": " ", "t": " ", "r": " ", '"': '"', "\\": "\\", "/": "/", "b": "", "f": ""}

    def __init__(self, on_head: Callable[[str], None], on_sentence: Callable[[str], None]):
        self.on_head, self.on_sentence = on_head, on_sentence
        self.raw, self.pos, self.text, self.sent, self.closed = "", None, "", 0, False

    def feed(self, piece: str) -> None:
        self.raw += piece
        if self.pos is None:
            m = self.KEY.search(self.raw)
            if not m:
                return
            self.pos = m.end()
            self.on_head(self.raw[:m.start()])
        self._decode()
        while (m := self.END.search(self.text, self.sent)):
            self._emit(m.end())

    def finish(self) -> None:
        self._emit(len(self.text))

    def _emit(self, end: int) -> None:
        s = self.text[self.sent:end].strip()
        self.sent = end
        if s:
            self.on_sentence(s)

    def _decode(self) -> None:
        r = self.raw
        while not self.closed and self.pos < len(r):
            c = r[self.pos]
            if c == '"':
                self.closed = True
                self.text += " "
            elif c == "\\":
                if self.pos + 1 >= len(r):
                    return
                e = r[self.pos + 1]
                if e == "u":
                    if self.pos + 6 > len(r):
                        return
                    self.text += chr(int(r[self.pos + 2:self.pos + 6], 16))
                    self.pos += 4
                else:
                    self.text += self.ESC.get(e, e)
                self.pos += 1
            else:
                self.text += c
            self.pos += 1


class Backend(Protocol):
    def chat(self, messages: list[dict], schema: dict, max_tokens: int) -> str: ...


class LLMCortex:
    """Runs the model on its own thread so the camera loop never waits for it.

    ``think`` reports back through ``out(kind, value)``: "act" with the actions
    (always before anything is said), "say" once per sentence, then "done" with
    the whole Thought, or None if the model failed.
    """

    def __init__(self, backend: Backend, max_tokens: int = 200):
        self.backend, self.max_tokens = backend, max_tokens
        self._jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True, name="cortex").start()

    def think(self, senses: dict, history: list[tuple[str, str]], out: Callable[[str, object], None],
              said: str | None = None, noticed: dict | None = None) -> None:
        self._jobs.put((messages(senses, history, said, noticed), said or "", out))

    def _loop(self) -> None:
        while True:
            msgs, said, out = self._jobs.get()
            acted = False

            def head(prefix: str) -> None:
                nonlocal acted
                try:
                    data = json.loads(prefix.rstrip().rstrip(",") + "}")
                except ValueError:
                    return
                acted = True
                out("act", actions_from(data, said))

            stream = ReplyStream(head, lambda s: out("say", s))
            try:
                if hasattr(self.backend, "stream"):
                    raw = self.backend.stream(msgs, SCHEMA, self.max_tokens, stream.feed)
                else:
                    raw = self.backend.chat(msgs, SCHEMA, self.max_tokens)
                    stream.feed(raw)
                thought = parse(raw, said)
            except Exception:
                if not acted:
                    out("act", [])
                out("done", None)
                continue
            if not acted:
                out("act", thought.actions)
            stream.finish()
            out("done", thought)
