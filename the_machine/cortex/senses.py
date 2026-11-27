"""Turns what the Mind senses and remembers into words for the cortex.

The Mind hands over plain data (who is in view, what it knows, what it wonders
about, how it feels). This is the only place that data becomes English, so a
new language means a new version of this file and the prompt in cortex.py.
"""
from __future__ import annotations

STAGES = {
    "newborn": "You are brand new and know almost nothing, so everything amazes you. You use very simple words.",
    "child": "You are still young. You learn something new almost every day and want to know everything.",
    "grown": "You have learned a lot by now and talk like someone who knows the people around them.",
}
SIDES = {"left": "on their left", "right": "on their right", "middle": "in the middle"}


def age(seconds: float) -> str:
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        n = int(seconds // size)
        if n >= 1:
            return f"{n} {unit}{'s' if n > 1 else ''}"
    return "less than a minute"


def _person(i: int, p: dict) -> str:
    who = "1 (talking to you)" if i == 0 else f"{i + 1} ({SIDES[p['where']]})"
    if p["name"]:
        return f"{who}: {p['name']}{', who made you' if p['maker'] else ''}"
    if p["seen"]:
        return f"{who}: seen {p['seen']} times, name unknown"
    return f"{who}: unknown"


def _thing(i: int, t: dict) -> str:
    if t["name"]:
        return f"{i + 1} ({SIDES[t['where']]}): {'' if t['sure'] else 'maybe '}{t['name']}"
    hint = f", looks like a {t['label']}" if t.get("label") else ""
    return f"{i + 1} ({SIDES[t['where']]}): unknown{hint}"


def _wonder(w: dict) -> str:
    k = w["kind"]
    if k == "who":
        return "who the person you keep seeing is" if w.get("seen") else "who the new person is"
    if k == "about":
        return f"what {w['name']} likes and does"
    if k == "object":
        return "what the unknown things are"
    if k == "confused":
        return f"how to tell {w['a']} and {w['b']} apart"
    return ""


def describe(s: dict) -> str:
    """The present moment and what the Machine remembers."""
    f = s["feel"]
    mood = ("sleepy" if f["energy"] < 0.25 else "very curious" if f["curiosity"] > 0.75
            else "lonely" if f["social"] > 0.8 else "calm")
    out = [f"You are {s['name']}, a young mind living in a camera and a microphone. Everything you know, you "
           f"learned from the people around you. You are nobody's assistant. You only know what you have seen "
           f"or been told, and when you don't know something you ask. {STAGES[s['stage']]}",
           "",
           f"Age: {age(s['age_s'])}. Mood: {mood}."
           + (" You have a head you can turn to look at things." if s.get("body") else "")]
    out.append("People in view: " + ("; ".join(_person(i, p) for i, p in enumerate(s["people"])) or "nobody"))
    if s["things"]:
        out.append("Things in view: " + "; ".join(_thing(i, t) for i, t in enumerate(s["things"])))
    out.append("People you know: " + (", ".join(s["people_known"]) or
                                      "nobody yet (the first person you learn is the one who made you)"))
    out.append("Things you know: " + (", ".join(s["things_known"]) or "nothing yet"))
    if s["facts"]:
        out.append("You were told: " + "; ".join(" ".join(x for x in f if x) for f in s["facts"]))
    wonders = [w for w in map(_wonder, s["wonder"]) if w]
    if wonders:
        out.append("You wonder: " + "; ".join(wonders))
    if s.get("quiet_s") is not None and s["quiet_s"] > 15:
        out.append(f"Nobody has spoken for {age(s['quiet_s'])}.")
    return "\n".join(out)


def event(e: dict) -> str:
    """Something the Machine just noticed, as a short line in parentheses."""
    k = e["kind"]
    made = f" {e['name']} made you." if e.get("maker") else ""
    text = {
        "arrived": lambda: f"You notice {e['name']} walk up to you.{made} Nobody has spoken yet.",
        "stranger": lambda: "You notice someone you don't know in front of you. Nobody has spoken yet.",
        "unnamed": lambda: f"The person you've seen {e['seen']} times without knowing their name is here again.",
        "left": lambda: f"{e['name']} just left.",
        "quiet": lambda: f"It has been quiet for {age(e['s'])}.",
        "tired": lambda: "Your energy is running low. Soon you'll need to sleep to sort out your memories.",
        "woke": lambda: (f"You just woke up. While sleeping you went over {e['replayed']} memories"
                         + (f" and found {e['found']} new faces you keep seeing" if e["found"] else "") + "."),
        "failed": lambda: {"no_face": "You tried to remember a face, but you can't see anyone.",
                           "no_vision": "You tried to look at it, but your camera isn't working.",
                           "not_allowed": "Only the person who made you can ask you to forget someone."}[e["why"]],
    }[k]()
    return f"({text})"
