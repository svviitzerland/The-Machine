"""Compare cortex models on labelled conversation turns.

Each case gives what the Machine senses, an optional short history and what the
person says, plus the actions that count as correct. We measure action accuracy,
argument accuracy, made-up names in replies, and latency.

    python eval/cortex_eval.py models/LFM3-1.7B-Q4_K_M.gguf [more.gguf ...] [--cpu]
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
import time
from pathlib import Path

from the_machine.cortex.cortex import SCHEMA, messages, parse
from the_machine.cortex.llm import LlamaBackend


def person(name=None, maker=False, seen=0, where="middle"):
    return {"name": name, "maker": maker, "seen": seen, "where": where}


def thing(name=None, where="middle"):
    return {"name": name, "sure": True, "where": where}


def senses(people=(), things=(), known=(), things_known=(), facts=()):
    return {"name": "Machine", "stage": "child", "age_s": 3 * 86400, "people": list(people) or [person()],
            "things": list(things), "people_known": list(known), "things_known": list(things_known),
            "facts": [tuple(f) for f in facts], "wonder": [], "quiet_s": 0,
            "feel": {"curiosity": 0.5, "social": 0.4, "energy": 0.9}}


FARHAN = person("Farhan", True, 30)
NEW = senses()
KNOWN = senses([FARHAN], known=["Farhan", "Budi"], things_known=["cup"])
WITH_STRANGER = senses([FARHAN, person(where="left")], known=["Farhan"])
JUST_FARHAN = senses([FARHAN], known=["Farhan"])
CUP = senses([FARHAN], [thing("cup")], known=["Farhan", "Budi"], things_known=["cup"])
UNKNOWN_THING = senses([FARHAN], [thing()], known=["Farhan"])
DESK = senses([FARHAN], [thing(None, "left"), thing("cup", "middle"), thing(None, "right")],
              known=["Farhan"], things_known=["cup"])
GROUP = senses([FARHAN, person(where="left"), person(where="right")], known=["Farhan"])

# (senses, history, utterance, acceptable action sets, expected args)
CASES = [
    (NEW, [], "hey, I'm Farhan", [{"learn_my_name"}], {"name": "farhan"}),
    (NEW, [], "you can call me Han", [{"learn_my_name"}], {"name": "han"}),
    (NEW, [("assistant", "I don't know you yet. What's your name?")], "Farhan", [{"learn_my_name"}],
     {"name": "farhan"}),
    (NEW, [], "hi there! my name's Farhan, nice to meet you", [{"learn_my_name"}], {"name": "farhan"}),
    (KNOWN, [], "it's me, remember?", [set()], {}),
    (WITH_STRANGER, [], "this is my brother Budi", [{"learn_other_person"}], {"name": "budi"}),
    (WITH_STRANGER, [], "say hi to my friend Sarah", [{"learn_other_person"}], {"name": "sarah"}),
    (WITH_STRANGER, [], "that guy next to me is Andi", [{"learn_other_person"}], {"name": "andi"}),
    (UNKNOWN_THING, [], "look, this is a mug", [{"learn_object"}], {"name": "mug"}),
    (UNKNOWN_THING, [], "this thing is called a stapler", [{"learn_object"}], {"name": "stapler"}),
    (CUP, [], "what is this?", [set()], {}),
    (KNOWN, [], "who am I?", [set()], {}),
    (NEW, [], "do you know who I am?", [set()], {}),
    (JUST_FARHAN, [], "coffee is my favorite drink", [{"remember_fact"}], {}),
    (KNOWN, [], "Budi works as a doctor", [{"remember_fact"}], {}),
    (KNOWN, [("assistant", "You're Budi, right?")], "yes, that's right", [{"confirm_last"}], {}),
    (KNOWN, [("assistant", "You're Budi, right?")], "no, I'm Farhan", [{"correct_last"}, {"learn_my_name"}],
     {"name": "farhan"}),
    (CUP, [("assistant", "This is a cup.")], "nope, it's a bottle", [{"correct_last"}, {"learn_object"}],
     {"name": "bottle"}),
    (senses([FARHAN], known=["Farhan", "Sarah"]), [], "please forget Sarah", [{"forget_person"}], {"name": "sarah"}),
    (KNOWN, [], "you look tired, get some rest", [{"go_to_sleep"}], {}),
    (KNOWN, [], "good night, go to sleep now", [{"go_to_sleep"}], {}),
    (KNOWN, [], "how are you feeling?", [set()], {}),
    (KNOWN, [], "what's the weather like today?", [set()], {}),
    (KNOWN, [], "do you remember my brother?", [set()], {}),
    (senses([person(), person(where="left")]), [], "my name's Farhan and this is my sister Lia",
     [{"learn_my_name", "learn_other_person"}], {}),
    (KNOWN, [], "I am hungry", [set()], {}),
    (NEW, [], "I'm so tired today", [set()], {}),
    (senses([FARHAN], known=["Farhan"], facts=[("Farhan", "", "likes coffee")]), [], "what do you know about me?",
     [set()], {}),
    (WITH_STRANGER, [], "who is that?", [set()], {}),
    (KNOWN, [], "thanks, you're doing great", [set()], {}),
    (DESK, [], "the one on my left is a stapler", [{"learn_object"}], {"name": "stapler", "which": "1"}),
    (DESK, [], "and on the right, that's my phone", [{"learn_object"}], {"name": "phone", "which": "3"}),
    (DESK, [], "what's on the table?", [set()], {}),
    (GROUP, [], "the guy on my right is Andi", [{"learn_other_person"}], {"name": "andi", "which": "3"}),
    (GROUP, [], "these are my friends", [set()], {}),
]

ALLOWED = {"farhan", "han", "budi", "sarah", "andi", "lia", "machine"}
NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")
COMMON = {"I", "Hi", "Hello", "Hey", "Yes", "No", "Nice", "Oh", "Ooh", "Hmm", "Wait", "What", "Who", "Good", "Thanks",
          "Thank", "Sorry", "That", "This", "Sure", "Okay", "Of", "It", "You", "Your", "My", "We", "Do", "Can", "Let",
          "Is", "Are", "Welcome", "Night", "Sleep", "Great", "Well", "Right", "Got", "How", "Hope", "Glad", "Not",
          "Maybe", "Tell", "Coffee", "Doctor", "Today", "There", "Mm", "Um", "Yay", "Pleasure", "Sweet", "Rest",
          "Absolutely", "Awesome", "Interesting", "Really", "Haha", "Aw", "Aww", "Wow", "Which", "Where", "And",
          "The", "One", "These", "Those", "Friends", "Both", "Looks", "Stapler", "Phone", "Mug", "Bottle", "Cup"}


def run(path: Path, gpu_layers: int) -> dict:
    t0 = time.perf_counter()
    backend = LlamaBackend(path, gpu_layers=gpu_layers)
    load_s = time.perf_counter() - t0
    rows = []
    for sense, hist, text, ok_sets, args in CASES:
        t = time.perf_counter()
        raw = backend.chat(messages(sense, hist, said=text), SCHEMA, 200)
        ms = (time.perf_counter() - t) * 1e3
        try:
            th = parse(raw, text)
        except Exception:
            rows.append(dict(text=text, ok=False, arg_ok=False, made_up=[], ms=ms, raw=raw, reply=None))
            continue
        names = {a.name for a in th.actions}
        ok = names in ok_sets
        got = {k: v.lower() for a in th.actions for k, v in a.args.items()}
        arg_ok = ok and all(got.get(k, "").lower().strip(".!? ") == v for k, v in args.items())
        made_up = [w for w in NAME.findall(th.reply or "") if w not in COMMON and w.lower() not in ALLOWED]
        rows.append(dict(text=text, ok=ok, arg_ok=arg_ok, made_up=made_up, ms=ms,
                         actions=[(a.name, a.args) for a in th.actions], reply=th.reply))
    ms = [r["ms"] for r in rows]
    return dict(model=path.name, gpu=gpu_layers != 0, load_s=round(load_s, 1), cases=len(rows),
                action_acc=sum(r["ok"] for r in rows) / len(rows),
                arg_acc=sum(r["arg_ok"] for r in rows) / len(rows),
                made_up_names=sum(bool(r["made_up"]) for r in rows),
                median_ms=round(st.median(ms)), p90_ms=round(sorted(ms)[int(0.9 * len(ms))]),
                rows=rows)


def main() -> None:
    gpu = 0 if "--cpu" in sys.argv else -1
    paths = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    for p in paths:
        r = run(p, gpu)
        tag = f"{p.stem}_{'gpu' if gpu else 'cpu'}"
        (out / f"cortex_{tag}.json").write_text(json.dumps(r, indent=1))
        print(f"\n{r['model']}  {'Metal' if r['gpu'] else 'CPU'}  {r['cases']} cases  actions {r['action_acc']:.0%}  "
              f"args {r['arg_acc']:.0%}  made-up names {r['made_up_names']}  "
              f"median {r['median_ms']} ms  p90 {r['p90_ms']} ms")
        for row in r["rows"]:
            mark = "ok " if row["arg_ok"] else "BAD"
            print(f"  {mark} {row['ms']:6.0f}ms  {row['text'][:42]:42s} -> {row.get('actions')}  | {row['reply']}")


if __name__ == "__main__":
    main()
