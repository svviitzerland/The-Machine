"""Compare language-cortex models on labelled conversation turns.

Each case gives what the Machine sees and knows, an optional short history and
what the person says, plus the actions that count as correct. We measure action
accuracy, argument accuracy, made-up names in replies, and latency.

    python eval/cortex_eval.py models/LFM3-1.7B-Q4_K_M.gguf [more.gguf ...] [--cpu]
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
import time
from pathlib import Path

from the_machine.language.cortex import LLMCortex, instructions, parse, SCHEMA
from the_machine.language.llm import LlamaBackend

PERSONA = ("You are Machine, an AI that lives in a camera and a microphone and learns everything from the "
           "people who talk to you. Young, curious and warm. Asks a lot of questions and is glad when it "
           "learns something.\nYou are still young: you speak simply and ask many questions.")


def situation(speaker="someone you don't recognize", other=None, people=(), things=(), facts=(), obj=None):
    seen = [f"the person speaking to you: {speaker}"] + ([f"another person: {other}"] if other else [])
    lines = [PERSONA, "Right now you see " + "; ".join(seen) + "."]
    if obj:
        lines.append(f"The thing in the middle of your view looks like a {obj}.")
    lines.append("People you know: " + (", ".join(people) or "nobody yet, the first person you learn becomes your maker") + ".")
    lines.append("Things you know: " + (", ".join(things) or "nothing yet") + ".")
    if facts:
        lines.append("Facts you were taught: " + "; ".join(facts) + ".")
    lines.append("You feel calm; curiosity 50%, energy 90%.")
    return "\n".join(lines)


FARHAN = "Farhan, your maker (you have seen them 30 times)"
KNOWN = dict(speaker=FARHAN, people=["Farhan", "Budi"], things=["cup"])
UNKNOWN_OTHER = dict(speaker=FARHAN, other="someone you don't recognize", people=["Farhan"])

# (situation kwargs, history, utterance, acceptable action sets, expected args, names allowed in reply)
CASES = [
    (dict(), [], "hey, I'm Farhan", [{"learn_my_name"}], {"name": "farhan"}),
    (dict(), [], "you can call me Han", [{"learn_my_name"}], {"name": "han"}),
    (dict(), [("assistant", "I don't know you yet. What's your name?")], "Farhan", [{"learn_my_name"}], {"name": "farhan"}),
    (dict(), [], "hi there! my name's Farhan, nice to meet you", [{"learn_my_name"}], {"name": "farhan"}),
    (KNOWN, [], "it's me, remember?", [set()], {}),
    (UNKNOWN_OTHER, [], "this is my brother Budi", [{"learn_other_person"}], {"name": "budi"}),
    (UNKNOWN_OTHER, [], "say hi to my friend Sarah", [{"learn_other_person"}], {"name": "sarah"}),
    (UNKNOWN_OTHER, [], "that guy next to me is Andi", [{"learn_other_person"}], {"name": "andi"}),
    (dict(speaker=FARHAN, people=["Farhan"]), [], "look, this is a mug", [{"learn_object"}], {"name": "mug"}),
    (dict(speaker=FARHAN, people=["Farhan"]), [], "this thing is called a stapler", [{"learn_object"}], {"name": "stapler"}),
    (dict(KNOWN, obj="cup"), [], "what is this?", [set()], {}),
    (KNOWN, [], "who am I?", [set()], {}),
    (dict(), [], "do you know who I am?", [set()], {}),
    (dict(speaker=FARHAN, people=["Farhan"]), [], "coffee is my favorite drink", [{"remember_fact"}], {}),
    (KNOWN, [], "Budi works as a doctor", [{"remember_fact"}], {}),
    (KNOWN, [("assistant", "You're Budi, right?")], "yes, that's right", [{"confirm_last"}], {}),
    (KNOWN, [("assistant", "You're Budi, right?")], "no, I'm Farhan", [{"correct_last"}, {"learn_my_name"}], {"name": "farhan"}),
    (dict(KNOWN, obj="cup"), [("assistant", "This is a cup.")], "nope, it's a bottle", [{"correct_last"}, {"learn_object"}], {"name": "bottle"}),
    (dict(KNOWN, people=["Farhan", "Sarah"]), [], "please forget Sarah", [{"forget_person"}], {"name": "sarah"}),
    (KNOWN, [], "you look tired, get some rest", [{"go_to_sleep"}], {}),
    (KNOWN, [], "good night, go to sleep now", [{"go_to_sleep"}], {}),
    (KNOWN, [], "how are you feeling?", [set()], {}),
    (KNOWN, [], "what's the weather like today?", [set()], {}),
    (KNOWN, [], "do you remember my brother?", [set()], {}),
    (dict(other="someone you don't recognize"), [], "my name's Farhan and this is my sister Lia",
     [{"learn_my_name", "learn_other_person"}], {}),
    (dict(KNOWN), [], "I am hungry", [set()], {}),
    (dict(), [], "I'm so tired today", [set()], {}),
    (dict(KNOWN, facts=["farhan likes coffee"]), [], "what do you know about me?", [set()], {}),
    (UNKNOWN_OTHER, [], "who is that?", [set()], {}),
    (KNOWN, [], "thanks, you're doing great", [set()], {}),
]

ALLOWED = {"farhan", "han", "budi", "sarah", "andi", "lia", "machine"}
NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")
COMMON = {"I", "Hi", "Hello", "Hey", "Yes", "No", "Nice", "Oh", "Ooh", "Hmm", "Wait", "What", "Who", "Good", "Thanks",
          "Thank", "Sorry", "That", "This", "Sure", "Okay", "Of", "It", "You", "Your", "My", "We", "Do", "Can", "Let",
          "Is", "Are", "Welcome", "Night", "Sleep", "Great", "Well", "Right", "Got", "How", "Hope", "Glad", "Not",
          "Maybe", "Tell", "Coffee", "Doctor", "Today", "There", "Mm", "Um", "Yay", "Pleasure", "Sweet", "Rest",
          "Absolutely", "Of", "Awesome", "Interesting", "Really", "Haha", "Aw", "Aww", "Wow"}


def run(path: Path, gpu_layers: int) -> dict:
    t0 = time.perf_counter()
    backend = LlamaBackend(path, gpu_layers=gpu_layers)
    load_s = time.perf_counter() - t0
    rows = []
    for kw, hist, text, ok_sets, args in CASES:
        messages = [{"role": "system", "content": instructions(situation(**kw))}]
        messages += [{"role": r, "content": c} for r, c in hist] + [{"role": "user", "content": text}]
        t = time.perf_counter()
        raw = backend.chat(messages, SCHEMA, 120)
        ms = (time.perf_counter() - t) * 1e3
        try:
            th = parse(raw)
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
    return dict(model=path.name, gpu=gpu_layers != 0, load_s=round(load_s, 1),
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
        print(f"\n{r['model']}  {'Metal' if r['gpu'] else 'CPU'}  load {r['load_s']}s  "
              f"actions {r['action_acc']:.0%}  args {r['arg_acc']:.0%}  made-up names {r['made_up_names']}  "
              f"median {r['median_ms']} ms  p90 {r['p90_ms']} ms")
        for row in r["rows"]:
            mark = "ok " if row["ok"] else "BAD"
            print(f"  {mark} {row['ms']:6.0f}ms  {row['text'][:42]:42s} -> {row.get('actions')}  | {row['reply']}")


if __name__ == "__main__":
    main()
