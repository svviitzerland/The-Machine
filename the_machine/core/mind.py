"""The Mind: perception, memory, drives and the global workspace.

It holds no words. What it senses goes to the cortex as plain data, and the
cortex decides what to say and which memories to change. The Mind does no I/O
of its own either, so tests drive it with fake faces, fake objects and a fake
cortex.
"""
from __future__ import annotations

import queue
import re
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from ..config import Config, MemoryConfig
from ..memory.consolidation import Consolidator, SleepReport
from ..memory.lexicon import Lexicon
from ..memory.semantic import PrototypeMemory
from ..memory.store import Store
from .drives import Drives
from .ethics import Ethics
from .workspace import Signal, Workspace

STAGES = ("newborn", "child", "grown")
SELF_WORDS = {"me", "i", "my", "myself", "mine"}       # what a speaker calls themselves in a fact
YOU_WORDS = {"you", "yourself"}


def stage_of(n_concepts: int, vocab: int) -> int:
    k = n_concepts + vocab
    return 0 if k < 3 else 1 if k < 25 else 2


def object_memory_cfg(base: MemoryConfig) -> MemoryConfig:
    """Object embeddings live on a different similarity scale than SFace, so they get their own operating point."""
    return replace(base, tau_known=base.obj_tau_known, tau_min=base.obj_tau_min, tau_max=base.obj_tau_max,
                   spawn_tau=base.obj_spawn_tau, spread_k=base.obj_spread_k)


@dataclass
class Pending:
    """An open teaching window: keep absorbing samples of the same track for a few seconds."""
    kind: str                     # face | object
    concept_id: int
    track_id: int
    until: float
    absorbed: int = 0


@dataclass
class Assertion:
    """Something the Machine claimed or was taught, which a following "yes" or "no" refers to."""
    kind: str
    concept_id: int | None
    word: str | None = None
    vecs: list[np.ndarray] = field(default_factory=list)
    episode_ids: list[int] = field(default_factory=list)


def where(box, frame_w: int) -> str:
    """Position from the point of view of the person facing the camera (their left is the image's right)."""
    cx = (box[0] + box[2] / 2) / max(frame_w, 1)
    return "right" if cx < 0.38 else "left" if cx > 0.62 else "middle"


def _overlap(a, b) -> float:
    """Fraction of box a covered by box b."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) / max(a[2] * a[3], 1e-6)


class Mind:
    def __init__(self, cfg: Config, store: Store, say: Callable[[str], None],
                 face_engine=None, object_engine=None, cortex=None, self_training: bool = True):
        self.cfg, self.store, self._say = cfg, store, say
        # Voice.say takes a speaking rate; plain sinks like print or list.append don't.
        self._say_rate = getattr(say, "__func__", None) is not None and "rate" in say.__func__.__code__.co_varnames
        self.faces = PrototypeMemory(store, cfg.memory, "face")
        self.objects = PrototypeMemory(store, object_memory_cfg(cfg.memory), "object")
        self.lex = Lexicon(store, cfg.memory)
        self.sleeper = Consolidator(store, cfg.memory, self.faces, self.lex)
        self.ethics = Ethics(store)
        self.drives = Drives(cfg.mind)
        self.ws = Workspace(cfg.mind.habituation_s)
        self.ws.subscribe(self._act)
        self.face_engine, self.object_engine, self.cortex = face_engine, object_engine, cortex
        from ..perception.vision import Tracker
        self.tracker = Tracker(cfg.vision)            # faces
        self.things = Tracker(cfg.vision)             # objects
        self.self_training = self_training
        self._out: queue.Queue = queue.Queue()        # what the cortex sends back
        self.thinking = 0
        self.frame: np.ndarray | None = None
        self.pending: Pending | None = None
        self.last: Assertion | None = None
        self._view: dict[str, list] = {"people": [], "things": [], "guesses": []}
        self._frame_i = 0
        self._last_store: dict[int, float] = {}
        self._last_self_train: dict[int, float] = {}
        self._last_exchange = self._last_muse = time.time()
        self._last_ask = self._last_tired = 0.0
        self.thought = ""
        self.sleeping = False
        self.metrics = {"frames": 0, "embeds": 0, "detect_ms": 0.0, "embed_ms": 0.0, "objects_ms": 0.0}

    @property
    def stage(self) -> int:
        return stage_of(len(self.faces.concepts()) + len(self.objects.concepts()), self.lex.vocabulary_size())

    def name_of(self, cid: int | None) -> str | None:
        if cid is None:
            return None
        c = self.store.concept(cid)
        if c and c["name"]:
            return c["name"]
        w = self.lex.word_for(cid)
        return w[0] if w else None

    def visible(self) -> list:
        return sorted(self.tracker.tracks.values(), key=lambda t: -t.area)

    def objects_in_view(self) -> list:
        return sorted(self.things.tracks.values(), key=lambda t: t.box[0])

    def speaker(self):
        """Who is talking: the largest (closest) face in view."""
        v = self.visible()
        return v[0] if v else None

    # perception

    def see(self, frame: np.ndarray, now: float | None = None) -> None:
        now = now or time.time()
        self.frame = frame
        self._frame_i += 1
        self.metrics["frames"] += 1
        if self.face_engine is not None:
            self._see_faces(frame, now)
        if self.object_engine is not None and (self._frame_i - 1) % self.cfg.vision.objects_every_n == 0:
            self._see_things(frame, now)
        self._continue_teaching(now)

    def _see_faces(self, frame, now: float) -> None:
        t0 = time.perf_counter()
        rows = self.face_engine.detect(frame)
        self.metrics["detect_ms"] = (time.perf_counter() - t0) * 1e3
        active, gone = self.tracker.update(rows, now)
        for t in active:
            if t.frames == 1 or self._frame_i % self.cfg.vision.embed_every_n == 0 or self._teaching("face", t):
                t1 = time.perf_counter()
                e = self.face_engine.embed(frame, t.row)
                self.metrics["embed_ms"] = (time.perf_counter() - t1) * 1e3
                self.metrics["embeds"] += 1
                t.embeddings.append(e)
                self._recognize_face(t, e, now)
        for t in gone:
            self._remember_face(t, now, force=True)
            if t.known and t.announced and t.last - t.first > 5 and (n := self.name_of(t.concept_id)):
                self.ws.post(Signal(0.35, "notice", f"left:{t.concept_id}", {"kind": "left", "name": n}, "vision"))

    def _teaching(self, kind: str, t) -> bool:
        return bool(self.pending and self.pending.kind == kind and self.pending.track_id == t.id)

    def _recognize_face(self, t, e: np.ndarray, now: float) -> None:
        r = self.faces.recognize(e)
        t.votes[r.concept_id if r.known else -1] += 1
        cid, _ = t.votes.most_common(1)[0]
        t.concept_id = None if cid == -1 else cid
        t.known, t.sim = cid != -1, r.sim
        if sum(t.votes.values()) >= 3 and not t.announced:
            t.announced = True
            name = self.name_of(t.concept_id) if t.known else None
            if t.known:
                self.store.touch_concept(t.concept_id)
            if name:
                self.ws.post(Signal(0.55 + 0.4 * self.drives.social, "notice", f"greet:{t.concept_id}",
                                    {"kind": "arrived", "name": name, "maker": self.ethics.is_admin(t.concept_id)},
                                    "vision"))
            else:
                self.ws.post(Signal(0.45 + 0.5 * self.drives.curiosity, "notice", f"ask:{t.id}",
                                    {"kind": "stranger"}, "vision", ttl=5.0))
        # Self-training only takes very confident sightings, and rarely, so wrong labels stay
        # rare enough for sleep to catch.
        if (self.self_training and t.known and r.known and r.concept_id == t.concept_id
                and r.sim >= r.threshold + 0.15 and r.margin >= 0.15
                and now - self._last_self_train.get(t.id, 0) > 10):
            self._last_self_train[t.id] = now
            self.faces.learn(e, t.concept_id)
            self.store.add_episode("see", modality="face", concept_id=t.concept_id, vec=e,
                                   salience=0.4, exemplar=True, source="self")
        self._remember_face(t, now)

    def _remember_face(self, t, now: float, force: bool = False) -> None:
        """Log a sighting. Unknown faces logged here feed stranger discovery during sleep."""
        if not force and now - self._last_store.get(t.id, 0) < 5:
            return
        m = t.mean_embedding()
        if m is None:
            return
        self._last_store[t.id] = now
        self.store.add_episode("see", modality="face", concept_id=t.concept_id if t.known else None,
                               vec=m, salience=0.6 if t.known else 0.5)

    def _see_things(self, frame, now: float) -> None:
        t0 = time.perf_counter()
        rows = [r for r in self.object_engine.detect(frame)
                if not any(_overlap(r[:4], f.box) > 0.5 for f in self.tracker.tracks.values())]  # a face is not a thing
        active, _ = self.things.update(np.array(rows, np.float32) if rows else np.zeros((0, 5), np.float32), now)
        for t in active:
            if t.frames == 1 or self._teaching("object", t) or t.frames % self.cfg.vision.embed_every_n == 0:
                e = self.object_engine.embed(frame, t.row)
                t.embeddings.append(e)
                r = self.objects.recognize(e)
                t.votes[r.concept_id if r.known else -1] += 1
                cid, _ = t.votes.most_common(1)[0]
                t.concept_id = None if cid == -1 else cid
                t.known, t.sim = cid != -1, r.sim - r.threshold
        self.metrics["objects_ms"] = (time.perf_counter() - t0) * 1e3

    # hearing and thinking

    def hear(self, text: str) -> None:
        self.store.add_episode("hear", text=text, salience=0.6)
        self._last_exchange = time.time()
        self.drives.interacted()
        in_view = [t.concept_id for t in self.visible() + self.objects_in_view() if t.known and t.concept_id]
        self.lex.observe(text, in_view)
        self.ws.post(Signal(1.0, "speech", "", {"text": text}, "ears"))

    def tick(self) -> Signal | None:
        if self.sleeping:
            return None
        while not self._out.empty():
            kind, value, ctx = self._out.get()
            if kind == "act":
                self._do(value, ctx)
            elif kind == "say":
                self._voice(value)
            else:
                self._done(value, ctx)
        self.drives.tick(len(self.tracker.tracks))
        now = time.time()
        if self.drives.sleepy and now - self._last_tired > self.cfg.mind.tired_every_s:
            self._last_tired = now
            self.ws.post(Signal(0.6, "notice", "tired", {"kind": "tired"}, "drives"))
        self._initiative(now)
        return self.ws.step()

    def _initiative(self, now: float) -> None:
        """Speak up on its own: about someone it keeps seeing, or when it has been quiet for a while."""
        if self.thinking or not self.tracker.tracks:
            return
        m, d = self.cfg.mind, self.drives
        if d.curiosity > 0.6 and now - self._last_ask > m.ask_cooldown_s:
            for t in self.visible():
                c = self.store.concept(t.concept_id) if t.concept_id else None
                if c and c["kind"] == "stranger":
                    self._last_ask = now
                    self.ws.post(Signal(0.5 + 0.4 * d.curiosity, "notice", f"unnamed:{c['id']}",
                                        {"kind": "unnamed", "seen": c["seen_count"]}, "drives"))
                    return
        quiet = now - self._last_exchange
        if quiet > m.quiet_s * (1.6 - d.curiosity) and now - self._last_muse > m.quiet_s:
            self._last_muse = now
            self.ws.post(Signal(0.3 + 0.5 * d.curiosity, "notice", f"quiet:{int(now)}",
                                {"kind": "quiet", "s": quiet}, "drives"))

    def _act(self, s: Signal) -> None:
        if s.kind == "speech":
            self._think(said=s.payload["text"])
        else:
            self._think(noticed=s.payload)

    def _think(self, said: str | None = None, noticed: dict | None = None) -> None:
        if self.cortex is None:
            return
        self.thinking += 1
        ctx = {"said": said, "noticed": noticed}
        self.cortex.think(self.senses(), self.history(answering=said is not None),
                          lambda kind, value: self._out.put((kind, value, ctx)), said=said, noticed=noticed)

    def _voice(self, text: str) -> None:
        self.thought = text
        self._last_exchange = time.time()
        if self._say_rate:
            m, d = self.cfg.mind, self.drives
            self._say(text, m.tired_rate if d.energy < 0.25 else m.eager_rate if d.curiosity > 0.75 else 0.0)
        else:
            self._say(text)

    def _do(self, actions: list, ctx: dict) -> None:
        ctx["failed"], ctx["acted"] = [], bool(actions)
        if ctx["noticed"] is not None:            # noticing something never teaches; only people do
            actions = [a for a in actions if a.name == "go_to_sleep"]
        for a in actions:
            if a.name == "go_to_sleep":
                ctx["sleep"] = True
                continue
            if a.name == "learn_my_name" and self._is_someone_else(a.args["name"]):
                a.name = "learn_other_person"
            why = getattr(self, f"_{a.name}")(**a.args)
            if why:
                ctx["failed"].append(why)

    def _done(self, thought, ctx: dict) -> None:
        self.thinking -= 1
        if thought is None:
            return
        if thought.reply:
            self.store.add_episode("say", text=thought.reply, salience=0.3)
            if not ctx.get("acted"):
                self._note_claim(thought.reply)
        if ctx.get("sleep"):
            self.sleep()
        if ctx.get("failed") and (ctx["noticed"] or {}).get("kind") != "failed":
            self._think(noticed={"kind": "failed", "why": ctx["failed"][0]})

    def _note_claim(self, reply: str) -> None:
        """If the reply names something it currently recognises, a following yes or no is about that."""
        low = reply.lower()
        for g in self._view["guesses"]:
            if g.word and re.search(r"\b" + re.escape(g.word.lower()) + r"\b", low):
                self.last = g
                return

    # actions the cortex can take; each returns a reason when it could not be done

    def _pick(self, kind: str, which: str | None, prefer_unknown: bool):
        """The track an action is about: the one the cortex pointed at, else the most likely one."""
        tracks = self._view["people" if kind == "face" else "things"]
        alive = self.tracker.tracks if kind == "face" else self.things.tracks
        if which and which.strip().isdigit():
            i = int(which) - 1
            if 0 <= i < len(tracks) and tracks[i].id in alive:
                return tracks[i]
        live = self.visible() if kind == "face" else self.objects_in_view()
        if prefer_unknown:
            unknown = [t for t in live if not t.known or (
                t.concept_id and self.store.concept(t.concept_id)["kind"] == "stranger")]
            live = unknown or live
        if kind == "object" and self.frame is not None:
            w = self.frame.shape[1]
            # being shown something: big and near the middle of the view
            return max(live, key=lambda t: t.area * (1.5 - abs((t.box[0] + t.box[2] / 2) / w - 0.5)), default=None)
        return live[0] if live else None

    def _learn_my_name(self, name: str) -> str | None:
        t = self.speaker()
        if t is None or not t.embeddings:
            return "no_face"
        self._teach(t, "face", name)

    def _learn_other_person(self, name: str, which: str | None = None) -> str | None:
        others = self.visible()[1:]
        if others and not which:
            t = next((x for x in others if not x.known), others[0])
        else:
            t = self._pick("face", which, prefer_unknown=True)
        if t is None or not t.embeddings:
            return "no_face"
        self._teach(t, "face", name)

    def _learn_object(self, name: str, which: str | None = None) -> str | None:
        if self.store.find_concept(name, "person"):
            return self._learn_other_person(name, which)
        t = self._pick("object", which, prefer_unknown=False)
        if t is None or not t.embeddings:
            return "no_vision"
        self._teach(t, "object", name.lower())

    def _teach(self, track, kind: str, name: str) -> None:
        name = name.strip().title() if kind == "face" else name.strip()
        mem = self.faces if kind == "face" else self.objects
        ckind = "person" if kind == "face" else "object"
        existing = self.store.find_concept(name, ckind)
        if existing:
            cid = existing["id"]
        elif track.concept_id and (c := self.store.concept(track.concept_id)) and c["kind"] == "stranger":
            cid = c["id"]                                    # naming a stranger found during sleep
            self.store.name_concept(cid, name, kind="person", role="user")
        else:
            cid = self.store.new_concept(ckind, name)
        vecs = list(track.embeddings)
        ep = []
        for v in vecs:
            mem.learn(v, cid)
            ep.append(self.store.add_episode("learn", modality=kind, concept_id=cid, vec=v, text=f"name:{name}",
                                             salience=1.0, exemplar=True, source="teacher"))
        track.concept_id, track.known, track.announced = cid, True, True
        track.votes.clear()
        track.votes[cid] += 5
        self.pending = Pending(kind, cid, track.id, time.time() + (6.0 if kind == "face" else 4.0))
        self.lex.teach(name, cid)
        self.last = Assertion(kind, cid, name, vecs, ep)
        self.drives.novelty(0.4 if kind == "face" else 0.3)
        if kind == "face" and self.ethics.admin_id is None:
            self.ethics.set_admin(cid)                     # the first person it learns made it

    def _continue_teaching(self, now: float) -> None:
        p = self.pending
        if not p:
            return
        if now > p.until:
            self.pending = None
            return
        t = (self.tracker if p.kind == "face" else self.things).tracks.get(p.track_id)
        if t and t.embeddings and p.absorbed < (15 if p.kind == "face" else 10):
            v = t.embeddings[-1]
            (self.faces if p.kind == "face" else self.objects).learn(v, p.concept_id)
            self.store.add_episode("learn", modality=p.kind, concept_id=p.concept_id, vec=v,
                                   salience=0.9, exemplar=True, source="teacher")
            p.absorbed += 1

    def _remember_fact(self, subject: str, fact: str) -> None:
        s = subject.strip()
        if s.lower() in SELF_WORDS:
            t = self.speaker()
            s = (self.name_of(t.concept_id) if t and t.known else None) or "someone I met"
        elif s.lower() in YOU_WORDS:
            s = self.cfg.name
        self.store.x("INSERT OR IGNORE INTO facts(subject,relation,object,ts) VALUES(?,?,?,?)",
                     (s, "", fact.strip(), time.time()))
        self.store.add_episode("learn", text=f"fact:{s} {fact}", salience=0.9)
        self.drives.novelty(0.2)

    def _confirm_last(self) -> None:
        a = self.last
        if a and a.concept_id is not None:
            mem = self.faces if a.kind == "face" else self.objects
            for v in a.vecs[-3:]:
                mem.learn(v, a.concept_id)
                self.store.add_episode("learn", modality=a.kind, concept_id=a.concept_id, vec=v,
                                       text="feedback:+", salience=0.8, exemplar=True, source="teacher")
            if a.word:
                self.lex.reinforce(a.word, a.concept_id, 1.0)
        self.drives.interacted(0.3)

    def _correct_last(self, name: str, which: str | None = None) -> str | None:
        a = self.last
        if a and a.concept_id is not None and a.word:
            self.lex.reinforce(a.word, a.concept_id, -2.0)
        if a and a.episode_ids:
            # Retract the wrong exemplars. Their effect on prototypes goes away at the next sleep.
            self.store.many("UPDATE episodes SET exemplar=0, concept_id=NULL WHERE id=?",
                            [(i,) for i in a.episode_ids])
        if a and a.kind == "object":
            return self._learn_object(name, which)
        t = next((x for x in self.visible() if a and x.concept_id == a.concept_id), None) or self.speaker()
        if t is None or not t.embeddings:
            return "no_face"
        self._teach(t, "face", name)

    def _forget_person(self, name: str) -> str | None:
        t = self.speaker()
        if not self.ethics.may_forget(t.concept_id if t else None):
            return "not_allowed"
        c = self.store.find_concept(name)
        if c:
            self.ethics.forget(c["id"])
            self.faces.reload()
            self.objects.reload()

    def _is_someone_else(self, name: str) -> bool:
        # A known speaker who "gives their name" as something new while another face
        # is in view is almost always introducing that other person.
        v = self.visible()
        if len(v) < 2 or not v[0].known or not v[0].concept_id:
            return False
        own = self.store.concept(v[0].concept_id)["name"]
        return bool(own) and own.lower() != name.lower()

    # what the cortex is told

    def senses(self) -> dict:
        now = time.time()
        admin = self.ethics.admin_id
        fw = self.frame.shape[1] if self.frame is not None else 1
        people, things, guesses, wonder = [], [], [], []
        view_people, view_things = self.visible(), self.objects_in_view()
        for t in view_people:
            c = self.store.concept(t.concept_id) if t.known and t.concept_id else None
            name = c["name"] if c else None
            people.append({"name": name, "maker": bool(c) and c["id"] == admin,
                           "seen": c["seen_count"] if c else 0, "where": where(t.box, fw)})
            if name:
                guesses.append(Assertion("face", c["id"], name, list(t.embeddings)))
                if self._facts_about(name) < 2:
                    wonder.append({"kind": "about", "name": name})
            else:
                wonder.append({"kind": "who", "seen": c["seen_count"] if c else 0})
        finds = getattr(self.object_engine, "finds_objects", False)
        for t in view_things:
            name = self.name_of(t.concept_id) if t.known else None
            if name:
                things.append({"name": name, "sure": t.sim > 0.05, "where": where(t.box, fw)})
                guesses.append(Assertion("object", t.concept_id, name, list(t.embeddings)[-3:]))
            elif finds:
                things.append({"name": None, "where": where(t.box, fw), "label": getattr(t, "label", None)})
        if finds and any(x["name"] is None for x in things):
            wonder.append({"kind": "object"})
        for a, b in self.store.get_self("questions", [])[:1]:
            na, nb = self.name_of(a), self.name_of(b)
            if na and nb:
                wonder.append({"kind": "confused", "a": na, "b": nb})
        # Keep the tracks in the same order as told, so "which": "2" points at the right one.
        self._view = {"people": view_people, "things": [t for t in view_things if t.known or finds],
                      "guesses": guesses}
        return {
            "name": self.cfg.name,
            "stage": STAGES[self.stage],
            "age_s": now - float(self.store.get_self("born")),
            "people": people,
            "things": things,
            "people_known": [r["name"] for r in self.store.q(
                "SELECT name FROM concepts WHERE kind='person' AND name IS NOT NULL ORDER BY last_seen DESC LIMIT 12")],
            "things_known": [r["name"] for r in self.store.q(
                "SELECT name FROM concepts WHERE kind='object' AND name IS NOT NULL ORDER BY last_seen DESC LIMIT 12")],
            "facts": self._facts([p["name"] for p in people if p["name"]]),
            "wonder": wonder[:3],
            "feel": self.drives.as_dict(),
            "quiet_s": now - self._last_exchange,
        }

    def _facts_about(self, name: str) -> int:
        return self.store.q("SELECT COUNT(*) c FROM facts WHERE lower(subject)=?", (name.lower(),))[0]["c"]

    def _facts(self, names: list[str]) -> list[tuple[str, str, str]]:
        """Facts about whoever is in view first, then the most recent ones."""
        rows = self.store.q("SELECT subject, relation, object FROM facts ORDER BY ts DESC LIMIT 200")
        here = {n.lower() for n in names}
        rows = sorted(rows, key=lambda r: r["subject"].lower() not in here)
        return [(r["subject"], r["relation"], r["object"]) for r in rows[:12]]

    def history(self, turns: int = 6, answering: bool = True) -> list[tuple[str, str]]:
        rows = self.store.q("SELECT kind, text FROM episodes WHERE kind IN ('hear','say') AND text IS NOT NULL "
                            "ORDER BY ts DESC, id DESC LIMIT ?", (turns + answering,))
        rows = list(reversed(rows))
        if answering:
            rows = rows[:-1]                   # the newest one is the utterance being answered
        return [("user" if r["kind"] == "hear" else "assistant", r["text"]) for r in rows]

    def sleep(self) -> SleepReport:
        self.sleeping = True
        try:
            rep = self.sleeper.sleep()
            self.objects.reload()
        finally:
            self.sleeping = False
        self.drives.slept()
        self._think(noticed={"kind": "woke", "replayed": rep.concepts_replayed, "found": rep.strangers_found})
        return rep
