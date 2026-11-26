"""The Mind: one perception-cognition-action cycle, I/O-free so it is testable.

    frame ──► FaceEngine ──► Tracker ──► PrototypeMemory.recognize ──┐
    text  ──► language pack.parse ─────────────────────────────────────┤ signals
    drives (curiosity / social / energy) ─────────────────────────────┘
                                   │
                         Global Workspace (compete → broadcast)
                                   │
              act: speak · learn (one-shot + burst) · ask · sleep
                                   │
          Store (SQLite): episodes (hippocampus) · prototypes (cortex) · lexicon · facts

Teaching protocol (all optional, all natural language):
  "my name is Farhan"            → learn the closest face (few-shot burst)
  "this is my friend Sam"        → learn the other / unknown face
  "this is a cup"                → learn the object held in the centre
  "who am I?" "what is this?"    → test it
  "yes" / "no, it's a plate"     → reinforce / correct
  "coffee is a drink"            → learn a declarative fact
  "go to sleep"                  → consolidate now
(Phrasing lives in language packs; the core only sees intents.)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..config import Config, MemoryConfig
from ..language.pack import STAGE_NAMES, Intent, load_pack, stage_of
from ..memory.consolidation import Consolidator, SleepReport
from ..memory.lexicon import Lexicon
from ..memory.semantic import PrototypeMemory, l2n
from ..memory.store import Store
from .drives import Drives
from .ethics import Ethics
from .workspace import Signal, Workspace


def object_memory_cfg(base: MemoryConfig) -> MemoryConfig:
    """The training-free object descriptor lives on a different similarity
    scale than SFace, so it gets its own operating point."""
    from dataclasses import replace
    return replace(base, tau_known=0.80, tau_min=0.70, tau_max=0.92, spawn_tau=0.85, spread_k=2.5)


@dataclass
class Pending:
    """An open teaching window: keep absorbing samples for a few seconds."""
    kind: str                     # face | object
    concept_id: int
    track_id: int | None
    until: float
    absorbed: int = 0


@dataclass
class Assertion:
    """The last thing the Machine claimed: target of teacher feedback."""
    kind: str
    concept_id: int | None
    word: str | None = None
    vecs: list[np.ndarray] = field(default_factory=list)
    episode_ids: list[int] = field(default_factory=list)


class Mind:
    def __init__(self, cfg: Config, store: Store, say: Callable[[str], None],
                 face_engine=None, object_encoder=None, llm=None, self_training: bool = True):
        self.cfg, self.store, self._say = cfg, store, say
        self.faces = PrototypeMemory(store, cfg.memory, "face")
        self.objects = PrototypeMemory(store, object_memory_cfg(cfg.memory), "object")
        self.lang = load_pack(cfg.mind.language)
        self.lex = Lexicon(store, cfg.memory, self.lang.content_words)
        self.sleeper = Consolidator(store, cfg.memory, self.faces, self.lex)
        self.ethics = Ethics(store)
        self.drives = Drives(cfg.mind)
        self.ws = Workspace(cfg.mind.habituation_s)
        self.ws.subscribe(self._act)
        self.face_engine, self.obj_enc, self.llm = face_engine, object_encoder, llm
        self.self_training = self_training
        from ..perception.vision import Tracker
        self.tracker = Tracker(cfg.vision)
        self.frame: np.ndarray | None = None
        self.pending: Pending | None = None
        self.awaiting: tuple[str, float] | None = None      # ("name", ts) after asking "who are you?"
        self.last: Assertion | None = None
        self._last_ask = 0.0
        self._last_store: dict[int, float] = {}
        self._last_self_train: dict[int, float] = {}
        self._frame_i = 0
        self.thought = "..."
        self.sleeping = False
        self.metrics = {"frames": 0, "embeds": 0, "detect_ms": 0.0, "embed_ms": 0.0}

    # ================================================================ helpers
    @property
    def stage(self) -> int:
        return stage_of(len(self.faces.concepts()) + len(self.objects.concepts()), self.lex.vocabulary_size())

    def speak(self, key: str, **kw) -> None:
        text = self.lang.say(key, self.stage, **kw)
        self.thought = text
        self.store.add_episode("say", text=text, salience=0.3)
        self._say(text)

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

    def speaker(self):
        """Who is talking: the largest (closest) face in view."""
        v = self.visible()
        return v[0] if v else None

    # ============================================================= perception
    def see(self, frame: np.ndarray, now: float | None = None) -> None:
        now = now or time.time()
        self.frame = frame
        self._frame_i += 1
        self.metrics["frames"] += 1
        if self.face_engine is None:
            return
        t0 = time.perf_counter()
        rows = self.face_engine.detect(frame)
        self.metrics["detect_ms"] = (time.perf_counter() - t0) * 1e3
        active, gone = self.tracker.update(rows, now)
        for t in active:
            if t.frames == 1 or self._frame_i % self.cfg.vision.embed_every_n == 0 or (
                    self.pending and self.pending.track_id == t.id):
                t1 = time.perf_counter()
                e = self.face_engine.embed(frame, t.row)
                self.metrics["embed_ms"] = (time.perf_counter() - t1) * 1e3
                self.metrics["embeds"] += 1
                t.embeddings.append(e)
                self._recognize_track(t, e, now)
        for t in gone:
            self._remember_track(t, now, force=True)
        self._continue_teaching(now)

    def _recognize_track(self, t, e: np.ndarray, now: float) -> None:
        r = self.faces.recognize(e)
        t.votes[r.concept_id if r.known else -1] += 1
        cid, n = t.votes.most_common(1)[0]
        t.concept_id = None if cid == -1 else cid
        t.known, t.sim = cid != -1, r.sim
        decided = sum(t.votes.values()) >= 3
        if decided and not t.announced:
            t.announced = True
            if t.known:
                self.store.touch_concept(t.concept_id)
                admin = self.ethics.is_admin(t.concept_id)
                self.ws.post(Signal(0.55 + 0.4 * self.drives.social, "greet", f"greet:{t.concept_id}",
                                    {"cid": t.concept_id, "admin": admin, "track": t.id}, "vision"))
            else:
                self.ws.post(Signal(0.45 + 0.5 * self.drives.curiosity, "ask_name", f"ask:{t.id}",
                                    {"track": t.id}, "vision", ttl=5.0))
        # Self-supervised growth: absorb very confident sightings (rate-limited)
        if (self.self_training and t.known and r.known and r.concept_id == t.concept_id
                and r.sim >= r.threshold + 0.15 and r.margin >= 0.15
                and now - self._last_self_train.get(t.id, 0) > 10):
            self._last_self_train[t.id] = now
            self.faces.learn(e, t.concept_id)
            self.store.add_episode("see", modality="face", concept_id=t.concept_id, vec=e,
                                   salience=0.4, exemplar=True)
        self._remember_track(t, now)

    def _remember_track(self, t, now: float, force: bool = False) -> None:
        """Hippocampal trace of who was seen; unknown faces feed stranger discovery."""
        if not force and now - self._last_store.get(t.id, 0) < 5:
            return
        m = t.mean_embedding()
        if m is None:
            return
        self._last_store[t.id] = now
        self.store.add_episode("see", modality="face", concept_id=t.concept_id if t.known else None,
                               vec=m, salience=0.6 if t.known else 0.5)

    # ================================================================ hearing
    def hear(self, text: str) -> None:
        self.store.add_episode("hear", text=text, salience=0.6)
        self.drives.interacted()
        self.ws.post(Signal(1.0, "speech", "", {"text": text}, "ears"))

    # ================================================================= tick
    def tick(self) -> Signal | None:
        if self.sleeping:
            return None
        self.drives.tick(len(self.tracker.tracks))
        now = time.time()
        if self.drives.sleepy:
            self.ws.post(Signal(0.6, "sleepy", "sleepy", {}, "drives"))
        # curiosity about recurring strangers discovered during sleep
        if self.drives.curiosity > 0.6 and now - self._last_ask > self.cfg.mind.ask_cooldown_s:
            for t in self.visible():
                c = self.store.concept(t.concept_id) if t.concept_id else None
                if c and c["kind"] == "stranger":
                    self.ws.post(Signal(0.5 + 0.4 * self.drives.curiosity, "ask_stranger", f"stranger:{c['id']}",
                                        {"cid": c["id"], "k": c["seen_count"], "track": t.id}, "drives"))
                    break
        return self.ws.step()

    # ================================================================== act
    def _act(self, s: Signal) -> None:
        now = time.time()
        if s.kind == "greet":
            n = self.name_of(s.payload["cid"])
            if n:
                self.speak("greet_admin" if s.payload["admin"] else "greet_known", n=n)
        elif s.kind == "ask_name":
            if now - self._last_ask > self.cfg.mind.ask_cooldown_s / 3:
                self._last_ask = now
                self.awaiting = ("name", now)
                self.speak("ask_name")
        elif s.kind == "ask_stranger":
            self._last_ask = now
            self.awaiting = ("other", now)
            self.speak("ask_stranger", k=s.payload["k"])
        elif s.kind == "sleepy":
            self.speak("sleepy")
        elif s.kind == "speech":
            self._understand(s.payload["text"])

    # ======================================================== understanding
    def _understand(self, text: str) -> None:
        it = self.lang.parse(text)
        # a bare answer to "who are you?" is a name
        if it.kind == "chat" and self.awaiting and time.time() - self.awaiting[1] < 15 \
                and 1 <= len(it.text.split()) <= 2:
            it = Intent("name_me" if self.awaiting[0] == "name" else "name_other", it.text.strip("?"), None, it.text)
        self.awaiting = None
        h = getattr(self, f"_on_{it.kind}", None)
        (h or self._on_chat)(it)

    # -- teaching faces --------------------------------------------------
    def _teach_face(self, track, name: str, first_person: bool) -> None:
        name = name.strip().title()
        existing = self.store.find_concept(name, "person")
        if existing:
            cid = existing["id"]
        elif track.concept_id and (c := self.store.concept(track.concept_id)) and c["kind"] == "stranger":
            cid = c["id"]                                    # naming a stranger discovered in sleep
            self.store.name_concept(cid, name, kind="person", role="user")
        else:
            cid = self.store.new_concept("person", name)
        vecs = list(track.embeddings)
        ep = []
        for v in vecs:
            self.faces.learn(v, cid)
            ep.append(self.store.add_episode("learn", modality="face", concept_id=cid, vec=v,
                                             text=f"name:{name}", salience=1.0, exemplar=True))
        track.concept_id, track.known, track.announced = cid, True, True
        track.votes.clear()
        track.votes[cid] += 5
        self.pending = Pending("face", cid, track.id, time.time() + 6.0)
        self.lex.teach(name, cid)
        self.last = Assertion("face", cid, name, vecs, ep)
        self.drives.novelty(0.4)
        if self.ethics.admin_id is None:
            self.ethics.set_admin(cid)
            self.speak("first_admin", n=name)
        else:
            self.speak("learned_person", n=name)

    def _on_name_me(self, it: Intent) -> None:
        t = self.speaker()
        if t is None or not t.embeddings:
            return self.speak("no_face")
        self._teach_face(t, it.arg, True)

    def _on_name_other(self, it: Intent) -> None:
        v = self.visible()
        if not v:
            return self.speak("no_face")
        unknown = [t for t in v if not t.known or (t.concept_id and self.store.concept(t.concept_id)["kind"] == "stranger")]
        target = unknown[0] if unknown else (v[1] if len(v) > 1 else v[0])
        if target.embeddings:
            self._teach_face(target, it.arg, False)

    def _continue_teaching(self, now: float) -> None:
        p = self.pending
        if not p:
            return
        if now > p.until:
            self.pending = None
            return
        if p.kind == "face":
            t = self.tracker.tracks.get(p.track_id)
            if t and t.embeddings and p.absorbed < 15:
                v = t.embeddings[-1]
                self.faces.learn(v, p.concept_id)
                self.store.add_episode("learn", modality="face", concept_id=p.concept_id, vec=v,
                                       salience=0.9, exemplar=True)
                p.absorbed += 1
        elif p.kind == "object" and self.frame is not None and self.obj_enc and self._frame_i % 2 == 0 \
                and p.absorbed < 8:
            v = self.obj_enc.embed(self.obj_enc.center_crop(self.frame))
            self.objects.learn(v, p.concept_id)
            self.store.add_episode("learn", modality="object", concept_id=p.concept_id, vec=v,
                                   salience=0.9, exemplar=True)
            p.absorbed += 1

    # -- testing faces ---------------------------------------------------
    def _on_who_am_i(self, it: Intent) -> None:
        t = self.speaker()
        if t is None:
            return self.speak("no_face")
        e = t.mean_embedding()
        r = self.faces.recognize(e) if e is not None else None
        if r and r.known:
            n = self.name_of(r.concept_id)
            self.last = Assertion("face", r.concept_id, n, list(t.embeddings))
            self.speak("you_are" if r.score > 0.08 else "you_maybe", n=n)
        else:
            self.awaiting = ("name", time.time())
            self.speak("dont_know_you")

    def _on_who_is(self, it: Intent) -> None:
        v = self.visible()
        if len(v) == 0:
            return self.speak("no_face")
        t = v[1] if len(v) > 1 else v[0]
        if t.known and (n := self.name_of(t.concept_id)):
            self.last = Assertion("face", t.concept_id, n, list(t.embeddings))
            self.speak("that_is", n=n)
        else:
            self.awaiting = ("other", time.time())
            self.speak("ask_name")

    # -- objects ---------------------------------------------------------
    def _object_vec(self) -> np.ndarray | None:
        if self.frame is None or self.obj_enc is None:
            return None
        return self.obj_enc.embed(self.obj_enc.center_crop(self.frame))

    def _on_name_object(self, it: Intent) -> None:
        word = it.arg.lower().strip()
        if self.store.find_concept(word, "person"):
            return self._on_name_other(it)
        v = self._object_vec()
        if v is None:
            return self.speak("no_vision")
        m = self.lex.meaning(word)
        cid = m[0] if m and m[1] > 0.5 and self.store.concept(m[0])["kind"] == "object" else None
        cid = cid or (self.store.find_concept(word, "object") or {"id": None})["id"] \
            or self.store.new_concept("object", word)
        self.objects.learn(v, cid)
        ep = self.store.add_episode("learn", modality="object", concept_id=cid, vec=v, text=f"word:{word}",
                                    salience=1.0, exemplar=True)
        self.lex.teach(word, cid)
        self.pending = Pending("object", cid, None, time.time() + 3.0)
        self.last = Assertion("object", cid, word, [v], [ep])
        self.drives.novelty(0.3)
        self.speak("learned_obj", n=word)

    def _on_what_is(self, it: Intent) -> None:
        v = self._object_vec()
        if v is None:
            return self.speak("obj_unknown")
        r = self.objects.recognize(v)
        if r.concept_id is not None and r.known:
            n = self.name_of(r.concept_id)
            self.last = Assertion("object", r.concept_id, n, [v])
            self.speak("obj_is" if r.score > 0.03 else "obj_maybe", n=n)
        else:
            self.awaiting = ("object", time.time())
            self.last = Assertion("object", None, None, [v])
            self.speak("obj_unknown")

    # -- feedback ---------------------------------------------------------
    def _on_positive(self, it: Intent) -> None:
        a = self.last
        if a and a.concept_id is not None:
            mem = self.faces if a.kind == "face" else self.objects
            for v in a.vecs[-3:]:
                mem.learn(v, a.concept_id)
                self.store.add_episode("learn", modality=a.kind, concept_id=a.concept_id, vec=v,
                                       text="feedback:+", salience=0.8, exemplar=True)
            if a.word:
                self.lex.reinforce(a.word, a.concept_id, 1.0)
        self.drives.interacted(0.3)
        self.speak("thanks")

    def _on_negative(self, it: Intent) -> None:
        a = self.last
        if a and a.concept_id is not None and a.word:
            self.lex.reinforce(a.word, a.concept_id, -2.0)
        if a and a.episode_ids:      # retract the wrong exemplars; sleep re-estimation removes their influence
            self.store.many("UPDATE episodes SET exemplar=0, concept_id=NULL WHERE id=?", [(i,) for i in a.episode_ids])
        self.speak("sorry")

    def _on_correct(self, it: Intent) -> None:
        self._on_negative(it)
        a, word = self.last, it.arg
        if a and a.kind == "object":
            self._on_name_object(Intent("name_object", word))
        else:
            t = self.speaker() if a is None else next((x for x in self.visible() if x.concept_id == a.concept_id),
                                                      self.speaker())
            if t:
                self._teach_face(t, word, False)

    # -- self, memory, facts ---------------------------------------------
    def _on_who_are_you(self, it: Intent) -> None:
        self.speak("i_am", me=self.cfg.name)

    def _on_how_old(self, it: Intent) -> None:
        d = self.lang.duration(time.time() - float(self.store.get_self("born")))
        n = len(self.faces.concepts()) + len(self.objects.concepts())
        self.speak("age", d=d, c=n)

    def _on_how_are_you(self, it: Intent) -> None:
        self.speak("feel", cu=int(self.drives.curiosity * 100), en=int(self.drives.energy * 100))

    def _on_memory(self, it: Intent) -> None:
        rows = self.store.q("SELECT DISTINCT c.name FROM episodes e JOIN concepts c ON c.id=e.concept_id "
                            "WHERE c.name IS NOT NULL ORDER BY e.ts DESC LIMIT 6")
        names = [r["name"] for r in rows]
        if not names:
            return self.speak("memory_empty")
        self.speak("memory", m=", ".join(names))

    def _on_fact(self, it: Intent) -> None:
        s, o = it.arg.lower(), it.arg2.lower()
        self.store.x("INSERT OR IGNORE INTO facts(subject,relation,object,ts) VALUES(?,?,?,?)",
                     (s, "is", o, time.time()))
        self.store.add_episode("learn", text=f"fact:{s} is {o}", salience=0.9)
        self.drives.novelty(0.2)
        self.speak("fact_learned", s=s, o=o)

    def _on_ask_fact(self, it: Intent) -> None:
        x = it.arg.lower().strip()
        r = self.store.q("SELECT object FROM facts WHERE subject=? ORDER BY confidence DESC LIMIT 1", (x,))
        if r:
            return self.speak("fact_known", s=x, o=r[0]["object"])
        self.speak("unknown_word", n=x)

    def _on_forget(self, it: Intent) -> None:
        t = self.speaker()
        if not self.ethics.may_forget(t.concept_id if t else None):
            return self.speak("not_allowed")
        c = self.store.find_concept(it.arg)
        if c:
            self.ethics.forget(c["id"])
            self.faces.reload()
            self.objects.reload()
            for tr in self.tracker.tracks.values():
                if tr.concept_id == c["id"]:
                    tr.concept_id, tr.known = None, False
                    tr.votes.clear()
        self.speak("forgot", n=it.arg)

    def _on_sleep(self, it: Intent) -> None:
        self.sleep()

    def _on_bye(self, it: Intent) -> None:
        t = self.speaker()
        self.speak("bye", n=self.name_of(t.concept_id) if t and t.known else "")

    def _on_chat(self, it: Intent) -> None:
        # language exposure: cross-situational learning with whatever is in view
        in_view = [t.concept_id for t in self.visible() if t.known and t.concept_id]
        self.lex.observe(it.text, in_view)
        if self.llm is not None:
            try:
                reply = self.llm.reply(it.text, self.context_summary())
                if reply:
                    self.thought = reply
                    self.store.add_episode("say", text=reply, salience=0.3)
                    return self._say(reply)
            except Exception:
                pass
        words = self.lang.content_words(it.text)
        self.speak("confused", w=words[-1] if words else "")

    def context_summary(self) -> str:
        people = [self.name_of(t.concept_id) for t in self.visible() if t.known]
        facts = self.store.q("SELECT subject, object FROM facts ORDER BY ts DESC LIMIT 8")
        return (f"I am {self.cfg.name}, an AI who learns from my teacher. "
                f"I see: {', '.join(p for p in people if p) or 'nobody I know'}. "
                f"I know: {'; '.join(f'{r[0]} is {r[1]}' for r in facts) or 'little yet'}. "
                f"Stage: {STAGE_NAMES[self.stage]}.")

    # ================================================================= sleep
    def sleep(self) -> SleepReport:
        self.speak("sleep_start")
        self.sleeping = True
        try:
            rep = self.sleeper.sleep()
            self.objects.reload()
        finally:
            self.sleeping = False
        self.drives.slept()
        self.speak("sleep_done", r=rep.concepts_replayed, s=rep.strangers_found)
        qs = self.store.get_self("questions", [])
        if qs:
            a, b = self.name_of(qs[0][0]) or "?", self.name_of(qs[0][1]) or "?"
            self.speak("confusion_q", a=a, b=b)
        return rep
