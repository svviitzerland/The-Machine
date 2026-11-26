"""Faces are simulated as clusters on the unit sphere, so the tests are deterministic and need no camera."""
from __future__ import annotations

import time

import numpy as np
import pytest

from the_machine.config import Config
from the_machine.core.mind import Mind
from the_machine.core.workspace import Signal, Workspace
from the_machine.language.pack import load_pack
from the_machine.memory.consolidation import Consolidator
from the_machine.memory.lexicon import Lexicon
from the_machine.memory.semantic import PrototypeMemory, l2n
from the_machine.memory.store import Store

D = 128
EN = load_pack("en")


def identity(rng, n_ids=5, spread=0.35):
    centers = l2n(rng.normal(size=(n_ids, D)))
    def sample(i, k=1):
        return l2n(centers[i] + spread * rng.normal(size=(k, D)) / np.sqrt(D) * 4)
    return centers, sample


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "m.db")


def test_prototype_memory_open_set(store):
    rng = np.random.default_rng(0)
    cfg = Config().memory
    mem = PrototypeMemory(store, cfg, "face")
    _, sample = identity(rng)
    cids = [store.new_concept("person", f"p{i}") for i in range(4)]
    for i, c in enumerate(cids):
        mem.learn_many(sample(i, 8), c)
    acc = np.mean([mem.recognize(x).concept_id == cids[i] and mem.recognize(x).known
                   for i in range(4) for x in sample(i, 10)])
    rej = np.mean([not mem.recognize(x).known for x in sample(4, 20)])   # never-seen identity
    assert acc > 0.9 and rej > 0.9


def test_no_forgetting_after_new_classes(store):
    rng = np.random.default_rng(1)
    mem = PrototypeMemory(store, Config().memory, "face")
    _, sample = identity(rng, 10)
    first = store.new_concept("person", "first")
    mem.learn_many(sample(0, 8), first)
    test = sample(0, 20)
    before = np.mean([mem.recognize(x).concept_id == first for x in test])
    for i in range(1, 10):
        mem.learn_many(sample(i, 8), store.new_concept("person", f"p{i}"))
    after = np.mean([mem.recognize(x).concept_id == first for x in test])
    assert after >= before - 0.05


def test_persistence(tmp_path):
    rng = np.random.default_rng(2)
    s = Store(tmp_path / "p.db")
    mem = PrototypeMemory(s, Config().memory, "face")
    _, sample = identity(rng)
    c = s.new_concept("person", "keep")
    mem.learn_many(sample(0, 5), c)
    s.close()
    s2 = Store(tmp_path / "p.db")
    mem2 = PrototypeMemory(s2, Config().memory, "face")
    assert mem2.recognize(sample(0, 1)[0]).concept_id == c


def test_lexicon_cross_situational(store):
    lex = Lexicon(store, Config().memory, EN.content_words)
    cup, ball, dog = (store.new_concept("object", None) for _ in range(3))
    rng = np.random.default_rng(3)
    scenes = [([cup, ball], "look at the cup"), ([cup, dog], "a cup here"), ([ball, dog], "nice ball"),
              ([ball, cup], "ball"), ([dog, cup], "the dog"), ([dog, ball], "dog")]
    for _ in range(5):
        for v, t in scenes:
            lex.observe(t, v)
    assert lex.meaning("cup")[0] == cup
    assert lex.meaning("ball")[0] == ball
    assert lex.meaning("dog")[0] == dog


def test_workspace_habituation():
    ws = Workspace(habituation_s=100)
    won = []
    ws.subscribe(lambda s: won.append(s.kind))
    ws.post(Signal(0.8, "greet", "greet:1"))
    ws.step()
    ws.post(Signal(0.8, "greet", "greet:1"))
    ws.post(Signal(0.5, "ask", "ask:2"))
    ws.step()
    assert won == ["greet", "ask"]


def test_language_pack_intents():
    cases = {"my name is Farhan": ("name_me", "Farhan"), "who am I?": ("who_am_i", None),
             "this is a cup": ("name_object", "cup"), "what is this?": ("what_is", None),
             "no, it's a plate": ("correct", "plate"), "coffee is a drink": ("fact", "coffee"),
             "go to sleep": ("sleep", None), "this is my friend Sam": ("name_other", "Sam"),
             "the weather today": ("chat", None)}
    for text, (kind, arg) in cases.items():
        it = EN.parse(text)
        assert it.kind == kind, (text, it)
        if arg:
            assert it.arg == arg, (text, it)


def test_sleep_discovers_stranger_and_replays(store):
    rng = np.random.default_rng(4)
    cfg = Config().memory
    mem = PrototypeMemory(store, cfg, "face")
    lex = Lexicon(store, cfg, EN.content_words)
    _, sample = identity(rng, spread=0.25)   # genuine-pair cosine ~0.5, like SFace
    known = store.new_concept("person", "known")
    for x in sample(0, 12):
        mem.learn(x, known)
        store.add_episode("learn", modality="face", concept_id=known, vec=x, exemplar=True)
    for x in sample(1, 10):            # a recurring, unnamed face
        store.add_episode("see", modality="face", vec=x)
    rep = Consolidator(store, cfg, mem, lex).sleep()
    assert rep.strangers_found == 1 and rep.concepts_replayed >= 1
    assert mem.recognize(sample(1, 1)[0]).known
class FakeFaces:
    """Stands in for YuNet+SFace: each 'frame' is a list of (box, embedding)."""

    def detect(self, frame):
        return np.array([np.r_[box, np.zeros(10), 0.9] for box, _ in frame], np.float32).reshape(-1, 15)

    def embed(self, frame, row):
        for box, e in frame:
            if np.allclose(row[:4], box):
                return e


def test_mind_learns_and_recognises_teacher(tmp_path):
    rng = np.random.default_rng(5)
    cfg = Config()
    cfg.memory.db_path = tmp_path / "mind.db"
    said: list[str] = []
    m = Mind(cfg, Store(cfg.memory.db_path), said.append, face_engine=FakeFaces())
    _, sample = identity(rng)
    box = np.array([100, 100, 120, 120], np.float32)
    for x in sample(0, 6):
        m.see([(box, x)])
        m.tick()
    m.hear("my name is Farhan")
    m.tick()
    assert any("Farhan" in s for s in said)
    assert m.ethics.admin_id is not None          # first teacher becomes admin
    for x in sample(0, 4):
        m.see([(box, x)])
    m.hear("who am I?")
    m.tick()
    assert "Farhan" in said[-1]
    # a new face is unknown
    m.tracker.tracks.clear()
    for x in sample(3, 5):
        m.see([(box, x)])
    m.hear("who am I?")
    m.tick()
    assert "Farhan" not in said[-1]
    rep = m.sleep()
    assert rep.prototypes_after >= 1


def test_sleep_demotes_wrong_self_labels(store):
    rng = np.random.default_rng(6)
    cfg = Config().memory
    mem = PrototypeMemory(store, cfg, "face")
    lex = Lexicon(store, cfg, EN.content_words)
    _, sample = identity(rng, spread=0.25)
    a = store.new_concept("person", "a")
    for x in sample(0, 10):
        mem.learn(x, a)
        store.add_episode("learn", modality="face", concept_id=a, vec=x, exemplar=True, source="teacher")
    for x in sample(0, 4):          # correct self-labels
        store.add_episode("see", modality="face", concept_id=a, vec=x, exemplar=True, source="self")
    for x in sample(2, 4):          # wrong self-labels: another person absorbed into "a"
        store.add_episode("see", modality="face", concept_id=a, vec=x, exemplar=True, source="self")
    rep = Consolidator(store, cfg, mem, lex).sleep()
    assert rep.self_demoted >= 3 and rep.self_verified >= 3


def test_persona_moods_change_rate_and_wording():
    from the_machine.language.persona import load_persona, mood_for
    p = load_persona("machine")
    assert mood_for("ask_name", 0.5, 1.0) == "curious"
    assert mood_for("greet_known", 0.5, 0.1) == "tired"
    texts = {p.style("Who are you?", "curious", stage=2)[0] for _ in range(50)}
    assert "Who are you?" in texts and len(texts) > 1
    assert p.style("Hi", "curious", 2)[1] < p.style("Hi", "tired", 2)[1]
    assert p.style("Farhan... Farhan!", "happy", 0)[0] == "Farhan... Farhan!"
