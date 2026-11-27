"""Faces are simulated as clusters on the unit sphere, so the tests are deterministic and need no camera."""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from the_machine.config import Config
from the_machine.core.mind import Mind
from the_machine.core.workspace import Signal, Workspace
from the_machine.memory.consolidation import Consolidator
from the_machine.memory.lexicon import Lexicon
from the_machine.memory.semantic import PrototypeMemory, l2n
from the_machine.memory.store import Store

D = 128


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
    lex = Lexicon(store, Config().memory)
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


def test_sleep_discovers_stranger_and_replays(store):
    rng = np.random.default_rng(4)
    cfg = Config().memory
    mem = PrototypeMemory(store, cfg, "face")
    lex = Lexicon(store, cfg)
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


def test_sleep_demotes_wrong_self_labels(store):
    rng = np.random.default_rng(6)
    cfg = Config().memory
    mem = PrototypeMemory(store, cfg, "face")
    lex = Lexicon(store, cfg)
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


class FakeFaces:
    """Stands in for YuNet+SFace: each 'frame' is a list of (box, embedding)."""

    def detect(self, frame):
        return np.array([np.r_[box, np.zeros(10), 0.9] for box, _ in frame["faces"]], np.float32).reshape(-1, 15)

    def embed(self, frame, row):
        for box, e in frame["faces"]:
            if np.allclose(row[:4], box):
                return e


class FakeThings:
    finds_objects = True

    def detect(self, frame):
        return np.array([np.r_[box, 0.9] for box, _ in frame["things"]], np.float32).reshape(-1, 5)

    def embed(self, frame, row):
        for box, e in frame["things"]:
            if np.allclose(row[:4], box):
                return e


class Frame(dict):
    """A fake camera frame: faces and things with their embeddings, and a size."""
    shape = (480, 640, 3)


class FakeLLM:
    """Stands in for the model. ``answer`` is a dict, or a function of the messages that returns one."""

    def __init__(self, answer):
        self.answer, self.seen = answer, []

    def chat(self, messages, schema, max_tokens):
        self.seen.append(messages)
        a = self.answer(messages) if callable(self.answer) else self.answer
        return json.dumps(a)


def settle(m, until=lambda: False):
    for _ in range(200):
        m.tick()
        if not m.thinking and until():
            return
        time.sleep(0.005)


def make_mind(tmp_path, llm, **kw):
    from the_machine.cortex.cortex import LLMCortex
    cfg = Config()
    cfg.memory.db_path = tmp_path / "mind.db"
    said: list[str] = []
    m = Mind(cfg, Store(cfg.memory.db_path), said.append, cortex=LLMCortex(llm) if llm else None, **kw)
    return m, said


FACE_BOX = np.array([280, 100, 120, 120], np.float32)


def test_mind_learns_a_name_from_natural_speech(tmp_path):
    rng = np.random.default_rng(5)
    _, sample = identity(rng)

    def brain(messages):
        heard = messages[-1]["content"]
        if "Farhan" in heard:
            return {"actions": [{"do": "learn_my_name", "name": "Farhan"}], "reply": "Farhan! Hi."}
        return {"actions": [], "reply": ""}

    llm = FakeLLM(brain)
    m, said = make_mind(tmp_path, llm, face_engine=FakeFaces())
    for x in sample(0, 6):
        m.see(Frame(faces=[(FACE_BOX, x)], things=[]))
    settle(m)
    m.hear("hey, it's me Farhan")
    settle(m, lambda: "Farhan! Hi." in said)
    assert m.ethics.admin_id is not None          # the first person it learns made it
    assert m.senses()["people"][0]["name"] == "Farhan"
    assert any("1 (talking to you): unknown" in msgs[0]["content"] for msgs in llm.seen)
    m.tracker.tracks.clear()                      # a new face is not Farhan
    for x in sample(3, 5):
        m.see(Frame(faces=[(FACE_BOX, x)], things=[]))
    assert m.senses()["people"][0]["name"] is None
    assert m.sleep().prototypes_after >= 1


def test_without_a_cortex_it_stays_quiet(tmp_path):
    m, said = make_mind(tmp_path, None)
    m.hear("my name is Farhan")
    settle(m)
    assert said == []


def test_a_failing_model_does_not_hang(tmp_path):
    class Broken:
        def chat(self, *a):
            raise RuntimeError("no model")

    m, said = make_mind(tmp_path, Broken())
    m.hear("hello")
    settle(m, lambda: True)
    assert m.thinking == 0 and said == []


def test_noticing_is_voiced_by_the_cortex_and_never_teaches(tmp_path):
    def brain(messages):
        assert messages[-1]["content"].startswith("(")
        return {"actions": [{"do": "learn_other_person", "name": "Farhan"}], "reply": "There you are, Farhan."}

    m, said = make_mind(tmp_path, FakeLLM(brain))
    m._think(noticed={"kind": "arrived", "name": "Farhan", "maker": True})
    settle(m, lambda: bool(said))
    assert said == ["There you are, Farhan."]
    assert m.store.find_concept("Farhan", "person") is None


def test_it_speaks_up_when_it_has_been_quiet(tmp_path):
    rng = np.random.default_rng(8)
    _, sample = identity(rng)
    llm = FakeLLM({"actions": [], "reply": "What are you working on?"})
    m, said = make_mind(tmp_path, llm, face_engine=FakeFaces())
    for x in sample(0, 2):
        m.see(Frame(faces=[(FACE_BOX, x)], things=[]))
    m.ws.habituation_s = 0
    m._last_exchange = m._last_muse = time.time() - 600
    settle(m, lambda: bool(said))
    assert said == ["What are you working on?"]
    assert any("quiet" in msgs[-1]["content"] for msgs in llm.seen)


def test_it_learns_the_thing_it_is_pointed_at_among_several(tmp_path):
    rng = np.random.default_rng(9)
    _, sample = identity(rng, n_ids=3, spread=0.05)
    boxes = [np.array([40 + 200 * i, 300, 120, 120], np.float32) for i in range(3)]

    def frame():
        return Frame(faces=[], things=[(boxes[i], sample(i, 1)[0]) for i in range(3)])

    llm = FakeLLM({"actions": [{"do": "learn_object", "name": "mug", "which": "2"}], "reply": "A mug."})
    m, said = make_mind(tmp_path, llm, object_engine=FakeThings())
    for _ in range(4):
        m.see(frame())
    assert len(m.senses()["things"]) == 3
    m.hear("the one in the middle is a mug")
    settle(m, lambda: bool(said))
    m.things.tracks.clear()
    m.pending = None
    for _ in range(4):
        m.see(frame())
    names = [t["name"] for t in m.senses()["things"]]
    assert names == [None, "mug", None]


def test_a_fact_about_me_is_stored_under_my_name(tmp_path):
    rng = np.random.default_rng(10)
    _, sample = identity(rng)
    answers = iter([{"actions": [{"do": "learn_my_name", "name": "Farhan"}], "reply": "Hi Farhan."},
                    {"actions": [{"do": "remember_fact", "subject": "me", "fact": "loves coffee"}],
                     "reply": "Coffee, got it."}])
    m, said = make_mind(tmp_path, FakeLLM(lambda _: next(answers)), face_engine=FakeFaces())
    for x in sample(0, 4):
        m.see(Frame(faces=[(FACE_BOX, x)], things=[]))
    m.hear("I'm Farhan")
    settle(m, lambda: len(said) >= 1)
    m.hear("I love coffee")
    settle(m, lambda: len(said) >= 2)
    assert ("Farhan", "", "loves coffee") in m.senses()["facts"]


def test_names_it_stores_must_have_been_said():
    from the_machine.cortex.cortex import parse
    raw = json.dumps({"actions": [{"do": "learn_my_name", "name": "I am you"},
                                  {"do": "learn_object", "name": "cup"}], "reply": "Hm."})
    assert parse(raw, "do you know who I am?").actions == []
    assert [a.name for a in parse(raw, "this is a cup").actions] == ["learn_object"]


def test_reply_streams_sentence_by_sentence():
    from the_machine.cortex.cortex import ReplyStream
    raw = json.dumps({"actions": [{"do": "go_to_sleep"}],
                      "reply": "Okay. I'll rest now! Say \"hi\" when\nyou're back?"})
    heads, sentences = [], []
    s = ReplyStream(heads.append, sentences.append)
    for ch in raw:
        s.feed(ch)
        if sentences:
            assert heads                          # actions always come before speech
    s.finish()
    assert json.loads(heads[0].rstrip().rstrip(",") + "}")["actions"][0]["do"] == "go_to_sleep"
    assert sentences == ["Okay.", "I'll rest now!", 'Say "hi" when you\'re back?']


def test_head_follows_the_person_it_sees():
    import socket
    from the_machine.body import Aim, Head
    from the_machine.config import BodyConfig

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    cfg = BodyConfig(gain=1.0, send_hz=1000, alive_deg=0)
    head = Head(f"127.0.0.1:{port}", cfg)
    conn, _ = srv.accept()
    for _ in range(100):
        if head.connected:
            break
        time.sleep(0.01)
    aim = Aim(cfg)
    pan, tilt = aim.update([(0, 200, 100, 100)], 640, 480, time.time())   # someone at the camera's left edge
    assert pan > 110 and abs(tilt - 90) < 15
    head.look(pan, tilt)
    conn.sendall(b"OK P115 T91\n")
    assert conn.recv(64).decode().startswith(f"P{round(pan)} T{round(tilt)}")
    for _ in range(100):
        if head.pan == 115:
            break
        time.sleep(0.01)
    assert head.pan == 115
    head.close()
    conn.close()
    srv.close()


def test_offline_guard_allows_only_this_computer_and_the_local_network():
    from the_machine.offline import _local
    assert _local(("127.0.0.1", 80)) and _local(("192.168.1.20", 7351)) and _local(("10.13.37.2", 7351))
    assert _local(("::1", 80)) and _local(("fe80::1%en0", 80, 0, 0))
    assert not _local(("104.20.23.154", 443))
    assert not _local(("fd00:aa:bb:2230::6814:179a", 443, 0, 0))   # a "private" IPv6 that leads outside
