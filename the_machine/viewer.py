"""A local web page to look inside the Machine: its memory as a map and its
neural network, token by token. It serves on 127.0.0.1 only, so nothing leaves
the computer. With a running Mind it is live; without one it reads the memory
file and loads the cortex on demand.
"""
from __future__ import annotations

import base64
import json
import math
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

PAGE = Path(__file__).parent / "web" / "index.html"


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def pack(act: np.ndarray, idx: np.ndarray) -> dict:
    """Activations as bytes (0-255, scaled per layer) plus each layer's scale, and neuron ids as uint16."""
    scale = act.max(-1, keepdims=True) + 1e-6
    return {"act": _b64(np.round(act / scale * 255).astype(np.uint8)), "scale": _b64(scale.astype(np.float32)),
            "idx": _b64(idx.astype(np.uint16))}


def memory_graph(store, admin_id, name: str, active: set[int]) -> dict:
    nodes = [{"id": "self", "kind": "self", "label": name, "size": 3.0, "info": {"what": "itself"}}]
    edges = []
    for c in store.q("SELECT * FROM concepts ORDER BY last_seen DESC LIMIT 120"):
        cid = c["id"]
        protos = store.q("SELECT COUNT(*) n FROM prototypes WHERE concept_id=?", (cid,))[0]["n"]
        shots = store.q("SELECT COUNT(*) n FROM episodes WHERE concept_id=? AND exemplar=1", (cid,))[0]["n"]
        kind = "maker" if cid == admin_id else c["kind"]
        info = {"what": {"maker": "the person who made it", "person": "a person", "stranger": "a face it keeps "
                         "seeing without a name", "object": "a thing"}.get(kind, c["kind"]),
                "seen": c["seen_count"], "prototypes": protos, "examples kept": shots,
                "first met": time.strftime("%d %b %Y %H:%M", time.localtime(c["created"])) if c["created"] else "-",
                "last seen": time.strftime("%d %b %Y %H:%M", time.localtime(c["last_seen"])) if c["last_seen"] else "-"}
        nodes.append({"id": f"c{cid}", "kind": kind, "label": c["name"] or f"#{cid}",
                      "size": 1.0 + math.log1p(c["seen_count"] or 0) / 2, "info": info})
        edges.append(["self", f"c{cid}", 1.0])
    by_name = {n["label"].lower(): n["id"] for n in nodes}
    for i, f in enumerate(store.q("SELECT * FROM facts ORDER BY ts DESC LIMIT 80")):
        text = " ".join(x for x in (f["relation"], f["object"]) if x)
        nodes.append({"id": f"f{i}", "kind": "fact", "label": text[:40], "size": 0.8,
                      "info": {"what": "something it was told", "about": f["subject"], "fact": text,
                               "when": time.strftime("%d %b %Y %H:%M", time.localtime(f["ts"]))}})
        edges.append([by_name.get(f["subject"].lower(), "self"), f"f{i}", 0.8])
    seen_words = set()
    for w in store.q("SELECT word, concept_id, weight, count FROM lexicon WHERE weight >= 1 "
                     "ORDER BY weight DESC LIMIT 120"):
        owner = f"c{w['concept_id']}"
        if not any(n["id"] == owner for n in nodes):
            continue
        nid = f"w{w['word']}"
        if nid not in seen_words:
            seen_words.add(nid)
            nodes.append({"id": nid, "kind": "word", "label": w["word"], "size": 0.6,
                          "info": {"what": "a word it heard"}})
        edges.append([owner, nid, min(1.0, w["weight"] / 10)])
        next(n for n in nodes if n["id"] == nid)["info"][f"means {owner}"] = f"weight {w['weight']:.1f}, heard {w['count']}x"
    return {"nodes": nodes, "edges": edges, "active": [f"c{a}" for a in active]}


class Feed:
    """What the running Machine shares with the page besides its Mind: the camera
    frame, its ears and voice, and a way to type to it."""

    def __init__(self, say=None, ears=None, speaking=None):
        self.frame = None
        self.fps = 0.0
        self.say, self.ears, self.speaking = say, ears, speaking


class Viewer:
    def __init__(self, cfg, mind=None, feed: Feed | None = None):
        self.cfg, self.mind, self.feed = cfg, mind, feed
        self._backend = None
        self._load = threading.Lock()
        if mind is None:
            from .core.mind import Mind
            from .memory.store import Store
            self.mind_offline = Mind(cfg, Store(cfg.memory.db_path), lambda *_: None)

    @property
    def live(self) -> bool:
        return self.mind is not None

    def _mind(self):
        return self.mind or self.mind_offline

    def backend(self):
        if self.mind is not None:
            return getattr(self.mind.cortex, "backend", None)
        with self._load:
            if self._backend is None:
                from .cortex.llm import LlamaBackend, find_model
                path = find_model(Path(self.cfg.models_dir), self.cfg.mind.llm_models)
                self._backend = LlamaBackend(path, gpu_layers=self.cfg.mind.llm_gpu_layers) if path else False
            return self._backend or None

    def memory(self) -> dict:
        m = self._mind()
        active = set()
        if self.live:
            try:
                active = {t.concept_id for t in list(m.tracker.tracks.values()) + list(m.things.tracks.values())
                          if t.known and t.concept_id}
            except RuntimeError:              # the tracks changed while we read them
                pass
        g = memory_graph(m.store, m.ethics.admin_id, self.cfg.name, active)
        g["live"] = self.live
        return g

    def brain(self) -> dict:
        from .cortex.llm import find_model
        b = getattr(self.mind.cortex, "backend", None) if self.live else self._backend
        path = None if b else find_model(Path(self.cfg.models_dir), self.cfg.mind.llm_models)
        tap = getattr(b, "tap", None) if b else None
        return {"live": self.live, "camera": bool(self.feed), "model": b.name if b else (path.stem if path else None),
                "layers": tap.layers if tap else None, "width": tap.width if tap else None,
                "loaded": bool(b), "watchable": tap is not None if b else None}

    def trace(self, text: str) -> dict:
        from .cortex.cortex import messages
        from .cortex.tap import trace
        b = self.backend()
        if b is None or b.tap is None:
            return {"error": "no cortex model to look into. Run: python -m the_machine download"}
        m = self._mind()
        view = m._view
        senses = m.senses()
        m._view = view                       # don't disturb what a running Mind points at
        r = trace(b, messages(senses, m.history(answering=False), said=text), text, max_new=96)
        act = np.array(r.pop("act"), np.float32)
        idx = np.array(r.pop("idx"))
        r.update(pack(act, idx))
        return r

    def live_frame(self) -> dict:
        b = self.backend() if self.live else None
        tap = getattr(b, "tap", None)
        if tap is None:
            return {"tokens": 0}
        from .cortex.tap import pool
        tap.on = True
        self._watched = time.time()
        frame, n = tap.read()
        act, idx = pool(frame, 64)
        out = pack(act[None], idx[None])
        out.update({"tokens": n, "layers": tap.layers, "width": tap.width, "groups": 64,
                    "thinking": bool(self.mind.thinking), "text": b.recent[-160:]})
        return out

    def state(self) -> dict:
        """Everything the live view draws: who and what is in view, and what it is doing."""
        m, f = self.mind, self.feed
        frame = f.frame if f else None
        h, w = frame.shape[:2] if frame is not None else (480, 640)
        admin = m.ethics.admin_id
        pending = m.pending
        faces, things = [], []
        try:
            for t in list(m.tracker.tracks.values()):
                c = m.store.concept(t.concept_id) if t.known and t.concept_id else None
                kind = "unknown" if c is None else "maker" if c["id"] == admin else c["kind"]
                faces.append({"id": t.id, "box": [float(v) for v in t.box], "name": c["name"] if c else None,
                              "kind": kind, "sim": round(float(t.sim), 2),
                              "teaching": bool(pending and pending.kind == "face" and pending.track_id == t.id)})
            finds = getattr(m.object_engine, "finds_objects", False)
            for t in list(m.things.tracks.values()):
                name = m.name_of(t.concept_id) if t.known else None
                teaching = bool(pending and pending.kind == "object" and pending.track_id == t.id)
                if name or finds or teaching:
                    things.append({"id": t.id, "box": [float(v) for v in t.box], "name": name, "teaching": teaching})
        except RuntimeError:                  # tracks changed while we read them; the next poll catches up
            pass
        ears = f.ears if f else None
        speaking = bool(f and f.speaking and f.speaking.is_set())
        status = ("thinking" if m.thinking else "speaking" if speaking else
                  "hearing" if getattr(ears, "hearing", False) else
                  "listening" if hasattr(ears, "level") else "typing")
        return {"size": [w, h], "faces": faces, "things": things, "status": status,
                "level": round(min(1.0, getattr(ears, "level", 0.0) / 3), 2),
                "drives": m.drives.as_dict(), "fps": round(f.fps, 1) if f else 0}

    def chat(self, n: int = 40) -> list[dict]:
        rows = self._mind().store.q("SELECT id, ts, kind, text FROM episodes WHERE kind IN ('hear','say') "
                                    "AND text IS NOT NULL ORDER BY ts DESC, id DESC LIMIT ?", (n,))
        return [{"id": r["id"], "who": "you" if r["kind"] == "hear" else "it", "text": r["text"], "ts": r["ts"]}
                for r in reversed(rows)]

    def jpeg(self) -> bytes | None:
        import cv2
        frame = self.feed.frame if self.feed else None
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        return buf.tobytes() if ok else None

    def idle(self) -> None:
        """Stop reading the network once nobody has asked for a while."""
        while True:
            time.sleep(2)
            b = getattr(getattr(self.mind, "cortex", None), "backend", None)
            if b is not None and b.tap is not None and time.time() - getattr(self, "_watched", 0) > 3:
                b.tap.on = False


def handler(viewer: Viewer):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, kind: str = "application/json", code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(json.dumps(obj).encode(), code=code)

        def do_GET(self):
            self.path = self.path.split("?")[0]
            try:
                if self.path in ("/", "/index.html"):
                    return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                if self.path == "/api/memory":
                    return self._json(viewer.memory())
                if self.path == "/api/brain":
                    return self._json(viewer.brain())
                if self.path == "/api/live":
                    return self._json(viewer.live_frame())
                if self.path == "/api/state" and viewer.live:
                    return self._json(viewer.state())
                if self.path == "/api/chat":
                    return self._json(viewer.chat())
                if self.path == "/api/video" and viewer.feed:
                    return self._video()
                self._json({"error": "not found"}, 404)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _video(self):
            """The camera as an MJPEG stream, which an <img> shows without any script."""
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = None
            try:
                while True:
                    frame = viewer.feed.frame
                    if frame is not None and frame is not last:
                        last = frame
                        jpg = viewer.jpeg()
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(1 / 20)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            try:
                if self.path == "/api/say" and viewer.feed and viewer.feed.say:
                    n = int(self.headers.get("Content-Length", 0))
                    text = json.loads(self.rfile.read(n) or b"{}").get("text", "").strip()[:400]
                    if text:
                        viewer.feed.say(text)
                    return self._json({"ok": bool(text)})
                if self.path == "/api/trace":
                    n = int(self.headers.get("Content-Length", 0))
                    text = json.loads(self.rfile.read(n) or b"{}").get("text", "").strip()[:400]
                    if not text:
                        return self._json({"error": "say something first"}, 400)
                    return self._json(viewer.trace(text))
                self._json({"error": "not found"}, 404)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
    return H


def serve(cfg, mind=None, feed: Feed | None = None, port: int = 7342, open_browser: bool = True,
          block: bool = False) -> str:
    viewer = Viewer(cfg, mind, feed)
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), handler(viewer))
            break
        except OSError:
            continue
    else:
        raise OSError("no free port for the viewer")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    if mind is not None:
        threading.Thread(target=viewer.idle, daemon=True, name="viewer-idle").start()
    if open_browser:
        webbrowser.open(url)
    if block:
        print(f"[mind] {url}  (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    else:
        threading.Thread(target=server.serve_forever, daemon=True, name="viewer").start()
    return url
