"""Benchmarks for per-stage latency, memory search scaling, sleep cost, RAM and disk.

Each run is repeated with one OpenCV thread to approximate a weak single-core device.
"""
from __future__ import annotations

import json
import os
import platform
import statistics as st
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil

from . import config as C


def _t(fn, n=30, warm=3):
    for _ in range(warm):
        fn()
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": round(st.median(xs), 3), "p90_ms": round(float(np.percentile(xs, 90)), 3)}


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def run(cfg: C.Config, threads: int | None = None) -> dict:
    import cv2
    if threads:
        cv2.setNumThreads(threads)
    from .memory.consolidation import Consolidator
    from .memory.lexicon import Lexicon
    from .memory.semantic import PrototypeMemory, l2n
    from .memory.store import Store
    from .language.pack import load_pack
    from .perception import models
    from .perception.vision import FaceEngine, ObjectEncoder

    out: dict = {
        "requested_threads": threads or "all",
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "processor": platform.processor(),
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "cv_threads": cv2.getNumThreads(),
        },
    }
    r0 = rss_mb()
    paths = models.ensure(cfg.models_dir, quiet=True)
    t0 = time.perf_counter()
    fe = FaceEngine(str(paths["yunet"]), str(paths["sface"]), cfg.vision)
    out["model_load_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    out["disk_mb"] = {k: round(p.stat().st_size / 1e6, 2) for k, p in paths.items()}

    sample = cfg.models_dir / "sample_face.jpg"           # public OpenCV sample image, fetched once
    if not sample.exists():
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg", sample)
    img = cv2.imread(str(sample))
    img = cv2.resize(img, (640, 480))
    faces = fe.detect(img)
    out["detect_640x480@320"] = _t(lambda: fe.detect(img))
    if len(faces):
        out["align+embed_per_face"] = _t(lambda: fe.embed(img, faces[0]))
    oe = ObjectEncoder()
    crop = oe.center_crop(img)
    out["object_descriptor"] = _t(lambda: oe.embed(crop))
    out["rss_after_models_mb"] = round(rss_mb(), 1)
    out["rss_models_delta_mb"] = round(rss_mb() - r0, 1)

    # recognition time and db size as the prototype count grows
    rng = np.random.default_rng(0)
    scale = {}
    for n in (100, 1_000, 10_000, 100_000):
        with tempfile.TemporaryDirectory() as td:
            s = Store(Path(td) / "b.db")
            mem = PrototypeMemory(s, cfg.memory, "face")
            P = l2n(rng.normal(size=(n, 128)).astype(np.float32))
            cid = s.new_concept("person", "x")
            s.many("INSERT INTO prototypes(concept_id,modality,vec,n,sim_mean,sim_m2,updated) VALUES(?,?,?,?,?,?,?)",
                   [(cid, "face", p.tobytes(), 5, 0.7, 0.1, 0.0) for p in P])
            mem.reload()
            q = P[0]
            scale[n] = {"recognize": _t(lambda: mem.recognize(q), n=20),
                        "db_mb": round((Path(td) / "b.db").stat().st_size / 1e6, 2)}
            s.close()
    out["recognize_vs_prototypes"] = scale

    # episode write throughput and bytes per episode
    with tempfile.TemporaryDirectory() as td:
        s = Store(Path(td) / "e.db")
        v = l2n(rng.normal(size=128)).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(2000):
            s.add_episode("see", modality="face", vec=v, salience=0.5)
        dt = time.perf_counter() - t0
        s._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        size = (Path(td) / "e.db").stat().st_size
        out["episode_write"] = {"per_s": round(2000 / dt), "bytes_per_episode": round(size / 2000)}

        # sleep cost: 20 people x 40 exemplars, plus 500 unlabelled sightings
        mem = PrototypeMemory(s, cfg.memory, "face")
        lex = Lexicon(s, cfg.memory, load_pack("en").content_words)
        centers = l2n(rng.normal(size=(25, 128)))
        for i in range(20):
            c = s.new_concept("person", f"p{i}")
            for x in l2n(centers[i] + 0.09 * rng.normal(size=(40, 128))):
                mem.learn(x, c)
                s.add_episode("learn", modality="face", concept_id=c, vec=x, exemplar=True)
        for x in l2n(centers[20:][rng.integers(0, 5, 500)] + 0.09 * rng.normal(size=(500, 128))):
            s.add_episode("see", modality="face", vec=x)
        t0 = time.perf_counter()
        rep = Consolidator(s, cfg.memory, mem, lex).sleep()
        out["sleep_20people_800ex_500unlab"] = {"seconds": round(time.perf_counter() - t0, 3),
                                               "strangers_found": rep.strangers_found,
                                               "protos_before": rep.prototypes_before,
                                               "protos_after": rep.prototypes_after}
        s.close()
    out["rss_peak_mb"] = round(rss_mb(), 1)
    # per-frame budget: one detection plus 1/N of an embedding per visible face
    det = out["detect_640x480@320"]["median_ms"]
    emb = out.get("align+embed_per_face", {}).get("median_ms", 0)
    out["est_cpu_load_1face_pct"] = round(100 * cfg.vision.fps_limit * (det + emb / cfg.vision.embed_every_n) / 1000, 1)
    return out


def main(cfg: C.Config, as_json: bool = False) -> None:
    # each configuration runs in a fresh process so RSS numbers are not contaminated
    import subprocess
    import sys
    res = {}
    for name, th in (("all_threads", 0), ("one_thread", 1)):
        code = (f"import json,sys; from the_machine import config as C; from the_machine.bench import run; "
                f"print(json.dumps(run(C.load(), threads={th} or None)))")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        res[name] = json.loads(r.stdout.strip().splitlines()[-1])
    outp = Path(__file__).resolve().parent.parent / "eval" / "results" / "bench.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2) if as_json else _pretty(res))
    print(f"\nsaved → {outp}")


def _pretty(res: dict) -> str:
    lines = []
    for k, r in res.items():
        lines.append(f"== {k}  ({r['platform']['machine']}, {r['platform']['cpus']} cpus, threads={r['requested_threads']})")
        lines.append(f"  models on disk: {r['disk_mb']}  load {r['model_load_ms']} ms  RSS {r['rss_after_models_mb']} MB")
        for key in ("detect_640x480@320", "align+embed_per_face", "object_descriptor"):
            if key in r:
                lines.append(f"  {key:24s} median {r[key]['median_ms']:.2f} ms  p90 {r[key]['p90_ms']:.2f} ms")
        for n, v in r["recognize_vs_prototypes"].items():
            lines.append(f"  recognize @ {n:>6} prototypes: {v['recognize']['median_ms']:.3f} ms  (db {v['db_mb']} MB)")
        lines.append(f"  episodes: {r['episode_write']}  sleep: {r['sleep_20people_800ex_500unlab']}")
        lines.append(f"  est. CPU load, 1 face @ {C.VisionConfig().fps_limit} fps: {r['est_cpu_load_1face_pct']}% of one core")
    return "\n".join(lines)
