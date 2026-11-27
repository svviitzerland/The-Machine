"""Command line entry point: ``python -m the_machine <command>``.

``run`` (the default) starts camera, microphone and voice; ``chat`` is text-only with no camera.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import config as C


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="the_machine", description="The Machine: a lifelong-learning companion")
    ap.add_argument("command", nargs="?", default="run",
                    choices=["run", "chat", "mind", "sleep", "status", "bench", "download", "reset"])
    ap.add_argument("--no-mic", action="store_true", help="type instead of speaking")
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser")
    ap.add_argument("--body", help="host:port of its head, e.g. localhost:7351 for the Wokwi simulator")
    ap.add_argument("--silent", action="store_true", help="print instead of speaking")
    ap.add_argument("--db", help="path to memory database")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    cfg = C.load()
    if a.db:
        from pathlib import Path
        cfg.memory.db_path = Path(a.db)

    if a.command != "download":
        # Models are fetched once by `download`; after that the Machine never touches the network.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    if a.body:
        cfg.body.address = a.body
    if a.command == "run":
        from .app import run
        run(cfg, use_mic=not a.no_mic, browser=not a.no_browser, silent=a.silent)
    elif a.command == "chat":
        _chat(cfg, a.silent)
    elif a.command == "mind":
        from .viewer import serve
        serve(cfg, open_browser=not a.no_browser, block=True)
    elif a.command == "download":
        from .perception import models
        for k, p in models.ensure(cfg.models_dir).items():
            print(f"{k:6s} {p}  {p.stat().st_size/1e6:.2f} MB")
        models.ensure_speech(cfg)
        models.ensure_brain(cfg)
        print("all models ready. The Machine now runs offline.")
    elif a.command == "sleep":
        from .core.mind import Mind
        from .memory.store import Store
        m = Mind(cfg, Store(cfg.memory.db_path), print)
        rep = m.sleeper.sleep()
        print(json.dumps(rep.__dict__, indent=2, default=str))
    elif a.command == "status":
        _status(cfg, a.json)
    elif a.command == "bench":
        from .bench import main as bench
        bench(cfg, as_json=a.json)
    elif a.command == "reset":
        if input(f"Delete all memories at {cfg.memory.db_path}? type 'forget everything': ") == "forget everything":
            for suf in ("", "-wal", "-shm"):
                p = cfg.memory.db_path.with_name(cfg.memory.db_path.name + suf)
                p.unlink(missing_ok=True)
            print("memory erased.")


def _chat(cfg: C.Config, silent: bool) -> None:
    from .core.mind import Mind
    from .cortex.llm import load_cortex
    from .memory.store import Store
    from .voice import Voice
    voice = Voice(cfg.speech.voice, cfg.speech.tts_model, cfg.speech.tts_quant, silent=silent,
                  language=cfg.speech.language, name=cfg.name)
    cortex = load_cortex(cfg)
    if cortex is None:
        sys.exit("[chat] no cortex model in models/. Run: python -m the_machine download")
    m = Mind(cfg, Store(cfg.memory.db_path), voice.say, cortex=cortex)
    print("[chat] text-only mode (no camera). Ctrl+D to quit.", file=sys.stderr)
    for line in sys.stdin:
        if line.strip():
            m.hear(line.strip())
            m.tick()
            while m.thinking:
                time.sleep(0.05)
                m.tick()
    voice.wait()


def _status(cfg: C.Config, as_json: bool) -> None:
    from .memory.store import Store
    s = Store(cfg.memory.db_path)
    born = float(s.get_self("born"))
    info = {
        "db": str(cfg.memory.db_path),
        "db_bytes": cfg.memory.db_path.stat().st_size if cfg.memory.db_path.exists() else 0,
        "age_hours": round((time.time() - born) / 3600, 2),
        "sleeps": s.get_self("sleeps", 0),
        "admin_id": s.get_self("admin_id"),
        "counts": s.stats(),
        "concepts": [dict(r) for r in s.q(
            "SELECT c.id, c.kind, c.name, c.role, c.seen_count, COUNT(p.id) AS prototypes FROM concepts c "
            "LEFT JOIN prototypes p ON p.concept_id=c.id GROUP BY c.id ORDER BY c.id")],
        "facts": [" ".join(x for x in (r["subject"], r["relation"], r["object"]) if x)
                  for r in s.q("SELECT * FROM facts ORDER BY ts DESC LIMIT 20")],
    }
    if as_json:
        print(json.dumps(info, indent=2))
        return
    print(f"The Machine: age {info['age_hours']} h, {info['sleeps']} sleeps, db {info['db_bytes']/1024:.0f} KB")
    print("counts:", info["counts"])
    for c in info["concepts"]:
        mark = "*" if c["id"] == info["admin_id"] else " "
        print(f" {mark} #{c['id']:<3} {c['kind']:8s} {c['name'] or '(unnamed)':16s} seen={c['seen_count']:<5} "
              f"protos={c['prototypes']}")
    for f in info["facts"]:
        print("   fact:", f)


if __name__ == "__main__":
    main()
