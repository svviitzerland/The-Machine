"""Command line entry point: ``python -m the_machine <command>``.

``run`` (the default) starts camera, microphone and voice; ``chat`` is text-only with no camera.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import config as C


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="the_machine", description="The Machine: a lifelong-learning companion")
    ap.add_argument("command", nargs="?", default="run",
                    choices=["run", "chat", "sleep", "status", "bench", "download", "reset"])
    ap.add_argument("--no-mic", action="store_true", help="type instead of speaking")
    ap.add_argument("--no-window", action="store_true", help="no camera preview window")
    ap.add_argument("--silent", action="store_true", help="print instead of speaking")
    ap.add_argument("--db", help="path to memory database")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    cfg = C.load()
    if a.db:
        from pathlib import Path
        cfg.memory.db_path = Path(a.db)

    if a.command == "run":
        from .app import run
        run(cfg, use_mic=not a.no_mic, show=not a.no_window, silent=a.silent)
    elif a.command == "chat":
        _chat(cfg, a.silent)
    elif a.command == "download":
        from .perception import models
        for k, p in models.ensure(cfg.models_dir).items():
            print(f"{k:6s} {p}  {p.stat().st_size/1e6:.2f} MB")
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
    from .language.voice import Voice
    from .memory.store import Store
    voice = Voice(cfg.mind.language, cfg.speech.piper_voice, silent=silent)
    m = Mind(cfg, Store(cfg.memory.db_path), voice.say)
    print("[chat] text-only mode (no camera). Ctrl+D to quit.", file=sys.stderr)
    for line in sys.stdin:
        if line.strip():
            m.hear(line.strip())
            m.tick()
    time.sleep(0.5)


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
        "facts": [f"{r['subject']} is {r['object']}" for r in s.q("SELECT * FROM facts ORDER BY ts DESC LIMIT 20")],
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
