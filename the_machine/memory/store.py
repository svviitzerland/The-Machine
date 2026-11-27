"""Long-term memory in a single SQLite file (WAL mode).

Vectors are stored as BLOBs and searched with NumPy. At household scale (under 1e5 vectors)
brute-force cosine is fast enough and simpler than an ANN index.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS concepts (
    id          INTEGER PRIMARY KEY,
    kind        TEXT NOT NULL,              -- person | object | stranger
    name        TEXT,                       -- NULL until the teacher names it
    created     REAL NOT NULL,
    last_seen   REAL,
    seen_count  INTEGER DEFAULT 0,
    role        TEXT DEFAULT 'user',        -- admin | user | stranger
    meta        TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_concepts_name ON concepts(name);

-- Semantic memory: a few prototypes per concept.
CREATE TABLE IF NOT EXISTS prototypes (
    id          INTEGER PRIMARY KEY,
    concept_id  INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    modality    TEXT NOT NULL,              -- face | object
    vec         BLOB NOT NULL,              -- float32, L2-normalised
    n           INTEGER NOT NULL,           -- samples absorbed
    sim_mean    REAL NOT NULL,              -- Welford stats of member similarity
    sim_m2      REAL NOT NULL,
    updated     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_protos_modality ON prototypes(modality);

-- Episodic memory: one row per event, decays unless rehearsed.
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY,
    ts          REAL NOT NULL,
    kind        TEXT NOT NULL,              -- see | hear | say | learn | feel | sleep
    modality    TEXT,
    concept_id  INTEGER REFERENCES concepts(id) ON DELETE SET NULL,
    vec         BLOB,                       -- float16 embedding, optional
    text        TEXT,
    salience    REAL NOT NULL DEFAULT 0.5,
    rehearsals  INTEGER NOT NULL DEFAULT 0,
    exemplar    INTEGER NOT NULL DEFAULT 0, -- kept for rehearsal during sleep
    source      TEXT NOT NULL DEFAULT 'sense', -- provenance: teacher | self | sleep | sense
    consolidated INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_ep_ts ON episodes(ts);
CREATE INDEX IF NOT EXISTS ix_ep_concept ON episodes(concept_id);
CREATE INDEX IF NOT EXISTS ix_ep_unlab ON episodes(modality, concept_id, consolidated);

-- Lexicon: word <-> concept association strengths (cross-situational learning).
CREATE TABLE IF NOT EXISTS lexicon (
    word        TEXT NOT NULL,
    concept_id  INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    weight      REAL NOT NULL,
    count       INTEGER NOT NULL,
    last        REAL NOT NULL,
    PRIMARY KEY (word, concept_id)
);

-- Declarative facts taught in conversation: (subject, relation, object).
CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY,
    subject     TEXT NOT NULL,
    relation    TEXT NOT NULL,
    object      TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    ts          REAL NOT NULL,
    UNIQUE(subject, relation, object)
);

-- Key/value state: birth time, admin, counters.
CREATE TABLE IF NOT EXISTS self (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);
"""


def f32(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def f16(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float16).tobytes()


def from_blob(b: bytes | None, dtype=np.float32) -> np.ndarray | None:
    if b is None:
        return None
    return np.frombuffer(b, dtype=dtype).astype(np.float32)


class Store:
    """One SQLite connection guarded by a lock. The Machine writes a few rows per second,
    so contention doesn't matter."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.executescript(SCHEMA)
            cols = {r[1] for r in self._db.execute("PRAGMA table_info(episodes)")}
            if "source" not in cols:                       # databases from before the source column
                self._db.execute("ALTER TABLE episodes ADD COLUMN source TEXT NOT NULL DEFAULT 'sense'")
                self._db.execute("UPDATE episodes SET source='teacher' WHERE kind='learn'")
        if self.get_self("born") is None:
            self.set_self("born", time.time())

    def q(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, tuple(args)).fetchall()

    def x(self, sql: str, args: Iterable[Any] = ()) -> int:
        with self._lock:
            cur = self._db.execute(sql, tuple(args))
            return cur.lastrowid

    def many(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        with self._lock:
            self._db.execute("BEGIN")
            try:
                self._db.executemany(sql, rows)
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def transaction(self):
        store = self

        class _Tx:
            def __enter__(self):
                store._lock.acquire()
                store._db.execute("BEGIN")
                return store

            def __exit__(self, et, ev, tb):
                try:
                    store._db.execute("ROLLBACK" if et else "COMMIT")
                finally:
                    store._lock.release()
                return False

        return _Tx()

    def get_self(self, key: str, default: Any = None) -> Any:
        r = self.q("SELECT value FROM self WHERE key=?", (key,))
        return json.loads(r[0]["value"]) if r else default

    def set_self(self, key: str, value: Any) -> None:
        self.x("INSERT INTO self(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
               (key, json.dumps(value)))

    def new_concept(self, kind: str, name: str | None = None, role: str = "user") -> int:
        now = time.time()
        return self.x("INSERT INTO concepts(kind,name,created,last_seen,role) VALUES(?,?,?,?,?)",
                      (kind, name, now, now, role))

    def concept(self, cid: int) -> sqlite3.Row | None:
        r = self.q("SELECT * FROM concepts WHERE id=?", (cid,))
        return r[0] if r else None

    def find_concept(self, name: str, kind: str | None = None) -> sqlite3.Row | None:
        sql, args = "SELECT * FROM concepts WHERE lower(name)=lower(?)", [name]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        r = self.q(sql + " ORDER BY seen_count DESC LIMIT 1", args)
        return r[0] if r else None

    def touch_concept(self, cid: int) -> None:
        self.x("UPDATE concepts SET last_seen=?, seen_count=seen_count+1 WHERE id=?", (time.time(), cid))

    def name_concept(self, cid: int, name: str, kind: str | None = None, role: str | None = None) -> None:
        self.x("UPDATE concepts SET name=?, kind=COALESCE(?,kind), role=COALESCE(?,role) WHERE id=?",
               (name, kind, role, cid))

    def delete_concept(self, cid: int) -> None:
        self.x("DELETE FROM concepts WHERE id=?", (cid,))

    def add_episode(self, kind: str, *, modality: str | None = None, concept_id: int | None = None,
                    vec: np.ndarray | None = None, text: str | None = None, salience: float = 0.5,
                    exemplar: bool = False, ts: float | None = None, source: str = "sense") -> int:
        return self.x(
            "INSERT INTO episodes(ts,kind,modality,concept_id,vec,text,salience,exemplar,source) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (ts or time.time(), kind, modality, concept_id, None if vec is None else f16(vec), text,
             float(salience), int(exemplar), source))

    def recent_episodes(self, n: int = 20, kinds: tuple[str, ...] | None = None) -> list[sqlite3.Row]:
        if kinds:
            ph = ",".join("?" * len(kinds))
            return self.q(f"SELECT * FROM episodes WHERE kind IN ({ph}) ORDER BY ts DESC LIMIT ?", (*kinds, n))
        return self.q("SELECT * FROM episodes ORDER BY ts DESC LIMIT ?", (n,))

    def stats(self) -> dict[str, int]:
        out = {}
        for t in ("concepts", "prototypes", "episodes", "lexicon", "facts"):
            out[t] = self.q(f"SELECT COUNT(*) c FROM {t}")[0]["c"]
        return out

    def close(self) -> None:
        with self._lock:
            self._db.close()
