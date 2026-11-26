"""Growing Prototype Memory (GPM): the Machine's semantic memory.

A class-incremental, open-set recogniser that never needs gradient training:

* Embeddings come from a frozen pretrained encoder (SFace for faces).
* Each concept owns 1..K prototypes (running means on the unit sphere). A new
  prototype is *grown* only when a labelled sample is far from all of the
  concept's prototypes (cf. Growing-When-Required networks), so one person can
  be represented with and without glasses, by day and by night.
* Each prototype keeps Welford statistics of its members' cosine similarity;
  the open-set acceptance threshold of that prototype is
  ``clip(mean - k*std, tau_min, tau_max)``: tight for consistent concepts,
  looser for variable ones. Anything below threshold is *unknown*.
* Because old prototypes are never overwritten by other classes' gradients,
  there is no catastrophic forgetting by construction; drift is bounded by the
  replay-based re-estimation done during sleep (see consolidation.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import MemoryConfig
from .store import Store, f32, from_blob


def l2n(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


@dataclass
class Recognition:
    concept_id: int | None      # best concept (even if rejected)
    sim: float                  # cosine to best prototype
    threshold: float            # adaptive threshold of that prototype
    margin: float               # best - second-best concept similarity
    known: bool

    @property
    def score(self) -> float:   # open-set score (higher = more likely known)
        return self.sim - self.threshold


class PrototypeMemory:
    def __init__(self, store: Store, cfg: MemoryConfig, modality: str):
        self.store, self.cfg, self.modality = store, cfg, modality
        self.reload()

    # -- state -------------------------------------------------------------
    def reload(self) -> None:
        rows = self.store.q("SELECT * FROM prototypes WHERE modality=? ORDER BY id", (self.modality,))
        self.ids = np.array([r["id"] for r in rows], dtype=np.int64)
        self.owner = np.array([r["concept_id"] for r in rows], dtype=np.int64)
        self.n = np.array([r["n"] for r in rows], dtype=np.int64)
        self.sim_mean = np.array([r["sim_mean"] for r in rows], dtype=np.float64)
        self.sim_m2 = np.array([r["sim_m2"] for r in rows], dtype=np.float64)
        vecs = [from_blob(r["vec"]) for r in rows]
        self.mean = np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, 0), np.float32)
        self.P = l2n(self.mean) if len(vecs) else self.mean

    def __len__(self) -> int:
        return len(self.ids)

    def concepts(self) -> set[int]:
        return set(self.owner.tolist())

    def thresholds(self) -> np.ndarray:
        c = self.cfg
        std = np.sqrt(self.sim_m2 / np.maximum(self.n - 1, 1))
        t = self.sim_mean - c.spread_k * std
        t = np.where(self.n >= 3, t, c.tau_known)          # not enough evidence yet
        return np.clip(t, c.tau_min, c.tau_max)

    # -- inference ---------------------------------------------------------
    def recognize(self, x: np.ndarray) -> Recognition:
        if len(self) == 0:
            return Recognition(None, 0.0, self.cfg.tau_known, 0.0, False)
        x = l2n(np.asarray(x, np.float32))
        sims = self.P @ x
        j = int(np.argmax(sims))
        cid, s1 = int(self.owner[j]), float(sims[j])
        other = sims[self.owner != cid]
        s2 = float(other.max()) if other.size else -1.0
        thr = float(self.thresholds()[j])
        return Recognition(cid, s1, thr, s1 - s2, s1 >= thr)

    def recognize_batch(self, X: np.ndarray) -> list[Recognition]:
        return [self.recognize(x) for x in X]

    # -- learning ----------------------------------------------------------
    def learn(self, x: np.ndarray, concept_id: int) -> int:
        """Absorb one labelled sample. Returns the prototype id that absorbed it."""
        x = l2n(np.asarray(x, np.float32))
        mine = np.where(self.owner == concept_id)[0]
        if mine.size:
            sims = self.P[mine] @ x
            k = int(np.argmax(sims))
            j, s = int(mine[k]), float(sims[k])
            if s >= self.cfg.spawn_tau or mine.size >= self.cfg.max_protos_per_concept:
                return self._update(j, x, s)
        return self._spawn(x, concept_id)

    def learn_many(self, X: np.ndarray, concept_id: int) -> None:
        for x in X:
            self.learn(x, concept_id)

    def _spawn(self, x: np.ndarray, cid: int) -> int:
        now = time.time()
        pid = self.store.x(
            "INSERT INTO prototypes(concept_id,modality,vec,n,sim_mean,sim_m2,updated) VALUES(?,?,?,?,?,?,?)",
            (cid, self.modality, f32(x), 1, 1.0, 0.0, now))
        self.ids = np.append(self.ids, pid)
        self.owner = np.append(self.owner, cid)
        self.n = np.append(self.n, 1)
        self.sim_mean = np.append(self.sim_mean, 1.0)
        self.sim_m2 = np.append(self.sim_m2, 0.0)
        self.mean = x[None] if self.mean.size == 0 else np.vstack([self.mean, x[None]])
        self.P = l2n(self.mean)
        return pid

    def _update(self, j: int, x: np.ndarray, s: float) -> int:
        n = self.n[j] + 1
        self.mean[j] += (x - self.mean[j]) / n               # running mean of unit vectors
        self.P[j] = l2n(self.mean[j])
        d = s - self.sim_mean[j]                             # Welford on member similarity
        self.sim_mean[j] += d / n
        self.sim_m2[j] += d * (s - self.sim_mean[j])
        self.n[j] = n
        self.store.x("UPDATE prototypes SET vec=?,n=?,sim_mean=?,sim_m2=?,updated=? WHERE id=?",
                     (f32(self.mean[j]), int(n), float(self.sim_mean[j]), float(self.sim_m2[j]),
                      time.time(), int(self.ids[j])))
        return int(self.ids[j])

    def forget_concept(self, concept_id: int) -> None:
        self.store.x("DELETE FROM prototypes WHERE concept_id=? AND modality=?", (concept_id, self.modality))
        self.reload()

    def replace_concept(self, concept_id: int, centroids: np.ndarray, stats: list[tuple[int, float, float]]) -> None:
        """Used by sleep: swap a concept's prototypes for re-estimated ones."""
        now = time.time()
        with self.store.transaction() as s:
            s._db.execute("DELETE FROM prototypes WHERE concept_id=? AND modality=?", (concept_id, self.modality))
            for c, (n, m, m2) in zip(centroids, stats):
                s._db.execute(
                    "INSERT INTO prototypes(concept_id,modality,vec,n,sim_mean,sim_m2,updated) VALUES(?,?,?,?,?,?,?)",
                    (concept_id, self.modality, f32(c), int(n), float(m), float(m2), now))
        self.reload()
