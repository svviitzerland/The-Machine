"""Sleep: offline memory consolidation (Complementary Learning Systems).

In Person of Interest, the Machine was wiped every midnight. Ours does the
opposite: at night (or when its "battery" is low, like Chappie) it replays the
day's hippocampal episodes into the semantic memory, then lets weak
episodes fade. One sleep cycle runs these stages:

1. Exemplar selection: keep a bounded, *diverse* rehearsal set per concept
   (drop the most redundant exemplar first).
2. Replay & re-estimation: spherical k-means over each concept's exemplars
   rebuilds its prototypes and similarity statistics. This undoes order
   effects and the slow drift of online self-training.
3. Stranger discovery: unlabelled face episodes are clustered; a recurring
   unknown face becomes a nameless "stranger" concept the Machine will be
   curious about and ask its teacher to name.
4. Confusion detection: pairs of different concepts that overlap become
   questions for the teacher (active learning).
5. Forgetting: episodic strength = salience·2^(-age/half_life)·(1+rehearsals);
   weak, non-exemplar episodes are deleted (Ebbinghaus-style decay).
6. Lexicon decay: unused word associations weaken.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import MemoryConfig
from .lexicon import Lexicon
from .semantic import PrototypeMemory, l2n
from .store import Store, from_blob


@dataclass
class SleepReport:
    concepts_replayed: int = 0
    prototypes_before: int = 0
    prototypes_after: int = 0
    exemplars_dropped: int = 0
    strangers_found: int = 0
    confusions: list[tuple[int, int, float]] = field(default_factory=list)
    episodes_forgotten: int = 0
    seconds: float = 0.0


def farthest_point_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = [int(rng.integers(len(X)))]
    d = 1 - X @ X[idx[0]]
    for _ in range(1, k):
        idx.append(int(np.argmax(d)))
        d = np.minimum(d, 1 - X @ X[idx[-1]])
    return X[idx].copy()


def spherical_kmeans(X: np.ndarray, k: int, iters: int = 15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    C = farthest_point_init(X, k, rng)
    for _ in range(iters):
        a = np.argmax(X @ C.T, axis=1)
        newC = np.stack([l2n(X[a == j].mean(0)) if np.any(a == j) else C[j] for j in range(k)])
        if np.allclose(newC, C, atol=1e-5):
            break
        C = newC
    return C, np.argmax(X @ C.T, axis=1)


def welford(s: np.ndarray) -> tuple[int, float, float]:
    n = len(s)
    m = float(s.mean()) if n else 1.0
    return n, m, float(((s - m) ** 2).sum())


def leader_cluster(X: np.ndarray, tau: float) -> list[list[int]]:
    """Single-pass online clustering (cheap, O(N·C)); fine for one day of faces."""
    cents: list[np.ndarray] = []
    members: list[list[int]] = []
    for i, x in enumerate(X):
        if cents:
            s = np.stack(cents) @ x
            j = int(np.argmax(s))
            if s[j] >= tau:
                members[j].append(i)
                cents[j] = l2n(X[members[j]].mean(0))
                continue
        cents.append(x.copy())
        members.append([i])
    # second pass: reassign every point to its best final centroid (removes order effects)
    if cents:
        C = np.stack(cents)
        S = X @ C.T
        best = S.argmax(1)
        members = [[i for i in range(len(X)) if best[i] == j and S[i, j] >= tau] for j in range(len(C))]
    return [m for m in members if m]


class Consolidator:
    def __init__(self, store: Store, cfg: MemoryConfig, faces: PrototypeMemory, lexicon: Lexicon):
        self.store, self.cfg, self.faces, self.lex = store, cfg, faces, lexicon

    def sleep(self) -> SleepReport:
        t0 = time.perf_counter()
        rep = SleepReport(prototypes_before=len(self.faces))
        for cid in sorted(self.faces.concepts()):
            rep.exemplars_dropped += self._prune_exemplars(cid)
            rep.concepts_replayed += self._replay(cid)
        rep.strangers_found = self._discover_strangers()
        rep.confusions = self._confusions()
        rep.episodes_forgotten = self._forget()
        self.lex.decay()
        rep.prototypes_after = len(self.faces)
        rep.seconds = time.perf_counter() - t0
        self.store.set_self("last_sleep", time.time())
        self.store.set_self("sleeps", int(self.store.get_self("sleeps", 0)) + 1)
        self.store.set_self("questions", [[a, b, s] for a, b, s in rep.confusions])
        self.store.add_episode("sleep", text=(
            f"replayed={rep.concepts_replayed} protos {rep.prototypes_before}->{rep.prototypes_after} "
            f"strangers={rep.strangers_found} forgot={rep.episodes_forgotten}"), salience=0.9)
        return rep

    # ------------------------------------------------------------------
    def _exemplars(self, cid: int) -> tuple[list[int], np.ndarray]:
        rows = self.store.q("SELECT id, vec FROM episodes WHERE concept_id=? AND modality=? AND exemplar=1 "
                            "AND vec IS NOT NULL ORDER BY ts", (cid, self.faces.modality))
        if not rows:
            return [], np.zeros((0, 0), np.float32)
        return [r["id"] for r in rows], l2n(np.stack([from_blob(r["vec"], np.float16) for r in rows]))

    def _prune_exemplars(self, cid: int) -> int:
        ids, X = self._exemplars(cid)
        budget = self.cfg.exemplars_per_concept
        drop: list[int] = []
        keep = list(range(len(ids)))
        while len(keep) > budget:
            S = X[keep] @ X[keep].T
            np.fill_diagonal(S, -1)
            worst = int(np.argmax(S.max(1)))          # most redundant exemplar
            drop.append(ids[keep.pop(worst)])
        if drop:
            self.store.many("UPDATE episodes SET exemplar=0 WHERE id=?", [(i,) for i in drop])
        return len(drop)

    def _replay(self, cid: int) -> int:
        _, X = self._exemplars(cid)
        if len(X) < 3:
            return 0
        k = max(1, min(self.cfg.max_protos_per_concept, math.ceil(len(X) / 8)))
        C, a = spherical_kmeans(X, k)
        stats = []
        for j in range(k):
            s = X[a == j] @ C[j]
            stats.append(welford(s) if s.size else (1, 1.0, 0.0))
        keep = [j for j in range(k) if stats[j][0] > 0]
        self.faces.replace_concept(cid, C[keep], [stats[j] for j in keep])
        self.store.x("UPDATE episodes SET rehearsals=rehearsals+1 WHERE concept_id=? AND exemplar=1", (cid,))
        return 1

    def _discover_strangers(self) -> int:
        rows = self.store.q("SELECT id, vec FROM episodes WHERE modality=? AND concept_id IS NULL "
                            "AND consolidated=0 AND vec IS NOT NULL", (self.faces.modality,))
        if len(rows) < self.cfg.stranger_min_count:
            return 0
        X = l2n(np.stack([from_blob(r["vec"], np.float16) for r in rows]))
        found = 0
        for members in leader_cluster(X, self.cfg.stranger_tau):
            if len(members) < self.cfg.stranger_min_count:
                continue
            if self.faces.recognize(l2n(X[members].mean(0))).known:
                continue                                   # it was someone we know after all
            cid = self.store.new_concept("stranger", None, role="stranger")
            for i in members:
                self.faces.learn(X[i], cid)
            self.store.many("UPDATE episodes SET concept_id=?, exemplar=1 WHERE id=?",
                            [(cid, rows[i]["id"]) for i in members])
            found += 1
        self.store.x("UPDATE episodes SET consolidated=1 WHERE modality=? AND concept_id IS NULL",
                     (self.faces.modality,))
        return found

    def _confusions(self) -> list[tuple[int, int, float]]:
        P, owner = self.faces.P, self.faces.owner
        if len(P) < 2:
            return []
        S = P @ P.T
        out: dict[tuple[int, int], float] = {}
        for i, j in zip(*np.where(S > self.cfg.tau_max)):
            a, b = int(owner[i]), int(owner[j])
            if a < b:
                out[(a, b)] = max(out.get((a, b), 0.0), float(S[i, j]))
        return [(a, b, s) for (a, b), s in sorted(out.items(), key=lambda kv: -kv[1])]

    def _forget(self) -> int:
        now = time.time()
        hl = self.cfg.half_life_h * 3600
        rows = self.store.q("SELECT id, ts, salience, rehearsals FROM episodes "
                            "WHERE exemplar=0 AND kind NOT IN ('learn','sleep')")
        dead = [(r["id"],) for r in rows
                if r["salience"] * 2 ** (-(now - r["ts"]) / hl) * (1 + r["rehearsals"]) < self.cfg.prune_strength]
        if dead:
            self.store.many("DELETE FROM episodes WHERE id=?", dead)
        return len(dead)
