"""Sleep: offline consolidation of episodes into prototype memory.

One cycle verifies self-labelled exemplars, prunes and replays each concept's exemplars to
rebuild its prototypes, clusters unlabelled faces into strangers, flags confusable concepts,
and forgets weak episodes.
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
    self_verified: int = 0
    self_demoted: int = 0
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
        rep.self_verified, rep.self_demoted = self._verify_self()
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

    def _exemplars(self, cid: int, sources: tuple[str, ...] | None = None) -> tuple[list[int], np.ndarray]:
        sql = ("SELECT id, vec, source FROM episodes WHERE concept_id=? AND modality=? AND exemplar=1 "
               "AND vec IS NOT NULL")
        args: list = [cid, self.faces.modality]
        if sources:
            sql += f" AND source IN ({','.join('?' * len(sources))})"
            args += list(sources)
        rows = self.store.q(sql + " ORDER BY ts", args)
        if not rows:
            return [], np.zeros((0, 0), np.float32)
        self._src = {r["id"]: r["source"] for r in rows}
        return [r["id"] for r in rows], l2n(np.stack([from_blob(r["vec"], np.float16) for r in rows]))

    def _verify_self(self) -> tuple[int, int]:
        """Demote self-labelled exemplars that don't clearly match their concept's trusted
        (teacher or sleep) exemplars, so a wrong self-label can't corrupt the concept."""
        anchors, owner = [], []
        for cid in self.faces.concepts():
            _, A = self._exemplars(cid, ("teacher", "sleep"))
            if len(A):
                k = max(1, min(self.cfg.max_protos_per_concept, math.ceil(len(A) / 8)))
                C, _ = spherical_kmeans(A, k) if len(A) >= k else (A, None)
                anchors.append(C)
                owner += [cid] * len(C)
        if not anchors:
            return 0, 0
        A, owner = np.vstack(anchors), np.array(owner)
        rows = self.store.q("SELECT id, vec, concept_id FROM episodes WHERE source='self' AND exemplar=1 "
                            "AND modality=? AND vec IS NOT NULL", (self.faces.modality,))
        demote, ok = [], 0
        for r in rows:
            x = l2n(from_blob(r["vec"], np.float16))
            s = A @ x
            mine = s[owner == r["concept_id"]]
            other = s[owner != r["concept_id"]]
            own = float(mine.max()) if mine.size else -1.0
            oth = float(other.max()) if other.size else -1.0
            if own >= self.cfg.tau_known and own - oth >= self.cfg.verify_margin:
                ok += 1
            else:
                demote.append((r["id"],))
        if demote:   # back to the unlabelled pool, where stranger discovery can pick them up
            self.store.many("UPDATE episodes SET exemplar=0, concept_id=NULL, consolidated=0, source='sense' "
                            "WHERE id=?", demote)
        return ok, len(demote)

    def _prune_exemplars(self, cid: int) -> int:
        ids, X = self._exemplars(cid)
        budget = self.cfg.exemplars_per_concept
        drop: list[int] = []
        keep = list(range(len(ids)))
        src = getattr(self, "_src", {})
        while len(keep) > budget:
            S = X[keep] @ X[keep].T
            np.fill_diagonal(S, -1)
            red = S.max(1)
            # drop the most redundant exemplar, preferring self-labelled over teacher-given ones
            selfish = np.array([src.get(ids[i]) == "self" for i in keep])
            if selfish.any():
                red = np.where(selfish, red, -np.inf)
            worst = int(np.argmax(red))
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
            self.store.many("UPDATE episodes SET concept_id=?, exemplar=1, source='sleep' WHERE id=?",
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
