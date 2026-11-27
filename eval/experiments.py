"""Experiments on LFW faces embedded by the Machine's own YuNet + SFace.

A: open-set class-incremental teaching. B: self-training over several days,
with and without sleep. C: finding recurring strangers during sleep.

Two conditions: "clean" uses the original photos everywhere. "far" keeps the
teacher's examples clean (taught up close) but everything the Machine sees
afterwards is degraded, as if seen across a room (see embed_lfw.py --degrade).

    python eval/experiments.py [--seeds 10] [--quick]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from the_machine.config import Config
from the_machine.memory.consolidation import Consolidator
from the_machine.memory.lexicon import Lexicon
from the_machine.memory.semantic import PrototypeMemory, l2n
from the_machine.memory.store import Store

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
BASE = Config().memory


def auroc(pos, neg):
    """Probability that a known face scores above an unknown one (ties count half)."""
    s = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = np.cumsum(cnt) - (cnt - 1) / 2
    r = ranks[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ccr_at_far(known_score, known_correct, unk_score, far=0.01):
    """Correct classification rate when the threshold is set so that `far` of unknowns get in."""
    t = np.quantile(unk_score, 1 - far)
    return float((known_correct & (known_score > t)).mean())


def oscr(known_score, known_correct, unk_score):
    """Area under correct-classification rate vs false-positive rate (Dhamija et al. 2018)."""
    thr = np.unique(np.concatenate([known_score, unk_score]))[::-1]
    ccr = [(known_correct & (known_score >= t)).mean() for t in thr]
    fpr = [(unk_score >= t).mean() for t in thr]
    return float(np.trapezoid([0] + ccr + [known_correct.mean()], [0] + fpr + [1.0]))


class GPMLearner:
    """The Machine's own memory, optionally with sleep."""

    def __init__(self, cfg, sleep=False, name="GPM"):
        self.cfg, self.name, self.use_sleep = cfg, name, sleep
        self.store = Store(":memory:")
        self.mem = PrototypeMemory(self.store, cfg, "face")
        self.cons = Consolidator(self.store, cfg, self.mem, Lexicon(self.store, cfg))
        self.cid, self.lab = {}, {}

    def concept(self, label):
        if label not in self.cid:
            c = self.store.new_concept("person", str(label))
            self.cid[label], self.lab[c] = c, label
        return self.cid[label]

    def teach(self, X, label):
        c = self.concept(label)
        for x in X:
            self.mem.learn(x, c)
            self.store.add_episode("learn", modality="face", concept_id=c, vec=x, exemplar=True, source="teacher")

    def predict(self, X):
        out = [self.mem.recognize(x) for x in X]
        lab = np.array([self.lab.get(r.concept_id, -1) for r in out])
        return lab, np.array([r.score for r in out]), np.array([r.known for r in out])

    def sleep(self):
        if self.use_sleep:
            return self.cons.sleep()

    def n_vectors(self):
        return len(self.mem)


class KNNLearner:
    """Nearest stored example with the fixed SFace threshold. Keeps everything."""
    name = "1-NN"

    def __init__(self, tau=BASE.tau_known):
        self.X, self.y, self.tau = [], [], tau

    def teach(self, X, label):
        self.X += list(l2n(X))
        self.y += [label] * len(X)

    def predict(self, X):
        S = l2n(X) @ np.stack(self.X).T
        j = S.argmax(1)
        s = S[np.arange(len(X)), j]
        return np.array(self.y)[j], s - self.tau, s >= self.tau

    def sleep(self):
        pass

    def n_vectors(self):
        return len(self.X)


class SLDALearner:
    """Streaming LDA (Hayes and Kanan 2020): class means plus one shared covariance."""
    name = "SLDA"

    def __init__(self, d=128, shrink=1e-2):
        self.mu, self.n, self.d, self.shrink = {}, {}, d, shrink
        self.S, self.N = np.zeros((d, d)), 0

    def teach(self, X, label):
        for x in l2n(X):
            if label not in self.mu:
                self.mu[label], self.n[label] = np.zeros(self.d), 0
            dx = x - self.mu[label]
            self.S = (self.N * self.S + np.outer(dx, dx) * self.n[label] / (self.n[label] + 1)) / (self.N + 1)
            self.N += 1
            self.n[label] += 1
            self.mu[label] += dx / self.n[label]

    def predict(self, X):
        labs = list(self.mu)
        M = np.stack([self.mu[k] for k in labs])
        W = M @ np.linalg.inv((1 - self.shrink) * self.S + self.shrink * np.eye(self.d))
        Z = l2n(X) @ W.T - 0.5 * np.sum(W * M, 1)
        j = Z.argmax(1)
        # SLDA has no rejection rule, so it accepts everything at its operating point
        return np.array(labs)[j], Z[np.arange(len(X)), j], np.ones(len(X), bool)

    def sleep(self):
        pass

    def n_vectors(self):
        return len(self.mu) + self.d


def make(name):
    fixed = dict(tau_min=BASE.tau_known, tau_max=BASE.tau_known)
    return {
        "NCM": lambda: GPMLearner(replace(BASE, max_protos_per_concept=1, **fixed), name="NCM"),
        "1-NN": KNNLearner,
        "SLDA": SLDALearner,
        "GPM-fixed": lambda: GPMLearner(replace(BASE, **fixed), name="GPM-fixed"),
        "GPM": lambda: GPMLearner(BASE),
        "GPM+sleep": lambda: GPMLearner(BASE, sleep=True, name="GPM+sleep"),
    }[name]()


class Data:
    def __init__(self, condition):
        clean = np.load(ROOT / ".data" / "lfw_sface.npz", allow_pickle=True)
        self.y = clean["y"]
        self.teach_X = l2n(clean["X"])
        if condition == "far":
            far = np.load(ROOT / ".data" / "lfw_sface_degraded.npz", allow_pickle=True)
            self.see_X, self.ok = l2n(far["X"]), far["ok"]
        else:
            self.see_X, self.ok = self.teach_X, np.ones(len(self.y), bool)

    def people(self, min_imgs):
        ids, cnt = np.unique(self.y[self.ok], return_counts=True)
        return ids[cnt >= min_imgs]

    def images(self, person, rng):
        return rng.permutation(np.where((self.y == person) & self.ok)[0])


def split(D, rng, n_known, shots, max_test=20, min_imgs=None):
    known = rng.permutation(D.people(min_imgs or shots + 3))[:n_known]
    tr, te = {}, {}
    for k in known:
        idx = D.images(k, rng)
        tr[k], te[k] = idx[:shots], idx[shots:shots + max_test]
    others = np.setdiff1d(np.unique(D.y[D.ok]), known)
    unk = np.concatenate([D.images(u, rng)[:5] for u in others])
    return known, tr, te, rng.permutation(unk)[:3000]


def exp_a(D, seed, methods, shots=5, sessions=10, per=10):
    rng = np.random.default_rng(seed)
    known, tr, te, unk = split(D, rng, sessions * per, shots)
    groups = [known[i * per:(i + 1) * per] for i in range(sessions)]
    res = {}
    for m in methods:
        L = make(m)
        R = np.zeros((sessions, sessions))
        t0 = time.perf_counter()
        for t, g in enumerate(groups):
            for k in g:
                L.teach(D.teach_X[tr[k]], k)
            L.sleep()
            for j in range(t + 1):
                idx = np.concatenate([te[k] for k in groups[j]])
                R[t, j] = (L.predict(D.see_X[idx])[0] == D.y[idx]).mean()
        seconds = time.perf_counter() - t0
        kidx = np.concatenate([te[k] for k in known])
        lab, ks, kacc = L.predict(D.see_X[kidx])
        _, us, uacc = L.predict(D.see_X[unk])
        correct = lab == D.y[kidx]
        res[m] = dict(
            acc_final=float(R[-1].mean()),
            forgetting=float(np.mean([R[:-1, j].max() - R[-1, j] for j in range(sessions - 1)])),
            auroc=auroc(ks, us), oscr=oscr(ks, correct, us), ccr_far1=ccr_at_far(ks, correct, us),
            ccr_op=float((correct & kacc).mean()), far_op=float(uacc.mean()),
            vectors=L.n_vectors(), seconds=seconds, R=R.tolist())
    return res


def exp_b(D, seed, days=6, n_known=30, shots=3, per_day=150, aggressive=False):
    """The teacher teaches once, then unlabelled sightings arrive for several days."""
    rng = np.random.default_rng(seed)
    known, tr, te, unk = split(D, rng, n_known, shots, max_test=10, min_imgs=shots + 25)
    pool = {k: np.setdiff1d(D.images(k, rng), np.concatenate([tr[k], te[k]])) for k in known}
    test_unk, stream_unk = unk[:600], unk[600:]
    # the live Mind self-trains above threshold + 0.15 with margin 0.15
    hi = 0.05 if aggressive else 0.15
    variants = {
        "teacher-only": (False, False, True),
        "self": (True, False, True),
        "self+sleep(no-verify)": (True, True, False),
        "self+sleep(verify)": (True, True, True),
    }
    days_plan = []
    for _ in range(days):
        seen = [rng.choice(pool[k], 3) for k in rng.choice(known, per_day // 3 * 2 // 3)]
        days_plan.append(rng.permutation(np.concatenate(seen + [rng.choice(stream_unk, per_day // 3)])))
    out = {}
    for name, (selftrain, sleep, verify) in variants.items():
        L = GPMLearner(BASE, sleep=sleep)
        if not verify:
            L.cons._verify_self = lambda: (0, 0)
        for k in known:
            L.teach(D.teach_X[tr[k]], k)
        curve, right, wrong = [], 0, 0
        for day in days_plan:
            for i in day:
                r = L.mem.recognize(D.see_X[i])
                if selftrain and r.known and r.sim >= r.threshold + hi and r.margin >= hi:
                    L.mem.learn(D.see_X[i], r.concept_id)
                    L.store.add_episode("see", modality="face", concept_id=r.concept_id, vec=D.see_X[i],
                                        exemplar=True, source="self")
                    truth = L.lab.get(r.concept_id)   # None for a stranger found during sleep
                    if truth == D.y[i]:
                        right += 1
                    elif truth is not None:
                        wrong += 1
                else:
                    L.store.add_episode("see", modality="face", concept_id=r.concept_id if r.known else None,
                                        vec=D.see_X[i])
            L.sleep()
            kidx = np.concatenate([te[k] for k in known])
            lab, ks, kacc = L.predict(D.see_X[kidx])
            _, us, uacc = L.predict(D.see_X[test_unk])
            correct = lab == D.y[kidx]
            curve.append(dict(acc=float(correct.mean()), ccr_op=float((correct & kacc).mean()),
                              far_op=float(uacc.mean()), oscr=oscr(ks, correct, us), protos=len(L.mem)))
        out[name] = dict(curve=curve, self_right=right, self_wrong=wrong)
    return out


def exp_noise(D, seed, noise=0.2, days=4, n_known=30, shots=3, per_day=150):
    """Stress test: a fraction of self-acquired labels is deliberately wrong.

    Self-training on LFW makes almost no mistakes, so to test the verification
    step we corrupt `noise` of the self labels (relabel the sighting as another
    known person) and check what sleep does with them."""
    rng = np.random.default_rng(seed)
    known, tr, te, unk = split(D, rng, n_known, shots, max_test=10, min_imgs=shots + 25)
    pool = {k: np.setdiff1d(D.images(k, rng), np.concatenate([tr[k], te[k]])) for k in known}
    plan = [rng.permutation(np.concatenate([rng.choice(pool[k], 3) for k in rng.choice(known, per_day // 3)]))
            for _ in range(days)]
    flips = [rng.random(len(d)) < noise for d in plan]
    wrong_to = [rng.choice(known, len(d)) for d in plan]
    out = {}
    for name, (sleep, verify) in {"no-sleep": (False, True), "sleep(no-verify)": (True, False),
                                  "sleep(verify)": (True, True)}.items():
        L = GPMLearner(BASE, sleep=sleep)
        if not verify:
            L.cons._verify_self = lambda: (0, 0)
        for k in known:
            L.teach(D.teach_X[tr[k]], k)
        bad_ids, good_ids = set(), set()
        for day, flip, to in zip(plan, flips, wrong_to):
            for i, f, w in zip(day, flip, to):
                label = w if f and w != D.y[i] else D.y[i]
                c = L.concept(label)
                L.mem.learn(D.see_X[i], c)
                eid = L.store.add_episode("see", modality="face", concept_id=c, vec=D.see_X[i],
                                          exemplar=True, source="self")
                (bad_ids if label != D.y[i] else good_ids).add(eid)
            L.sleep()
        kept = {r["id"] for r in L.store.q("SELECT id FROM episodes WHERE source='self' AND exemplar=1")}
        kidx = np.concatenate([te[k] for k in known])
        lab, ks, kacc = L.predict(D.see_X[kidx])
        _, us, uacc = L.predict(D.see_X[unk[:600]])
        correct = lab == D.y[kidx]
        out[name] = dict(acc=float(correct.mean()), ccr_op=float((correct & kacc).mean()),
                         far_op=float(uacc.mean()), oscr=oscr(ks, correct, us),
                         bad_removed=1 - len(bad_ids & kept) / max(len(bad_ids), 1),
                         good_kept=len(good_ids & kept) / max(len(good_ids), 1))
    return out


def exp_c(D, seed, n_known=20, n_recurring=10, sightings=8, singles=150):
    """Recurring strangers among one-off passers-by, then one sleep."""
    rng = np.random.default_rng(seed)
    rich = rng.permutation(D.people(sightings + 10))
    known, recur = rich[:n_known], rich[n_known:n_known + n_recurring]
    others = np.setdiff1d(np.unique(D.y[D.ok]), np.concatenate([known, recur]))
    teach = {k: D.images(k, rng)[:3] for k in known}
    L = GPMLearner(BASE, sleep=True)
    for k in known:
        L.teach(D.teach_X[teach[k]], k)
    seen, test = {}, {}
    for r in recur:
        idx = D.images(r, rng)
        seen[r], test[r] = idx[:sightings], idx[sightings:sightings + 10]
    passers = [D.images(o, rng)[:1] for o in rng.choice(others, singles, replace=False)]
    stream = rng.permutation(np.concatenate(list(seen.values()) + passers))
    for i in stream:
        r = L.mem.recognize(D.see_X[i])
        L.store.add_episode("see", modality="face", concept_id=r.concept_id if r.known else None, vec=D.see_X[i])
    rep = L.cons.sleep()

    S = D.see_X[stream]
    found, purity, fake = set(), [], 0
    for s in [r["id"] for r in L.store.q("SELECT id FROM concepts WHERE kind='stranger'")]:
        rows = L.store.q("SELECT vec FROM episodes WHERE concept_id=? AND exemplar=1", (s,))
        V = l2n(np.stack([np.frombuffer(r["vec"], np.float16).astype(np.float32) for r in rows]))
        members = D.y[stream[np.argmax(V @ S.T, axis=1)]]
        vals, cnt = np.unique(members, return_counts=True)
        top, p = vals[cnt.argmax()], cnt.max() / len(members)
        purity.append(p)
        if top in recur and p >= 0.5:
            found.add(top)
            L.lab[s] = top          # the teacher answers "who is this?" once
        else:
            fake += 1

    # Compare with the alternative: the teacher notices the person and names them from one sighting.
    B = GPMLearner(BASE)
    for k in known:
        B.teach(D.teach_X[teach[k]], k)
    for r in recur:
        B.teach(D.see_X[seen[r][-1:]], r)
    acc_cluster = [float((L.predict(D.see_X[test[r]])[0] == r).mean()) for r in found]
    acc_single = [float((B.predict(D.see_X[test[r]])[0] == r).mean()) for r in found]
    return dict(strangers=len(purity), recurring=n_recurring, found=len(found), fake=fake,
                purity=float(np.mean(purity)) if purity else 0.0,
                acc_named_cluster=float(np.mean(acc_cluster)) if acc_cluster else 0.0,
                acc_one_shot=float(np.mean(acc_single)) if acc_single else 0.0, sleep_s=rep.seconds)


def run(condition, seeds):
    D = Data(condition)
    methods = ["NCM", "1-NN", "SLDA", "GPM-fixed", "GPM", "GPM+sleep"]
    return {
        "meta": dict(condition=condition, n_images=int(D.ok.sum()), n_ids=int(len(np.unique(D.y[D.ok]))),
                     seeds=len(seeds)),
        "A": {s: exp_a(D, s, methods) for s in seeds},
        "A_shots": {k: {s: exp_a(D, s, ["NCM", "GPM", "GPM+sleep"], shots=k) for s in seeds} for k in (1, 3, 10)},
        "B": {s: exp_b(D, s) for s in seeds},
        "B_aggressive": {s: exp_b(D, s, aggressive=True) for s in seeds},
        "C": {s: exp_c(D, s) for s in seeds},
        "noise": {p: {s: exp_noise(D, s, noise=p) for s in seeds} for p in (0.1, 0.2, 0.3)},
    }


def summarize(res):
    def ms(v):
        v = np.asarray(v, float)
        return f"{100 * v.mean():5.1f} ± {100 * v.std():4.1f}"

    print(f"\n### {res['meta']}")
    A = res["A"]
    for m in next(iter(A.values())):
        g = lambda k: [A[s][m][k] for s in A]
        print(f"A {m:10s} acc {ms(g('acc_final'))} forget {ms(g('forgetting'))} AUROC {ms(g('auroc'))} "
              f"OSCR {ms(g('oscr'))} CCR@FAR1% {ms(g('ccr_far1'))} CCR@op {ms(g('ccr_op'))} FAR@op {ms(g('far_op'))} vec {np.mean(g('vectors')):.0f}")
    for k, AS in res["A_shots"].items():
        for m in next(iter(AS.values())):
            g = lambda key: [AS[s][m][key] for s in AS]
            print(f"  shots={k:<2} {m:10s} CCR@op {ms(g('ccr_op'))} OSCR {ms(g('oscr'))} FAR@op {ms(g('far_op'))}")
    for key in ("B", "B_aggressive"):
        B = res[key]
        for v in next(iter(B.values())):
            last = lambda k: [B[s][v]["curve"][-1][k] for s in B]
            rt = np.mean([B[s][v]["self_right"] for s in B])
            wr = np.mean([B[s][v]["self_wrong"] for s in B])
            print(f"{key:12s} {v:22s} CCR@op {ms(last('ccr_op'))} FAR@op {ms(last('far_op'))} "
                  f"OSCR {ms(last('oscr'))} self right/wrong {rt:.0f}/{wr:.1f}")
    for p, NS in res["noise"].items():
        for v in next(iter(NS.values())):
            g = lambda k: [NS[s][v][k] for s in NS]
            print(f"noise={p} {v:17s} acc {ms(g('acc'))} CCR@op {ms(g('ccr_op'))} FAR@op {ms(g('far_op'))} "
                  f"OSCR {ms(g('oscr'))} wrong removed {ms(g('bad_removed'))} right kept {ms(g('good_kept'))}")
    C = res["C"]
    g = lambda k: np.array([C[s][k] for s in C])
    print(f"C found {g('found').mean():.1f}/{g('recurring').mean():.0f} fake {g('fake').mean():.1f} "
          f"purity {ms(g('purity'))} named-cluster acc {ms(g('acc_named_cluster'))} vs one sighting {ms(g('acc_one_shot'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--conditions", default="clean,far")
    a = ap.parse_args()
    seeds = range(2 if a.quick else a.seeds)
    OUT.mkdir(exist_ok=True)
    for cond in a.conditions.split(","):
        res = run(cond, seeds)
        (OUT / f"experiments_{cond}.json").write_text(json.dumps(res, indent=1, default=float))
        summarize(res)


if __name__ == "__main__":
    main()
