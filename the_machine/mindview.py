"""A live map of what the Machine knows: itself in the middle, then the people,
things, facts and words it has learned, linked the way they are linked in memory.
Whatever it sees or talks about right now lights up, and signals run out to it.
"""
from __future__ import annotations

import math
import random
import time

import cv2
import numpy as np

# BGR
COLORS = {"self": (255, 255, 255), "person": (90, 210, 255), "maker": (0, 215, 255), "stranger": (255, 170, 70),
          "object": (220, 200, 90), "fact": (170, 120, 255), "word": (140, 140, 140)}
EDGE_TINTS = [(90, 120, 255), (90, 210, 140), (230, 200, 120), (120, 220, 255), (200, 130, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
REFRESH_S = 2.0


class Graph:
    def __init__(self):
        self.ids: list[str] = []
        self.kind: dict[str, str] = {}
        self.label: dict[str, str] = {}
        self.weight: dict[str, float] = {}
        self.edges: list[tuple[str, str, float]] = []
        self.pos: dict[str, np.ndarray] = {}
        self.vel: dict[str, np.ndarray] = {}

    def node(self, nid: str, kind: str, label: str, weight: float = 1.0) -> None:
        self.ids.append(nid)
        self.kind[nid], self.label[nid], self.weight[nid] = kind, label, weight

    def load(self, mind) -> None:
        """Rebuild nodes and edges from memory, keeping positions of nodes that already exist."""
        old_pos, old_vel = self.pos, self.vel
        self.__init__()
        q = mind.store.q
        self.node("self", "self", mind.cfg.name.upper(), 4.0)
        admin = mind.ethics.admin_id
        for r in q("SELECT id, kind, name, seen_count FROM concepts ORDER BY last_seen DESC LIMIT 60"):
            nid = f"c{r['id']}"
            kind = "maker" if r["id"] == admin else r["kind"] if r["kind"] in COLORS else "object"
            label = r["name"] or (f"#{r['id']}" if r["kind"] == "stranger" else "?")
            self.node(nid, kind, label, 1.0 + math.log1p(r["seen_count"] or 0) / 2)
            self.edges.append(("self", nid, 1.0))
        names = {self.label[n].lower(): n for n in self.ids if n != "self"}
        for i, r in enumerate(q("SELECT subject, relation, object FROM facts ORDER BY ts DESC LIMIT 40")):
            nid = f"f{i}"
            self.node(nid, "fact", " ".join(x for x in (r["relation"], r["object"]) if x)[:28], 0.8)
            self.edges.append((names.get(r["subject"].lower(), "self"), nid, 0.8))
        for i, r in enumerate(q("SELECT word, concept_id, weight FROM lexicon WHERE weight >= 1 "
                                "ORDER BY weight DESC LIMIT 60")):
            owner = f"c{r['concept_id']}"
            if owner not in self.kind or r["word"] == self.label[owner].lower():
                continue
            nid = f"w{r['word']}"
            if nid not in self.kind:
                self.node(nid, "word", r["word"], 0.6)
            self.edges.append((owner, nid, min(1.0, r["weight"] / 10)))
        for n in self.ids:
            if n in old_pos:
                self.pos[n], self.vel[n] = old_pos[n], old_vel[n]
            else:
                parent = next((a for a, b, _ in self.edges if b == n), None)
                base = old_pos.get(parent, np.zeros(2)) if parent else np.zeros(2)
                self.pos[n] = base + np.random.normal(0, 0.15, 2)
                self.vel[n] = np.zeros(2)
        self.pos["self"] = np.zeros(2)

    def settle(self, steps: int = 2) -> None:
        """A few steps of a force layout: everything repels, links pull, and the middle pulls gently."""
        n = len(self.ids)
        if n < 2:
            return
        idx = {k: i for i, k in enumerate(self.ids)}
        P = np.stack([self.pos[k] for k in self.ids])
        V = np.stack([self.vel[k] for k in self.ids])
        for _ in range(steps):
            d = P[:, None, :] - P[None, :, :]
            dist2 = (d ** 2).sum(-1) + 1e-3
            F = (d / dist2[..., None] * 0.012).sum(1)
            for a, b, w in self.edges:
                i, j = idx[a], idx[b]
                v = P[j] - P[i]
                L = np.linalg.norm(v) + 1e-6
                rest = 0.35 if a == "self" else 0.18
                f = v / L * (L - rest) * 0.08 * (0.5 + w)
                F[i] += f
                F[j] -= f
            F -= P * 0.02
            V = (V + F) * 0.82
            V[idx["self"]] = 0
            P = P + V
        for k, i in idx.items():
            self.pos[k], self.vel[k] = P[i], V[i]


class MindView:
    def __init__(self, size: tuple[int, int] = (960, 600)):
        self.w, self.h = size
        self.graph = Graph()
        self.loaded = 0.0
        self.signals: list[list] = []          # [edge index, progress, speed]
        self._bg = self._background()

    def _background(self) -> np.ndarray:
        y = np.linspace(0, 1, self.h, dtype=np.float32)[:, None]
        x = np.linspace(0, 1, self.w, dtype=np.float32)[None, :]
        glow = np.exp(-(((x - 0.5) / 0.6) ** 2 + ((y - 0.45) / 0.5) ** 2))
        b = 40 + 50 * glow
        g = 18 + 22 * glow
        r = 14 + 26 * glow + 18 * y
        return np.dstack([b, g, r]).astype(np.uint8)

    def _xy(self, p: np.ndarray, scale: float) -> tuple[int, int]:
        return int(self.w / 2 + p[0] * scale), int(self.h / 2 + p[1] * scale)

    def active(self, mind) -> set[str]:
        """Nodes the Machine is seeing or talking about right now."""
        on = {"self"}
        for t in list(mind.visible()) + list(mind.objects_in_view()):
            if t.known and t.concept_id:
                on.add(f"c{t.concept_id}")
        said = (mind.thought or "").lower()
        if said:
            for n, label in self.graph.label.items():
                if len(label) > 2 and label.lower() in said:
                    on.add(n)
        return on

    def draw(self, mind) -> np.ndarray:
        now = time.time()
        g = self.graph
        if now - self.loaded > REFRESH_S:
            g.load(mind)
            self.loaded = now
        g.settle()
        on = self.active(mind)
        pts = np.stack([g.pos[k] for k in g.ids]) if g.ids else np.zeros((1, 2))
        extent = max(0.4, float(np.abs(pts).max()))
        scale = 0.42 * min(self.w, self.h) / extent
        glow = np.zeros((self.h, self.w, 3), np.uint8)
        xy = {k: self._xy(g.pos[k], scale) for k in g.ids}

        for e, (a, b, w) in enumerate(g.edges):
            hot = a in on and b in on
            tint = EDGE_TINTS[e % len(EDGE_TINTS)]
            col = tuple(int(c * (0.9 if hot else 0.35 + 0.3 * w)) for c in tint)
            self._curve(glow, xy[a], xy[b], col, e, now, 2 if hot else 1)

        # signals run along the links to whatever is active
        hot_edges = [i for i, (a, b, _) in enumerate(g.edges) if b in on]
        if hot_edges and random.random() < 0.35:
            self.signals.append([random.choice(hot_edges), 0.0, random.uniform(0.6, 1.2)])
        alive = []
        for s in self.signals:
            s[1] += s[2] / 30
            if s[1] < 1 and s[0] < len(g.edges):
                a, b, _ = g.edges[s[0]]
                p = self._along(xy[a], xy[b], s[0], now, s[1])
                cv2.circle(glow, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
                alive.append(s)
        self.signals = alive[-80:]

        for k in g.ids:
            col = COLORS[g.kind[k]]
            r = int(2 + 2.2 * g.weight[k])
            if k in on:
                pulse = 0.5 + 0.5 * math.sin(now * 5)
                cv2.circle(glow, xy[k], r + 6 + int(4 * pulse), col, 2, cv2.LINE_AA)
            cv2.circle(glow, xy[k], r, col, -1, cv2.LINE_AA)

        bloom = cv2.GaussianBlur(glow, (0, 0), 6)
        out = cv2.add(self._bg, cv2.add(glow, cv2.addWeighted(bloom, 1.6, bloom, 0, 0)))
        for k in g.ids:
            if g.kind[k] == "word" and k not in on:
                continue
            col = (255, 255, 255) if k in on else (190, 190, 190)
            x, y = xy[k]
            r = int(2 + 2.2 * g.weight[k]) + (10 if k in on else 4)
            cv2.putText(out, g.label[k].upper(), (x + r, y - r // 2), FONT, 0.36 if k != "self" else 0.5, col, 1,
                        cv2.LINE_AA)
        cv2.rectangle(out, (0, 0), (self.w - 1, self.h - 1), (200, 190, 60), 1)
        return out

    @staticmethod
    def _control(p, q, e, now):
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        dx, dy = q[0] - p[0], q[1] - p[1]
        bend = 0.25 * math.sin(e * 1.7 + now * 0.6)
        return mx - dy * bend, my + dx * bend

    def _along(self, p, q, e, now, t):
        cx, cy = self._control(p, q, e, now)
        x = (1 - t) ** 2 * p[0] + 2 * (1 - t) * t * cx + t ** 2 * q[0]
        y = (1 - t) ** 2 * p[1] + 2 * (1 - t) * t * cy + t ** 2 * q[1]
        return int(x), int(y)

    def _curve(self, img, p, q, col, e, now, thick):
        pts = np.array([self._along(p, q, e, now, t) for t in np.linspace(0, 1, 14)], np.int32)
        cv2.polylines(img, [pts], False, col, thick, cv2.LINE_AA)
