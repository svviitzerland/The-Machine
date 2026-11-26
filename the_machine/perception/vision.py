"""Eyes: camera → faces (YuNet) → tracks → embeddings (SFace); plus a cheap
object encoder for things the teacher holds up to the camera.

Cost control (the Machine must run on a Raspberry Pi):
* detection on a downscaled frame (≤320 px), at ≤8 fps;
* a lightweight IoU tracker, so a face is embedded every N frames per *track*
  rather than every frame, and identity is decided by voting over the track.
"""
from __future__ import annotations

import itertools
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import VisionConfig


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


@dataclass
class Track:
    id: int
    box: np.ndarray                     # x, y, w, h in full-frame pixels
    row: np.ndarray                     # raw YuNet row (box + landmarks + score)
    first: float
    last: float
    frames: int = 0
    embeddings: deque = field(default_factory=lambda: deque(maxlen=12))
    votes: Counter = field(default_factory=Counter)
    concept_id: int | None = None       # current identity decision
    sim: float = 0.0
    known: bool = False
    announced: bool = False

    @property
    def area(self) -> float:
        return float(self.box[2] * self.box[3])

    def mean_embedding(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        m = np.mean(np.stack(self.embeddings), 0)
        return m / max(np.linalg.norm(m), 1e-12)


class FaceEngine:
    def __init__(self, yunet: str, sface: str, cfg: VisionConfig):
        self.cfg = cfg
        self.det = cv2.FaceDetectorYN.create(yunet, "", (320, 320), cfg.det_score, 0.3, 50)
        self.rec = cv2.FaceRecognizerSF.create(sface, "")
        self._size = (0, 0)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        s = min(1.0, self.cfg.det_input_max / max(h, w))
        small = cv2.resize(frame, (int(w * s), int(h * s))) if s < 1 else frame
        sz = (small.shape[1], small.shape[0])
        if sz != self._size:
            self.det.setInputSize(sz)
            self._size = sz
        _, faces = self.det.detect(small)
        if faces is None:
            return np.zeros((0, 15), np.float32)
        faces = faces.copy()
        faces[:, :14] /= s                               # back to full-res coordinates
        return faces

    def embed(self, frame: np.ndarray, row: np.ndarray) -> np.ndarray:
        aligned = self.rec.alignCrop(frame, row)
        f = self.rec.feature(aligned).reshape(-1).astype(np.float32)
        return f / max(np.linalg.norm(f), 1e-12)


class ObjectEncoder:
    """Training-free appearance descriptor (HSV colour histogram + HOG shape).
    Deliberately tiny (no model file); a CLIP-class ONNX encoder can be
    dropped in behind the same ``embed`` interface."""

    @staticmethod
    def hog(g: np.ndarray, cell: int = 8, bins: int = 9) -> np.ndarray:
        """Minimal HOG (unsigned gradients, per-cell histograms, global L2).
        Written in NumPy because OpenCV 5 dropped ``HOGDescriptor``."""
        g = g.astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
        mag = np.hypot(gx, gy)
        ang = (np.degrees(np.arctan2(gy, gx)) % 180.0) / (180.0 / bins)
        b = np.minimum(ang.astype(np.int32), bins - 1)
        h, w = g.shape
        cy, cx = h // cell, w // cell
        idx = ((np.arange(h)[:, None] // cell) * cx + np.arange(w)[None, :] // cell) * bins + b
        hist = np.bincount(idx.ravel(), weights=mag.ravel(), minlength=cy * cx * bins)
        return np.sqrt(hist)                              # power-normalise

    def embed(self, crop: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(cv2.resize(crop, (96, 96)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).reshape(-1)
        hist = np.sqrt(hist / max(hist.sum(), 1))         # Hellinger
        g = cv2.cvtColor(cv2.resize(crop, (64, 64)), cv2.COLOR_BGR2GRAY)
        h = self.hog(g)
        h = h / max(np.linalg.norm(h), 1e-12)
        v = np.concatenate([hist, 0.7 * h]).astype(np.float32)
        v -= v.mean()                                     # centre so cosine is discriminative
        return v / max(np.linalg.norm(v), 1e-12)

    @staticmethod
    def center_crop(frame: np.ndarray, frac: float = 0.5) -> np.ndarray:
        h, w = frame.shape[:2]
        ch, cw = int(h * frac), int(w * frac)
        y, x = (h - ch) // 2, (w - cw) // 2
        return frame[y:y + ch, x:x + cw]


class Tracker:
    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self.tracks: dict[int, Track] = {}
        self._ids = itertools.count(1)

    def update(self, faces: np.ndarray, now: float) -> tuple[list[Track], list[Track]]:
        """Returns (active tracks, tracks that just disappeared)."""
        used: set[int] = set()
        for row in faces:
            box = row[:4]
            best, best_iou = None, self.cfg.track_iou
            for t in self.tracks.values():
                if t.id in used:
                    continue
                v = iou(box, t.box)
                if v > best_iou:
                    best, best_iou = t, v
            if best is None:
                best = Track(next(self._ids), box, row, now, now, embeddings=deque(maxlen=self.cfg.track_buffer))
                self.tracks[best.id] = best
            best.box, best.row, best.last = box, row, now
            best.frames += 1
            used.add(best.id)
        gone = [t for t in self.tracks.values() if now - t.last > self.cfg.track_ttl_s]
        for t in gone:
            del self.tracks[t.id]
        active = [t for t in self.tracks.values() if t.id in used]
        return active, gone


class Camera:
    """Grabs frames on a background thread; consumers always get the latest."""

    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("camera not available (check permissions / index)")
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True, name="camera").start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = f
            else:
                time.sleep(0.05)

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self) -> None:
        self._stop.set()
        self.cap.release()
