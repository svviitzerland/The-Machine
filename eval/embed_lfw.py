"""Embed LFW faces with the Machine's own perception stack (YuNet + SFace).

usage: python eval/embed_lfw.py              -> eval/.data/lfw_sface.npz
       python eval/embed_lfw.py --degrade    -> eval/.data/lfw_sface_degraded.npz

--degrade simulates a person seen across a room by a cheap webcam: the image
is shrunk so the face is about 25 px wide and upscaled again, darkened
(gamma 1.8), given sensor noise (sigma 8) and JPEG-compressed (quality 40).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

from the_machine import config as C
from the_machine.perception import models
from the_machine.perception.vision import FaceEngine

DATA = Path(__file__).resolve().parent / ".data"


def degrade(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    x = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = np.power(x, 1.8)
    x = x * 255.0 + rng.normal(0, 8, x.shape)
    x = x.clip(0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, 40])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def main() -> None:
    deg = "--degrade" in sys.argv
    rng = np.random.default_rng(0)
    cfg = C.load()
    paths = models.ensure(cfg.models_dir)
    fe = FaceEngine(str(paths["yunet"]), str(paths["sface"]), cfg.vision)
    d = fetch_lfw_people(data_home=str(DATA), min_faces_per_person=1, color=True, resize=1.0, slice_=None)
    imgs = d.images
    scale = 255.0 if imgs.max() <= 1.0 else 1.0
    X, y, ok, miss, t_det, t_emb = [], [], [], 0, [], []
    for img, label in zip(imgs, d.target):
        bgr = cv2.cvtColor((img * scale).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if deg:
            bgr = degrade(bgr, rng)
        t0 = time.perf_counter()
        faces = fe.detect(bgr)
        t_det.append(time.perf_counter() - t0)
        if len(faces) == 0:
            miss += 1
            if deg:                                   # keep alignment with the clean file
                X.append(np.zeros(128, np.float32)); y.append(label); ok.append(False)
            continue
        h, w = bgr.shape[:2]
        cx = faces[:, 0] + faces[:, 2] / 2 - w / 2
        cy = faces[:, 1] + faces[:, 3] / 2 - h / 2
        row = faces[int(np.argmin(cx ** 2 + cy ** 2))]          # LFW: the labelled face is centred
        t0 = time.perf_counter()
        X.append(fe.embed(bgr, row))
        t_emb.append(time.perf_counter() - t0)
        y.append(label)
        ok.append(True)
    X, y = np.stack(X).astype(np.float32), np.array(y)
    name = "lfw_sface_degraded.npz" if deg else "lfw_sface.npz"
    np.savez_compressed(DATA / name, X=X, y=y, ok=np.array(ok), names=d.target_names)
    print(f"images={len(imgs)} embedded={len(X)} no_face={miss} identities={len(np.unique(y))} "
          f"det_ms={1e3*np.median(t_det):.2f} emb_ms={1e3*np.median(t_emb):.2f}")


if __name__ == "__main__":
    main()
