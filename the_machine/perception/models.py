"""Model registry and one-time download. After that the Machine needs no network access."""
from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    purpose: str
    license: str


ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODELS: dict[str, ModelSpec] = {
    "yunet": ModelSpec(
        "yunet", "face_detection_yunet_2023mar.onnx",
        f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "face detection + 5 landmarks", "MIT"),
    "sface": ModelSpec(
        "sface", "face_recognition_sface_2021dec.onnx",
        f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "128-d face embedding", "Apache-2.0"),
}


def path_of(models_dir: Path, key: str) -> Path:
    return models_dir / MODELS[key].filename


def ensure(models_dir: Path, keys: list[str] | None = None, quiet: bool = False) -> dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for key in keys or list(MODELS):
        spec = MODELS[key]
        dst = models_dir / spec.filename
        if not dst.exists() or dst.stat().st_size < 1024:
            if not quiet:
                print(f"[models] downloading {spec.name} ({spec.purpose}) ...", file=sys.stderr)
            tmp = dst.with_suffix(".part")
            urllib.request.urlretrieve(spec.url, tmp)
            tmp.rename(dst)
        out[key] = dst
    return out
