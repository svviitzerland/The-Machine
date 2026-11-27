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


def ensure_speech(cfg) -> None:
    """Download the speech models once (Whisper for hearing, Phonon for the voice).

    Both libraries fetch from Hugging Face on first use. Doing it here, with
    the network allowed, means every later run can stay offline.
    """
    try:
        from .audio import skip_pyav
        skip_pyav()
        from faster_whisper import WhisperModel
        print(f"[models] speech recognition: whisper {cfg.speech.asr_model} ...", file=sys.stderr)
        WhisperModel(cfg.speech.asr_model, device="cpu", compute_type=cfg.speech.asr_compute)
    except ImportError:
        print("[models] skipping whisper (install the 'ears' extra)", file=sys.stderr)
    try:
        import ptts
        print(f"[models] voice: phonon with voice {cfg.speech.voice} ...", file=sys.stderr)
        ptts.TTS(cfg.speech.tts_model, lang=cfg.speech.language, quant=cfg.speech.tts_quant).synth(
            "hi", voice=cfg.speech.voice)
    except ImportError:
        print("[models] skipping phonon (install the 'voice' extra)", file=sys.stderr)


BRAIN = ModelSpec(
    "brain", "LFM3-1.7B-Q4_K_M.gguf",
    "https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/LFM3-1.7B-Q4_K_M.gguf",
    "the cortex: understanding and talking", "LFM Open License v1.0")


def fetch(url: str, dst: Path, tries: int = 50) -> None:
    """Download with resume, since a 1 GB file on a slow line will be cut off now and then."""
    import time
    tmp = dst.with_name(dst.name + ".part")
    for _ in range(tries):
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab" if have and r.status == 206 else "wb") as f:
                total = have + int(r.headers.get("Content-Length", 0)) if r.status == 206 else \
                    int(r.headers.get("Content-Length", 0))
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    print(f"\r[models] {dst.name}: {f.tell() / 1e6:.0f} / {total / 1e6:.0f} MB", end="",
                          file=sys.stderr)
            if total and tmp.stat().st_size >= total:
                tmp.rename(dst)
                print(file=sys.stderr)
                return
        except OSError as e:
            print(f"\n[models] connection dropped ({type(e).__name__}), resuming ...", file=sys.stderr)
            time.sleep(3)
    raise RuntimeError(f"could not download {url}")


def ensure_brain(cfg) -> Path | None:
    """Download the cortex model once, unless some GGUF model is already in models/."""
    if any(cfg.models_dir.glob("*.gguf")):
        return None
    print(f"[models] downloading {BRAIN.name} ({BRAIN.purpose}, about 1.1 GB) ...", file=sys.stderr)
    dst = cfg.models_dir / BRAIN.filename
    fetch(BRAIN.url, dst)
    return dst
