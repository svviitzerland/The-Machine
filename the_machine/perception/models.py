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
        from ..language.persona import load_persona
        voice = load_persona(cfg.mind.persona).voice
        print(f"[models] voice: phonon with voice {voice} ...", file=sys.stderr)
        ptts.TTS(cfg.speech.tts_model, lang=cfg.mind.language, quant=cfg.speech.tts_quant).synth("hi", voice=voice)
    except ImportError:
        print("[models] skipping phonon (install the 'voice' extra)", file=sys.stderr)
