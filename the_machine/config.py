"""Central configuration. Every magic number of the system lives here so the paper
can report them and ablations can change them from one place."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class VisionConfig:
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps_limit: float = 8.0                 # perception rate; low = low CPU
    det_score: float = 0.8                 # YuNet confidence
    det_input_max: int = 320               # downscale frame for detection (speed)
    track_iou: float = 0.3                 # IoU to keep a track alive
    track_ttl_s: float = 1.0
    embed_every_n: int = 3                 # embed a track every N frames (not every frame)
    track_buffer: int = 12                 # embeddings kept per track (few-shot burst learning)


@dataclass
class MemoryConfig:
    db_path: Path = ROOT / "data" / "machine.db"
    # Open-set recognition on SFace cosine similarity. 0.363 is the OpenCV Zoo
    # reference threshold for SFace; per-concept thresholds adapt around it.
    tau_known: float = 0.363
    tau_min: float = 0.30
    tau_max: float = 0.55
    spread_k: float = 2.0                  # threshold = mean_sim - k * std_sim (clipped)
    spawn_tau: float = 0.55                # below this (same concept) → spawn new prototype
    max_protos_per_concept: int = 6        # multi-prototype: pose / lighting / glasses
    exemplars_per_concept: int = 40        # rehearsal memory size (bounded)
    # Episodic memory forgetting (Ebbinghaus-style exponential decay with rehearsal)
    half_life_h: float = 72.0
    prune_strength: float = 0.05
    # Sleep: discovering recurring strangers from unlabeled episodes
    stranger_tau: float = 0.45
    stranger_min_count: int = 6
    # Cross-situational word learning
    lex_window_s: float = 4.0
    lex_explicit_weight: float = 5.0
    lex_implicit_weight: float = 0.5
    lex_decay: float = 0.98                # per sleep


@dataclass
class MindConfig:
    tick_hz: float = 10.0
    habituation_s: float = 20.0            # don't greet the same person again within this
    curiosity_gain: float = 0.004          # per tick without novelty
    social_gain: float = 0.002             # per tick alone
    energy_drain: float = 0.00002          # per tick ("battery", as in Chappie)
    sleep_energy: float = 0.15             # below this → ask to sleep (consolidate)
    ask_cooldown_s: float = 25.0           # min time between unsolicited questions
    language: str = "en"                   # language pack code: the_machine/language/packs/<code>.py


@dataclass
class SpeechConfig:
    asr_model: str = "tiny.en"             # faster-whisper: tiny.en / base.en (English-only = faster, more accurate)
    asr_compute: str = "int8"
    sample_rate: int = 16000
    vad_energy: float = 0.012              # RMS gate (fallback VAD)
    min_utt_s: float = 0.4
    max_silence_s: float = 0.7
    piper_voice: Path = ROOT / "models" / "en_US-lessac-medium.onnx"


@dataclass
class Config:
    name: str = "Machine"
    vision: VisionConfig = field(default_factory=VisionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    mind: MindConfig = field(default_factory=MindConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    models_dir: Path = ROOT / "models"
    save_thumbnails: bool = False          # privacy: store embeddings, not faces, by default
    headless: bool = bool(os.environ.get("MACHINE_HEADLESS"))


def load() -> Config:
    cfg = Config()
    cfg.memory.db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    return cfg
