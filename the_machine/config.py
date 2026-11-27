"""All tunable numbers live here, so experiments and ablations can change them in one place."""
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
    fps_limit: float = 8.0                 # kept low to save CPU
    det_score: float = 0.8                 # YuNet confidence
    det_input_max: int = 320               # detect on a downscaled frame for speed
    track_iou: float = 0.3                 # IoU to keep a track alive
    track_ttl_s: float = 1.0
    embed_every_n: int = 3                 # embedding is the main per-frame cost
    track_buffer: int = 12                 # embeddings kept per track, all learned when a name is taught


@dataclass
class MemoryConfig:
    db_path: Path = ROOT / "data" / "machine.db"
    # Open-set recognition on SFace cosine similarity. 0.363 is the OpenCV Zoo
    # reference threshold for SFace; per-concept thresholds adapt around it.
    tau_known: float = 0.363
    tau_min: float = 0.30
    tau_max: float = 0.55
    spread_k: float = 2.0                  # threshold = mean_sim - k * std_sim (clipped)
    spawn_tau: float = 0.55                # a same-concept sample below this starts a new prototype
    max_protos_per_concept: int = 6        # room for pose, lighting, glasses
    exemplars_per_concept: int = 40        # rehearsal set size per concept
    # Episodic forgetting: exponential decay, slowed by rehearsal
    half_life_h: float = 72.0
    prune_strength: float = 0.05
    # Sleep: checking exemplars the Machine labelled itself
    verify_margin: float = 0.05            # self exemplar must beat other concepts' anchors by this
    # Sleep: clustering unlabelled faces into recurring strangers
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
    energy_drain: float = 0.00002          # per tick
    sleep_energy: float = 0.15             # below this the Machine asks to sleep
    ask_cooldown_s: float = 25.0           # min time between unsolicited questions
    language: str = "en"
    # Language cortex: first model found in models/ wins. Set MACHINE_LLM to force one.
    llm_models: list[str] = field(default_factory=lambda: [
        "Qwen3.5-4B-Q4_K_M.gguf", "LFM3-1.7B-Q4_K_M.gguf",
        "LFM2.5-1.2B-Instruct-Q4_K_M.gguf", "granite-4.0-1b-Q4_K_M.gguf"])
    llm_gpu_layers: int = -1               # -1 = use the GPU (Metal) if there is one, 0 = CPU only
    persona: str = "machine"                # the_machine/language/personas/<name>.py                   # language pack code: the_machine/language/packs/<code>.py


@dataclass
class SpeechConfig:
    asr_model: str = "tiny.en"             # English-only models are faster and more accurate
    asr_compute: str = "int8"
    sample_rate: int = 16000
    vad_energy: float = 0.012              # lowest RMS that can count as speech
    vad_ratio: float = 2.5                 # speech must be this many times louder than the room
    min_utt_s: float = 0.4
    max_silence_s: float = 0.7
    max_utt_s: float = 8.0                 # transcribe anyway after this, so a noisy room can't block it
    tts_model: str = "hf://gradium/phonon-31m-en"
    tts_quant: str = "q8"


@dataclass
class Config:
    name: str = "Machine"
    vision: VisionConfig = field(default_factory=VisionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    mind: MindConfig = field(default_factory=MindConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    models_dir: Path = ROOT / "models"
    save_thumbnails: bool = False          # off for privacy: keep embeddings, not face images
    headless: bool = bool(os.environ.get("MACHINE_HEADLESS"))


def load() -> Config:
    cfg = Config()
    cfg.memory.db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    return cfg
