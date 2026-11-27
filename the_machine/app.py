"""Main loop that connects camera, microphone, voice and the Mind, and draws the overlay."""
from __future__ import annotations

import queue
import sys
import time

import cv2
import numpy as np

from . import config as C
from .core.mind import Mind
from .language.pack import STAGE_NAMES
from .language.llm import load_cortex
from .language.persona import load_persona
from .language.voice import Voice
from .memory.store import Store
from .perception import models
from .perception.audio import make_ears
from .perception.vision import Camera, FaceEngine, ObjectEncoder

# BGR: yellow admin, white known user, blue stranger, red unknown
COLORS = {"admin": (0, 215, 255), "user": (235, 235, 235), "stranger": (255, 160, 60), None: (60, 60, 230)}


def draw(frame: np.ndarray, mind: Mind, fps: float) -> np.ndarray:
    out = frame.copy()
    for t in mind.visible():
        x, y, w, h = [int(v) for v in t.box]
        c = mind.store.concept(t.concept_id) if t.known and t.concept_id else None
        role = c["role"] if c else None
        col = COLORS.get(role, COLORS[None])
        L = max(10, w // 5)                                  # corner bracket length
        for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            cv2.line(out, (px, py), (px + dx * L, py), col, 2)
            cv2.line(out, (px, py), (px, py + dy * L), col, 2)
        label = (c["name"] or f"STRANGER #{c['id']}") if c else "UNKNOWN"
        tag = {"admin": "ADMIN", "user": "KNOWN", "stranger": "UNIDENTIFIED"}.get(role, "ANALYZING")
        cv2.rectangle(out, (x, y - 34), (x + max(w, 150), y - 2), (20, 20, 20), -1)
        cv2.putText(out, f"{tag}", (x + 4, y - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        cv2.putText(out, f"{label.upper()}  {t.sim:.2f}", (x + 4, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if mind.pending and mind.pending.kind == "object":
        h, w = out.shape[:2]
        cv2.rectangle(out, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 215, 255), 1)
    d = mind.drives
    lines = [f"THE MACHINE  stage:{STAGE_NAMES[mind.stage]}  fps:{fps:.1f}",
             f"curiosity {d.curiosity:.2f}  social {d.social:.2f}  energy {d.energy:.2f}",
             f"faces:{len(mind.faces.concepts())} objects:{len(mind.objects.concepts())} "
             f"words:{mind.lex.vocabulary_size()}  det {mind.metrics['detect_ms']:.0f}ms emb {mind.metrics['embed_ms']:.0f}ms",
             f"> {mind.thought[:70]}"]
    cv2.rectangle(out, (0, 0), (out.shape[1], 16 * len(lines) + 8), (10, 10, 10), -1)
    for i, s in enumerate(lines):
        cv2.putText(out, s, (8, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
    return out


def run(cfg: C.Config, use_mic: bool = True, show: bool = True, silent: bool = False) -> None:
    paths = models.ensure(cfg.models_dir)
    store = Store(cfg.memory.db_path)
    inbox: queue.Queue[str] = queue.Queue()
    ears = None
    voice = Voice(load_persona(cfg.mind.persona), cfg.speech.tts_model, cfg.speech.tts_quant, silent=silent,
                  on_start=lambda: ears and hasattr(ears, "muted") and ears.muted.set(),
                  on_end=lambda: ears and hasattr(ears, "muted") and ears.muted.clear())
    mind = Mind(cfg, store, voice.say, FaceEngine(str(paths["yunet"]), str(paths["sface"]), cfg.vision),
                ObjectEncoder(), cortex=load_cortex(cfg))
    ears = make_ears(cfg.speech, inbox.put, mind.lang.asr_code, prefer_mic=use_mic)
    cam = Camera(cfg.vision)
    print(f"[machine] awake. voice={voice.backend}. Type or speak. Ctrl+C to stop.", file=sys.stderr)
    period = 1.0 / cfg.vision.fps_limit
    last, fps = time.time(), 0.0
    try:
        while True:
            t0 = time.time()
            while not inbox.empty():
                text = inbox.get()
                print(f"\033[96m[you]\033[0m {text}", flush=True)
                mind.hear(text)
            frame = cam.read()
            if frame is not None:
                mind.see(frame)
            mind.tick()
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now
            if show and frame is not None and not cfg.headless:
                cv2.imshow("THE MACHINE", draw(frame, mind, fps))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        ears.close()
        if show:
            cv2.destroyAllWindows()
        store.close()
        print("[machine] asleep. memories saved.", file=sys.stderr)
