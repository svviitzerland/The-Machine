"""Main loop that connects camera, microphone, voice and the Mind, and draws the overlay."""
from __future__ import annotations

import queue
import sys
import threading
import time

import cv2

from . import config as C
from .core.mind import Mind
from .hud import Hud
from .language.llm import load_cortex
from .language.persona import load_persona
from .language.voice import Voice
from .memory.store import Store
from .perception import models
from .perception.audio import make_ears
from .perception.vision import Camera, FaceEngine, ObjectEncoder

def run(cfg: C.Config, use_mic: bool = True, show: bool = True, silent: bool = False) -> None:
    paths = models.ensure(cfg.models_dir)
    store = Store(cfg.memory.db_path)
    inbox: queue.Queue[str] = queue.Queue()
    ears = None
    speaking = threading.Event()

    def started():
        speaking.set()
        if hasattr(ears, "muted"):
            ears.muted.set()

    def ended():
        speaking.clear()
        if hasattr(ears, "muted"):
            ears.muted.clear()

    voice = Voice(load_persona(cfg.mind.persona), cfg.speech.tts_model, cfg.speech.tts_quant, silent=silent,
                  on_start=started, on_end=ended)
    mind = Mind(cfg, store, voice.say, FaceEngine(str(paths["yunet"]), str(paths["sface"]), cfg.vision),
                ObjectEncoder(), cortex=load_cortex(cfg))
    ears = make_ears(cfg.speech, inbox.put, mind.lang.asr_code, prefer_mic=use_mic)
    cam = Camera(cfg.vision)
    hud = Hud()
    print(f"[machine] awake. voice={voice.backend}. Type or speak. Ctrl+C to stop.", file=sys.stderr)
    period = 1.0 / cfg.vision.fps_limit
    last, fps = time.time(), 0.0
    try:
        while True:
            t0 = time.time()
            while not inbox.empty():
                text = inbox.get()
                print(f"\033[96m[you]\033[0m {text}", flush=True)
                hud.hear(text)
                mind.hear(text)
            frame = cam.read()
            if frame is not None:
                mind.see(frame)
            mind.tick()
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now
            if show and frame is not None and not cfg.headless:
                cv2.imshow("THE MACHINE", hud.draw(frame, mind, fps, ears, speaking.is_set()))
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
