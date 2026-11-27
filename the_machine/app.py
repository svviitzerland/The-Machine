"""Main loop that connects camera, microphone, voice and the Mind, and draws the overlay."""
from __future__ import annotations

import queue
import sys
import threading
import time

import cv2

from . import config as C
from .core.mind import Mind
from .brainview import BrainView
from .hud import Hud
from .mindview import MindView
from .cortex.llm import load_cortex
from .memory.store import Store
from .perception import models
from .perception.audio import make_ears
from .perception.vision import Camera, FaceEngine, load_object_engine
from .voice import Voice


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

    voice = Voice(cfg.speech.voice, cfg.speech.tts_model, cfg.speech.tts_quant, silent=silent,
                  on_start=started, on_end=ended, language=cfg.speech.language, name=cfg.name)
    cortex = load_cortex(cfg)
    if cortex is None:
        print("[machine] no cortex model in models/, so it can see and remember but not talk. "
              "Run: python -m the_machine download", file=sys.stderr)
    mind = Mind(cfg, store, voice.say, FaceEngine(str(paths["yunet"]), str(paths["sface"]), cfg.vision),
                load_object_engine(cfg), cortex=cortex)
    ears = make_ears(cfg.speech, inbox.put, cfg.speech.language, prefer_mic=use_mic)
    cam = Camera(cfg.vision)
    hud = Hud()
    mind_view, show_mind = MindView(), False
    brain_view, show_brain = BrainView(), False
    print(f"[machine] awake. voice={voice.backend}. Type or speak. Press m for its memory, n for its neural network, q to stop.",
          file=sys.stderr)
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
                if show_mind:
                    cv2.imshow("MIND", mind_view.draw(mind))
                if show_brain:
                    cv2.imshow("CORTEX", brain_view.draw(mind))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    show_mind = not show_mind
                    if not show_mind:
                        cv2.destroyWindow("MIND")
                if key == ord("n"):
                    show_brain = not show_brain
                    if not show_brain:
                        brain_view.close(mind)
                        cv2.destroyWindow("CORTEX")
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
