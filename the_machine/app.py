"""Main loop that connects camera, microphone, voice and the Mind, and serves the page you watch it on."""
from __future__ import annotations

import queue
import sys
import threading
import time

from . import config as C
from .body import DIRECTIONS, Aim, Head
from .core.mind import Mind
from .cortex.llm import load_cortex
from .memory.store import Store
from .perception import models
from .perception.audio import make_ears
from .perception.vision import Camera, FaceEngine, load_object_engine
from .viewer import Feed, serve
from .voice import Voice


def run(cfg: C.Config, use_mic: bool = True, browser: bool = True, silent: bool = False) -> None:
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
    # People type on the page, so the terminal's input isn't read.
    ears = make_ears(cfg.speech, inbox.put, cfg.speech.language, prefer_mic=use_mic, keyboard=False)
    cam = Camera(cfg.vision)
    head = Head(cfg.body.address, cfg.body) if cfg.body.address else None
    aim = Aim(cfg.body)
    feed = Feed(say=inbox.put, ears=ears, speaking=speaking, head=head)
    mind.has_body = head is not None
    url = serve(cfg, mind, feed, open_browser=browser and not cfg.headless)
    print(f"[machine] awake at {url}  voice={voice.backend}. Ctrl+C to stop.", file=sys.stderr)
    period = 1.0 / cfg.vision.fps_limit
    last = time.time()
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
                feed.frame = frame
                if head:
                    if feed.point:
                        aim.look_at(*feed.point, time.time())
                        feed.point = None
                    g, now = mind.gaze, time.time()
                    if g and now < g["until"] and "direction" in g:     # it chose a direction
                        aim.look_at(*DIRECTIONS[g["direction"]], now, hold=0.2)
                    chosen = mind.gaze_box()
                    boxes = [chosen] if chosen is not None else [t.box for t in mind.visible()]
                    head.look(*aim.update(boxes, frame.shape[1], frame.shape[0], time.time()))
            mind.tick()
            now = time.time()
            feed.fps = 0.9 * feed.fps + 0.1 / max(now - last, 1e-3)
            last = now
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        voice.close()
        if head:
            head.close()
        cam.close()
        ears.close()
        store.close()
        print("[machine] asleep. memories saved.", file=sys.stderr)
