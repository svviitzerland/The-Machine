"""Mouth: offline text-to-speech on a worker thread.

Backends, best first: Piper (neural, ONNX, runs on a Pi) → macOS `say`
→ espeak-ng → print only."""
from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable


class Voice:
    def __init__(self, language: str = "en", piper_voice: Path | None = None,
                 on_start: Callable[[], None] | None = None, on_end: Callable[[], None] | None = None,
                 silent: bool = False):
        self.language = language
        self.on_start, self.on_end = on_start, on_end
        self.backend = "print"
        self._piper = None
        if not silent:
            self._choose(piper_voice)
        self._q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._loop, daemon=True, name="voice").start()

    def _choose(self, piper_voice: Path | None) -> None:
        if piper_voice and Path(piper_voice).exists():
            try:
                from piper import PiperVoice  # type: ignore
                import sounddevice  # noqa: F401
                self._piper = PiperVoice.load(str(piper_voice))
                self.backend = "piper"
                return
            except Exception as e:
                print(f"[voice] piper unavailable: {e}", file=sys.stderr)
        if sys.platform == "darwin" and shutil.which("say"):
            self.backend = "say"
        elif shutil.which("espeak-ng"):
            self.backend = "espeak-ng"

    def say(self, text: str) -> None:
        print(f"\033[93m[Machine]\033[0m {text}", flush=True)
        if self.backend != "print":
            self._q.put(text)

    def _loop(self) -> None:
        while True:
            text = self._q.get()
            if self.on_start:
                self.on_start()
            try:
                self._speak(text)
            except Exception as e:
                print(f"[voice] {e}", file=sys.stderr)
            finally:
                if self.on_end:
                    self.on_end()

    def _speak(self, text: str) -> None:
        if self.backend == "piper":
            import numpy as np
            import sounddevice as sd
            chunks = list(self._piper.synthesize(text))
            audio = np.concatenate([c.audio_float_array for c in chunks])
            sd.play(audio, chunks[0].sample_rate)
            sd.wait()
        elif self.backend == "say":
            subprocess.run(["say", text], capture_output=True)
        elif self.backend == "espeak-ng":
            subprocess.run(["espeak-ng", "-v", self.language, text], capture_output=True)
