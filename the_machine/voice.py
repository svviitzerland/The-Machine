"""Offline text-to-speech on a worker thread.

Uses Phonon (a 31M-parameter neural TTS that runs faster than real time on
one CPU core) when ``ptts`` is installed, then macOS ``say``, then espeak-ng,
and otherwise only prints.
"""
from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import threading
from typing import Callable


class Voice:
    def __init__(self, voice: str = "Micah", model: str = "hf://gradium/phonon-31m-en", quant: str = "q8",
                 on_start: Callable[[], None] | None = None, on_end: Callable[[], None] | None = None,
                 silent: bool = False, language: str = "en", name: str = "Machine"):
        self.voice, self.language, self.name = voice, language, name
        self.on_start, self.on_end = on_start, on_end
        self.backend = "print"
        self._tts = None
        self._q: queue.Queue[tuple[str, float]] = queue.Queue()
        # Load the voice on the worker thread so the camera can open right away;
        # anything said before it is ready waits in the queue.
        threading.Thread(target=self._loop, args=(None if silent else (model, quant),),
                         daemon=True, name="voice").start()

    def _choose(self, model: str, quant: str) -> None:
        try:
            import ptts
            import sounddevice  # noqa: F401
            self._tts = ptts.TTS(model, lang=self.language, quant=quant)
            self._tts.synth("ready", voice=self.voice)   # loads the voice now, not mid-sentence
            self.backend = "phonon"
            return
        except Exception as e:
            hint = " (run: python -m the_machine download)" if "ffline" in str(e) or "ncomplete" in str(e) else ""
            print(f"[voice] neural voice unavailable{hint}: {type(e).__name__}", file=sys.stderr)
        if sys.platform == "darwin" and shutil.which("say"):
            self.backend = "say"
        elif shutil.which("espeak-ng"):
            self.backend = "espeak-ng"

    def say(self, text: str, rate: float = 0.0) -> None:
        print(f"\033[93m[{self.name}]\033[0m {text}", flush=True)
        self._q.put((text, rate))

    def _loop(self, load: tuple[str, str] | None) -> None:
        if load:
            self._choose(*load)
        while True:
            text, rate = self._q.get()
            if self.on_start:
                self.on_start()
            try:
                self._speak(text, rate)
            except Exception as e:
                print(f"[voice] {e}", file=sys.stderr)
            finally:
                if self.on_end:
                    self.on_end()
                self._q.task_done()

    def close(self) -> None:
        """Stop talking now, so audio isn't cut off mid-buffer when the program ends."""
        if self.backend == "phonon":
            import sounddevice as sd
            sd.stop()

    def wait(self) -> None:
        """Block until everything queued has been spoken."""
        self._q.join()

    def _speak(self, text: str, rate: float) -> None:
        if self.backend == "phonon":
            import sounddevice as sd
            self._tts.set_conditions({"duration_delta": rate})
            audio = self._tts.synth(text, voice=self.voice)
            sd.play(audio.reshape(-1), self._tts.sample_rate)
            sd.wait()
        elif self.backend == "say":
            wpm = int(185 / (1 + rate))          # positive rate means slower, as with duration_delta
            subprocess.run(["say", "-r", str(wpm), text], capture_output=True)
        elif self.backend == "espeak-ng":
            subprocess.run(["espeak-ng", "-v", self.language, "-s", str(int(165 / (1 + rate))), text],
                           capture_output=True)
