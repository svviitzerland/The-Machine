"""Speech input: an energy VAD feeding offline ASR (faster-whisper).

Falls back to typed input when there is no microphone or the audio packages are missing.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from ..config import SpeechConfig


class KeyboardEars:
    """Reads typed lines from stdin."""

    def __init__(self, on_text: Callable[[str], None]):
        self.on_text = on_text
        threading.Thread(target=self._loop, daemon=True, name="keyboard").start()

    def _loop(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if line:
                self.on_text(line)

    def close(self) -> None:
        pass


class MicEars:
    """Energy-gated VAD with hangover + faster-whisper int8 on CPU."""

    def __init__(self, cfg: SpeechConfig, on_text: Callable[[str], None], language: str | None = "en"):
        import sounddevice as sd  # noqa: F401  (import errors handled by caller)
        from faster_whisper import WhisperModel

        self.cfg, self.on_text, self.language = cfg, on_text, language
        self.model = WhisperModel(cfg.asr_model, device="cpu", compute_type=cfg.asr_compute, cpu_threads=2)
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self.muted = threading.Event()          # set while the Machine speaks, so it doesn't hear itself
        self._stop = threading.Event()
        self._sd = sd
        self.stream = sd.InputStream(samplerate=cfg.sample_rate, channels=1, dtype="float32",
                                     blocksize=int(cfg.sample_rate * 0.03), callback=self._cb)
        self.stream.start()
        threading.Thread(target=self._loop, daemon=True, name="ears").start()

    def _cb(self, indata, frames, t, status) -> None:
        if not self.muted.is_set():
            self._q.put(indata[:, 0].copy())

    def _loop(self) -> None:
        buf: list[np.ndarray] = []
        voiced, silence = 0.0, 0.0
        blk = 0.03
        # Rooms differ a lot, so the speech gate follows the background level:
        # the 20th percentile of the last ~3 s of loudness. Speech comes in
        # bursts, so it barely moves that percentile.
        history: deque[float] = deque(maxlen=100)
        while not self._stop.is_set():
            try:
                x = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            rms = float(np.sqrt(np.mean(x * x)))
            history.append(rms)
            if len(history) < 30:          # first second: just listen to the room
                continue
            floor = float(np.percentile(history, 20))
            gate = max(self.cfg.vad_energy, floor * self.cfg.vad_ratio)
            if rms >= gate:
                buf.append(x)
                voiced += blk
                silence = 0.0
            elif buf:
                buf.append(x)
                silence += blk
            if buf and (silence >= self.cfg.max_silence_s or len(buf) * blk >= self.cfg.max_utt_s):
                if voiced >= self.cfg.min_utt_s:
                    self._transcribe(np.concatenate(buf))
                buf, voiced, silence = [], 0.0, 0.0

    def _transcribe(self, audio: np.ndarray) -> None:
        segs, _ = self.model.transcribe(audio, language=self.language, beam_size=1, vad_filter=True,
                                        condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segs).strip()
        if text:
            self.on_text(text)

    def close(self) -> None:
        self._stop.set()
        self.stream.stop()


def make_ears(cfg: SpeechConfig, on_text: Callable[[str], None], language: str, prefer_mic: bool = True):
    if prefer_mic:
        try:
            ears = MicEars(cfg, on_text, language)
            print("[ears] microphone + faster-whisper ready (you can also type)", file=sys.stderr)
            KeyboardEars(on_text)
            return ears
        except Exception as e:  # missing optional deps or no mic
            print(f"[ears] microphone unavailable ({type(e).__name__}: {e}); type to talk", file=sys.stderr)
    return KeyboardEars(on_text)


def _now() -> float:
    return time.time()
