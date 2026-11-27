"""Reads the activity of every layer of the cortex while it thinks.

llama.cpp calls back for each tensor it computes. We ask only for the output of
each layer (the residual stream, one vector per layer) of the newest token, and
only while someone is watching, so it costs nothing when the view is closed.
"""
from __future__ import annotations

import ctypes
import pathlib
import threading

import numpy as np


class Tap:
    def __init__(self, llm, every: int = 3):
        import llama_cpp
        from llama_cpp import llama_cpp as L
        from llama_cpp._internals import LlamaContext

        lib = pathlib.Path(llama_cpp.__file__).parent / "lib"
        base = ctypes.CDLL(str(next(lib.glob("libggml-base.*dylib"), None) or next(lib.glob("libggml-base*"))))
        base.ggml_get_name.restype = ctypes.c_char_p
        base.ggml_get_name.argtypes = [ctypes.c_void_p]
        base.ggml_nbytes.restype = ctypes.c_size_t
        base.ggml_nbytes.argtypes = [ctypes.c_void_p]
        base.ggml_backend_tensor_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        self._base = base

        meta = llm.metadata
        arch = meta.get("general.architecture", "llama")
        self.layers = int(meta[f"{arch}.block_count"])
        self.width = int(meta[f"{arch}.embedding_length"])
        self.on = False
        self.every, self._k, self._take = every, 0, False   # read one token in `every`: reading stalls the GPU
        self.tokens = 0                      # tokens seen while watching
        self.log: list[np.ndarray] | None = None   # every frame, while tracing
        self.frame = np.zeros((self.layers, self.width), np.float32)
        self._next = np.zeros_like(self.frame)
        self._lock = threading.Lock()
        self._cb = L.ggml_backend_sched_eval_callback(self._eval)   # keep a reference, or it is freed

        # The callback is a context setting, so rebuild the context with it.
        old = llm._ctx
        llm.context_params.cb_eval = self._cb
        llm._ctx = LlamaContext(model=llm._model, params=llm.context_params, verbose=False)
        old.close()

    def _eval(self, t, ask, _):
        name = self._base.ggml_get_name(t)
        if ask:
            if name == b"l_out-0":
                self._k += 1
                self._take = self.on and self._k % self.every == 0
            return self._take and name.startswith(b"l_out-")
        i = int(name[6:])
        n = self._base.ggml_nbytes(t)
        buf = np.empty(n // 4, np.float32)
        self._base.ggml_backend_tensor_get(t, buf.ctypes.data, 0, n)
        self._next[i] = buf[-self.width:]                 # the newest token
        if i == self.layers - 1:
            with self._lock:
                self.frame = self._next.copy()
                self.tokens += 1
                if self.log is not None:
                    self.log.append(self.frame)
        return True

    def read(self) -> tuple[np.ndarray, int]:
        with self._lock:
            return self.frame, self.tokens


def pool(frame: np.ndarray, groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Each layer's neurons in `groups` blocks: the strongest |activation| in each block and which neuron it was."""
    layers, width = frame.shape
    g = width // groups
    blocks = np.abs(frame[:, :g * groups]).reshape(layers, groups, g)
    arg = blocks.argmax(-1)
    return blocks.max(-1), arg + np.arange(groups)[None, :] * g


def trace(backend, messages: list[dict], said: str, max_new: int = 48, groups: int = 64) -> dict:
    """Run the cortex on one turn and keep every layer's activity for every token
    of what was said and of what it answers."""
    from .cortex import SCHEMA
    llm, tap = backend.llm, backend.tap
    prompt = backend.fmt(messages=messages, enable_thinking=False).prompt
    tokens = llm.tokenize(prompt.encode(), special=True)
    cut = prompt.rfind(said)
    head = llm.tokenize(prompt[:cut].encode(), special=True) if cut > 0 else tokens[:-24]
    if tokens[:len(head)] != head:
        head = tokens[:-24]
    rest = tokens[len(head):]
    out: list[int] = []
    with backend.lock:
        old = tap.on, tap.every
        try:
            llm.reset()
            tap.on = False
            llm.eval(head)
            tap.on, tap.every, tap.log = True, 1, []
            for t in rest[:-1]:
                llm.eval([t])
            for t in llm.generate(rest[-1:], top_k=40, top_p=0.9, temp=0.6, repeat_penalty=1.05, reset=False,
                                  grammar=backend._grammar(SCHEMA)):
                if t == llm.token_eos() or len(out) >= max_new:
                    break
                out.append(t)
            frames = tap.log
        finally:
            tap.on, tap.every = old
            tap.log = None
            llm.reset()
    ids = (rest + out)[:len(frames)]
    words = [llm.detokenize([t], special=True).decode(errors="replace") for t in ids]
    act, idx = zip(*(pool(f, groups) for f in frames)) if frames else ((), ())
    return {"tokens": words, "said": len(rest), "layers": tap.layers, "width": tap.width, "groups": groups,
            "act": [np.round(a, 3).tolist() for a in act], "idx": [i.tolist() for i in idx],
            "reply": llm.detokenize(out).decode(errors="replace")}
