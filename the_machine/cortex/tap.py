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
        return True

    def read(self) -> tuple[np.ndarray, int]:
        with self._lock:
            return self.frame, self.tokens
