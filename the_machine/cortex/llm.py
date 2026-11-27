"""Local LLM backend for the cortex, using llama.cpp.

Any instruction-tuned GGUF model in ``models/`` works. Output is constrained by
a grammar built from the JSON schema, so the cortex always gets valid JSON with
known action names. Thinking modes are switched off: the Machine has to answer
while the person is still looking at it.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import threading
from pathlib import Path


def _quiet() -> None:
    """llama.cpp writes its own log to stderr; keep only the errors."""
    import ctypes
    import llama_cpp
    if getattr(_quiet, "cb", None):
        return

    @llama_cpp.llama_log_callback
    def log(level, text, _):
        if level >= 4:                       # GGML_LOG_LEVEL_ERROR
            print(text.decode(errors="replace"), end="", file=sys.stderr)

    _quiet.cb = log                           # keep a reference, or it is freed
    llama_cpp.llama_log_set(log, ctypes.c_void_p(0))


class LlamaBackend:
    def __init__(self, path: Path, n_ctx: int = 4096, gpu_layers: int = -1, threads: int | None = None):
        from llama_cpp import Llama
        _quiet()
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        self.name = Path(path).stem
        self.llm = Llama(model_path=str(path), n_ctx=n_ctx, n_gpu_layers=gpu_layers,
                         n_threads=threads or max(2, (os.cpu_count() or 4) // 2), verbose=False)
        model = self.llm._model
        self.fmt = Jinja2ChatFormatter(template=self.llm.metadata["tokenizer.chat_template"],
                                       eos_token=model.token_get_text(self.llm.token_eos()),
                                       bos_token=model.token_get_text(self.llm.token_bos()))
        self._grammars: dict[str, object] = {}
        self.layers = int(self.llm.metadata.get(f"{self.llm.metadata.get('general.architecture')}.block_count", 0))
        self.recent = ""                     # the tail of what it is writing, for the brain view
        self.lock = threading.Lock()          # one generation at a time
        try:
            from .tap import Tap
            self.tap = Tap(self.llm)
        except Exception:                    # an older llama.cpp, or another platform: no brain view
            self.tap = None
        # Free the model before the interpreter tears down Metal, or llama.cpp aborts on exit.
        atexit.register(self.llm.close)

    def _grammar(self, schema: dict):
        from llama_cpp import LlamaGrammar
        key = json.dumps(schema, sort_keys=True)
        if key not in self._grammars:
            self._grammars[key] = LlamaGrammar.from_json_schema(key, verbose=False)
        return self._grammars[key]

    def _complete(self, messages: list[dict], schema: dict, max_tokens: int, stream: bool):
        prompt = self.fmt(messages=messages, enable_thinking=False).prompt
        return self.llm.create_completion(prompt, grammar=self._grammar(schema), max_tokens=max_tokens,
                                          temperature=0.6, top_p=0.9, top_k=40, repeat_penalty=1.05,
                                          stream=stream)

    def chat(self, messages: list[dict], schema: dict, max_tokens: int = 200) -> str:
        with self.lock:
            return self._complete(messages, schema, max_tokens, False)["choices"][0]["text"]

    def stream(self, messages: list[dict], schema: dict, max_tokens: int, on_text) -> str:
        raw = ""
        with self.lock:
            for chunk in self._complete(messages, schema, max_tokens, True):
                piece = chunk["choices"][0]["text"]
                raw += piece
                self.recent = (self.recent + piece)[-300:]
                on_text(piece)
        return raw


def find_model(models_dir: Path, preferred: list[str]) -> Path | None:
    """The first preferred model that is present, else any GGUF in the folder."""
    override = os.environ.get("MACHINE_LLM")
    if override:
        p = Path(override)
        return p if p.exists() else models_dir / override
    for name in preferred:
        if (models_dir / name).exists():
            return models_dir / name
    ggufs = sorted(models_dir.glob("*.gguf"))
    return ggufs[0] if ggufs else None


def load_cortex(cfg):
    """The cortex backed by a local LLM, or None if no model or llama.cpp is installed."""
    path = find_model(Path(cfg.models_dir), cfg.mind.llm_models)
    if path is None:
        return None
    try:
        from .cortex import LLMCortex
        return LLMCortex(LlamaBackend(path, gpu_layers=cfg.mind.llm_gpu_layers))
    except Exception as e:
        import sys
        print(f"[cortex] could not load {path.name}: {type(e).__name__}: {e}", file=sys.stderr)
        return None
