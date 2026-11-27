"""Local LLM backend for the language cortex, using llama.cpp.

Any instruction-tuned GGUF model in ``models/`` works. Output is constrained by
a grammar built from the JSON schema, so the cortex always gets valid JSON with
known action names. Thinking modes are switched off: the Machine has to answer
while the person is still looking at it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


class LlamaBackend:
    def __init__(self, path: Path, n_ctx: int = 4096, gpu_layers: int = -1, threads: int | None = None):
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        self.name = Path(path).stem
        self.llm = Llama(model_path=str(path), n_ctx=n_ctx, n_gpu_layers=gpu_layers,
                         n_threads=threads or max(2, (os.cpu_count() or 4) // 2), verbose=False)
        model = self.llm._model
        self.fmt = Jinja2ChatFormatter(template=self.llm.metadata["tokenizer.chat_template"],
                                       eos_token=model.token_get_text(self.llm.token_eos()),
                                       bos_token=model.token_get_text(self.llm.token_bos()))
        self._grammars: dict[str, object] = {}

    def _grammar(self, schema: dict):
        from llama_cpp import LlamaGrammar
        key = json.dumps(schema, sort_keys=True)
        if key not in self._grammars:
            self._grammars[key] = LlamaGrammar.from_json_schema(key, verbose=False)
        return self._grammars[key]

    def chat(self, messages: list[dict], schema: dict, max_tokens: int = 120) -> str:
        prompt = self.fmt(messages=messages, enable_thinking=False).prompt
        out = self.llm.create_completion(prompt, grammar=self._grammar(schema), max_tokens=max_tokens,
                                         temperature=0.5, top_p=0.9, top_k=40, repeat_penalty=1.05)
        return out["choices"][0]["text"]


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
    """Language cortex backed by a local LLM, or None if no model or llama.cpp is installed."""
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
