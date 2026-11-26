"""Optional local LLM (llama.cpp) for replies to anything that isn't a teaching intent.

It loads the first ``*.gguf`` file in ``models/``. The prompt includes what the Machine sees and
knows, so replies stay about its own world.
"""
from __future__ import annotations

from pathlib import Path


class LocalLLM:
    def __init__(self, path: Path, n_threads: int = 2):
        from llama_cpp import Llama
        self.llm = Llama(model_path=str(path), n_ctx=1024, n_threads=n_threads, verbose=False)

    @classmethod
    def maybe(cls, cfg) -> "LocalLLM | None":
        ggufs = sorted(Path(cfg.models_dir).glob("*.gguf"))
        if not ggufs:
            return None
        try:
            return cls(ggufs[0])
        except Exception:
            return None

    def reply(self, text: str, context: str) -> str:
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": context + " Answer in one or two short sentences, "
                       "honestly; if you don't know, say so and ask to be taught."},
                      {"role": "user", "content": text}],
            max_tokens=60, temperature=0.6)
        return out["choices"][0]["message"]["content"].strip()
