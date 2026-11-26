"""Word to concept associations, learned from explicit teaching ("this is a cup") and from
cross-situational statistics over everything heard while concepts are in view (Yu & Smith, 2007).
"""
from __future__ import annotations

import time
from typing import Callable

from ..config import MemoryConfig
from .store import Store


class Lexicon:
    def __init__(self, store: Store, cfg: MemoryConfig, content_words: Callable[[str], list[str]]):
        # Tokenising comes from the language pack; the word table itself is language-agnostic.
        self.store, self.cfg, self.content_words = store, cfg, content_words

    def _add(self, word: str, cid: int, w: float) -> None:
        self.store.x(
            "INSERT INTO lexicon(word,concept_id,weight,count,last) VALUES(?,?,?,1,?) "
            "ON CONFLICT(word,concept_id) DO UPDATE SET weight=weight+excluded.weight, count=count+1, last=excluded.last",
            (word, cid, w, time.time()))

    def observe(self, text: str, in_view: list[int]) -> None:
        """Implicit cross-situational update from an utterance heard in a scene."""
        words = self.content_words(text)
        if not words or not in_view:
            return
        share = self.cfg.lex_implicit_weight / len(in_view)
        for w in words:
            for c in in_view:
                self._add(w, c, share)

    def teach(self, word: str, cid: int) -> None:
        for w in self.content_words(word) or [word.lower()]:
            self._add(w, cid, self.cfg.lex_explicit_weight)

    def reinforce(self, word: str, cid: int, delta: float) -> None:
        self.store.x("UPDATE lexicon SET weight=MAX(0.0, weight+?) WHERE word=? AND concept_id=?",
                     (delta, word.lower(), cid))

    def meaning(self, word: str) -> tuple[int, float] | None:
        rows = self.store.q("SELECT concept_id, weight FROM lexicon WHERE word=? ORDER BY weight DESC",
                            (word.lower(),))
        tot = sum(r["weight"] for r in rows)
        if not rows or tot <= 0:
            return None
        return rows[0]["concept_id"], rows[0]["weight"] / tot

    def word_for(self, cid: int) -> tuple[str, float] | None:
        rows = self.store.q("SELECT word, weight FROM lexicon WHERE concept_id=? ORDER BY weight DESC", (cid,))
        tot = sum(r["weight"] for r in rows)
        if not rows or tot <= 0:
            return None
        return rows[0]["word"], rows[0]["weight"] / tot

    def vocabulary_size(self, min_weight: float = 1.0) -> int:
        return self.store.q("SELECT COUNT(DISTINCT word) c FROM lexicon WHERE weight>=?", (min_weight,))[0]["c"]

    def decay(self) -> None:
        self.store.x("UPDATE lexicon SET weight=weight*?", (self.cfg.lex_decay,))
        self.store.x("DELETE FROM lexicon WHERE weight<0.05")
