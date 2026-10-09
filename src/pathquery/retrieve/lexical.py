"""BM25 over paragraphs.

Paragraphs rather than sentences, although the annotations are sentence level
and so is the final answer. A HotpotQA sentence averages about twenty words and
often contains a pronoun where the subject should be, so a sentence index
retrieves badly on exactly the questions that matter. The paragraph carries the
title and the surrounding context that make it findable, and picking sentences
out of the few paragraphs that came back is a cheaper problem than finding them
in two million.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np


@dataclass
class ParagraphIndex:
    retriever: bm25s.BM25
    titles: np.ndarray

    @classmethod
    def build(cls, titles: Sequence[str], texts: Sequence[str]) -> ParagraphIndex:
        """Index the title alongside the body.

        A bridge question names the first entity and the title is usually that
        entity, so leaving it out loses the strongest term in the paragraph.
        """
        tokens = bm25s.tokenize(
            [f"{title}. {text}" for title, text in zip(titles, texts, strict=True)],
            stopwords="en",
            show_progress=False,
        )
        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)
        return cls(retriever=retriever, titles=np.asarray(titles, dtype=object))

    def save(self, path: Path) -> None:
        self.retriever.save(str(path))
        np.save(path / "titles.npy", self.titles, allow_pickle=True)

    @classmethod
    def load(cls, path: Path) -> ParagraphIndex:
        return cls(
            retriever=bm25s.BM25.load(str(path), load_corpus=False),
            titles=np.load(path / "titles.npy", allow_pickle=True),
        )

    def search(self, queries: Sequence[str], k: int) -> list[list[str]]:
        """Rank paragraph titles for each query.

        Titles rather than positions, because a position is only meaningful to
        this index and a title is what the annotations and the citations both
        use.
        """
        if not queries:
            return []
        tokens = bm25s.tokenize(list(queries), stopwords="en", show_progress=False)
        depth = min(k, len(self.titles))
        positions, _ = self.retriever.retrieve(tokens, k=depth, show_progress=False)
        return [[str(t) for t in self.titles[row]] for row in positions]
