"""The negative controls, run through the same code as the real scoring.

A citation checker that always passes and an evidence metric that always finds
support would both produce a repository full of excellent numbers. These exist
so that the good numbers mean the checker works rather than that it is asleep.
"""

from __future__ import annotations

import random

from pathquery.core.citation import Citation
from pathquery.core.evidence import Fact


def shifted_citations(citations: list[Citation], offset: int = 3) -> list[Citation]:
    """Move every span, keeping the quoted text as it was.

    The quote is deliberately left alone. A checker comparing the quote against
    the corpus at the new offsets must now disagree, and one that still passes
    these is not reading the offsets at all.
    """
    if offset == 0:
        raise ValueError("a shift of zero is not a control, it is the original")
    return [
        citation.model_copy(
            update={"start": max(citation.start + offset, 0), "end": max(citation.end + offset, 1)}
        )
        for citation in citations
    ]


def shuffled_gold(gold: dict[str, set[Fact]], seed: int) -> dict[str, set[Fact]]:
    """Give each question another question's supporting facts.

    A derangement, so no question keeps its own. The shape is unchanged: the
    same questions, each with the same number of supporting sentences, drawn
    from the same corpus. Only the pairing is wrong, and a metric that still
    reports recall against these is measuring itself.
    """
    questions = sorted(gold)
    if len(questions) < 2:
        raise ValueError("a permutation of fewer than two questions cannot derange anything")

    rng = random.Random(seed)
    for _ in range(100):
        shuffled = questions[:]
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(questions, shuffled, strict=True)):
            return {a: set(gold[b]) for a, b in zip(questions, shuffled, strict=True)}
    raise RuntimeError("could not draw a derangement in 100 attempts")
