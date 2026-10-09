"""Choosing which sentences inside the retrieved paragraphs are the evidence.

Each one comes back with the citation that proves where it came from, built
from the paragraph's own offsets rather than by searching for the text again.
Searching for it would find the first occurrence, which is not necessarily the
sentence that was scored, and the citation would then point at a different
place that happens to read the same.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pathquery.core import Citation, Fact
from pathquery.corpus.hotpot import Paragraph


@dataclass(frozen=True)
class Evidence:
    fact: Fact
    citation: Citation
    score: float
    hop: int


def _overlap(query_terms: set[str], sentence: str) -> float:
    """Proportion of the question's terms the sentence accounts for.

    Deliberately crude. This is the lexical half of the fusion and its job is
    to be a floor that the dense half and the reranker have to beat, not to be
    good on its own.
    """
    terms = {t.strip(".,;:()\"'").lower() for t in sentence.split()}
    terms.discard("")
    if not query_terms:
        return 0.0
    return len(query_terms & terms) / len(query_terms)


def select_evidence(
    query: str,
    titles: Sequence[str],
    paragraphs: Mapping[str, Paragraph],
    *,
    hop: int = 1,
    per_paragraph: int = 3,
) -> list[Evidence]:
    """Score every sentence in the retrieved paragraphs, keeping the best few each.

    Capped per paragraph so one long passage cannot fill the whole result. A
    multi-hop question needs evidence from two different paragraphs, and a
    ranking that lets the first one occupy all ten places has lost before the
    second hop begins.
    """
    query_terms = {t.strip(".,;:()\"'").lower() for t in query.split()}
    query_terms.discard("")

    found: list[Evidence] = []
    for rank, title in enumerate(titles):
        paragraph = paragraphs.get(title)
        if paragraph is None:
            continue
        spans = paragraph.spans()
        scored: list[Evidence] = []
        for index, sentence in enumerate(paragraph.sentences):
            start, end = spans[index]
            if end <= start:
                continue
            # The paragraph's rank decays the sentence score, so a good
            # sentence in a poorly matched paragraph does not outrank a good
            # one in the paragraph the retriever was confident about.
            score = _overlap(query_terms, sentence) / (1.0 + rank * 0.1)
            scored.append(
                Evidence(
                    fact=Fact(document=title, sentence=index),
                    citation=Citation(document=title, start=start, end=end, quoted=sentence),
                    score=score,
                    hop=hop,
                )
            )
        scored.sort(key=lambda e: -e.score)
        found.extend(scored[:per_paragraph])

    found.sort(key=lambda e: -e.score)
    return found
