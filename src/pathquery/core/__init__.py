"""The citation rule and the evidence metrics, free of any retriever.

Every number this repository publishes is computed here. Keeping it free of
models and indexes is what lets the citation check be tested against hand
written spans rather than against another run of the same pipeline, which is
the only way an off-by-one shows up as a failing test instead of as a slightly
better score.
"""

from pathquery.core.citation import Citation, Resolution, resolution_counts, resolve
from pathquery.core.controls import shifted_citations, shuffled_gold
from pathquery.core.evidence import EvidenceScores, Fact, recall_at_k, score_evidence

__all__ = [
    "Citation",
    "EvidenceScores",
    "Fact",
    "Resolution",
    "recall_at_k",
    "resolution_counts",
    "resolve",
    "score_evidence",
    "shifted_citations",
    "shuffled_gold",
]
