"""What a citation is, and what it means for one to be true.

A citation here is not a reference to a document. It is a document and a
half-open character range, together with the text that range is claimed to
contain. Checking it is an exact string comparison: either the corpus at that
range is the quoted text character for character, or the citation is wrong.

The range counts characters, not bytes. HotpotQA is full of accented names, and
in UTF-8 those take more than one byte each, so the two counts disagree after
the first one in a paragraph. Characters are what a reader of the text sees.

That strictness is the point. A fuzzy or normalised comparison would accept a
span that is off by a few characters, and an off-by-few span is how a quote
ends up attributed to the sentence next to the one it came from. Whitespace is
not normalised for the same reason.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Resolution(StrEnum):
    """Why a citation failed, kept apart so each cause can be counted.

    A single failure rate hides which thing broke. A missing document means the
    retriever cited something outside the corpus; a range past the end means
    the offsets were computed against a different version of the text; a text
    mismatch means they were computed against the right text and are still
    wrong, which is the off-by-one case.
    """

    RESOLVED = "resolved"
    MISSING_DOCUMENT = "missing_document"
    OUT_OF_RANGE = "out_of_range"
    TEXT_MISMATCH = "text_mismatch"


class Citation(BaseModel):
    """A claim that `document[start:end]` reads exactly `quoted`."""

    model_config = ConfigDict(frozen=True)

    document: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    quoted: str

    @property
    def length(self) -> int:
        return self.end - self.start


def resolve(citation: Citation, corpus: Mapping[str, str]) -> Resolution:
    """Check a citation against the text it points at.

    An empty or reversed range is treated as out of range rather than as a
    vacuous success. A zero-length span quoting the empty string would satisfy
    a naive comparison for every document in the corpus, which is exactly the
    sort of citation that should never pass.
    """
    text = corpus.get(citation.document)
    if text is None:
        return Resolution.MISSING_DOCUMENT
    if citation.end <= citation.start or citation.end > len(text):
        return Resolution.OUT_OF_RANGE
    if text[citation.start : citation.end] != citation.quoted:
        return Resolution.TEXT_MISMATCH
    return Resolution.RESOLVED


def resolution_counts(
    citations: list[Citation], corpus: Mapping[str, str]
) -> dict[Resolution, int]:
    counts: dict[Resolution, int] = {}
    for citation in citations:
        outcome = resolve(citation, corpus)
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts
