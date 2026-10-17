"""Tests for the citation rule, written against spans counted by hand.

This is the claim. A citation is a character range and the check is an exact
string comparison, so the way it fails matters as much as the way it passes: an off-by-one
span is how a quote gets attributed to the sentence beside the one it came
from, and that must not be tolerated anywhere.
"""

from __future__ import annotations

import pytest

from pathquery.core import Citation, Resolution, resolution_counts, resolve
from pathquery.core.controls import shifted_citations

# Offsets counted by hand:      0123456789...
CORPUS = {"ada": "Ada Lovelace wrote the first algorithm.", "empty": ""}


def cite(start: int, end: int, quoted: str, document: str = "ada") -> Citation:
    return Citation(document=document, start=start, end=end, quoted=quoted)


def test_an_exact_span_resolves() -> None:
    """Characters 0 to 12 are 'Ada Lovelace'."""
    assert resolve(cite(0, 12, "Ada Lovelace"), CORPUS) is Resolution.RESOLVED


def test_a_span_one_character_short_is_rejected() -> None:
    assert resolve(cite(0, 11, "Ada Lovelace"), CORPUS) is Resolution.TEXT_MISMATCH


def test_a_span_shifted_by_one_is_rejected() -> None:
    """The failure that matters: nearly right is wrong."""
    assert resolve(cite(1, 13, "Ada Lovelace"), CORPUS) is Resolution.TEXT_MISMATCH


def test_offsets_count_characters_not_bytes() -> None:
    """'é' is one character and two UTF-8 bytes, so the counts part after it.

    A producer counting bytes would place every span after the first accent in
    a paragraph too far right, and this pins which of the two is meant.
    """
    corpus = {"b": "Beyoncé sang."}
    assert resolve(cite(8, 12, "sang", document="b"), corpus) is Resolution.RESOLVED
    assert resolve(cite(9, 13, "sang", document="b"), corpus) is Resolution.TEXT_MISMATCH


def test_whitespace_is_not_normalised_away() -> None:
    assert resolve(cite(0, 12, "Ada  Lovelace"), CORPUS) is Resolution.TEXT_MISMATCH
    assert resolve(cite(0, 12, " Ada Lovelace"), CORPUS) is Resolution.TEXT_MISMATCH


def test_case_is_not_normalised_away() -> None:
    assert resolve(cite(0, 12, "ada lovelace"), CORPUS) is Resolution.TEXT_MISMATCH


def test_a_document_outside_the_corpus_is_named_as_such() -> None:
    """Distinct from a bad range: the retriever cited something that is not there."""
    assert resolve(cite(0, 3, "Ada", document="nobody"), CORPUS) is Resolution.MISSING_DOCUMENT


def test_a_range_past_the_end_is_rejected() -> None:
    assert resolve(cite(0, 500, "whatever"), CORPUS) is Resolution.OUT_OF_RANGE


def test_an_empty_span_cannot_quote_the_empty_string() -> None:
    """Otherwise a vacuous citation would pass against every document."""
    assert resolve(cite(5, 5, ""), CORPUS) is Resolution.OUT_OF_RANGE


def test_a_reversed_span_is_rejected() -> None:
    assert resolve(cite(9, 4, "Love"), CORPUS) is Resolution.OUT_OF_RANGE


def test_a_span_at_the_very_end_resolves() -> None:
    text = CORPUS["ada"]
    assert resolve(cite(len(text) - 10, len(text), text[-10:]), CORPUS) is Resolution.RESOLVED


def test_the_shift_control_breaks_every_citation() -> None:
    """The control the plan requires: a checker that passes these reads nothing."""
    good = [cite(0, 12, "Ada Lovelace"), cite(13, 18, "wrote")]
    assert all(resolve(c, CORPUS) is Resolution.RESOLVED for c in good)
    assert all(resolve(c, CORPUS) is not Resolution.RESOLVED for c in shifted_citations(good, 3))


def test_a_shift_of_zero_is_refused_as_a_control() -> None:
    with pytest.raises(ValueError, match="not a control"):
        shifted_citations([cite(0, 12, "Ada Lovelace")], offset=0)


def test_counts_are_reported_by_cause() -> None:
    """One rate would hide which thing broke."""
    counts = resolution_counts(
        [cite(0, 12, "Ada Lovelace"), cite(1, 13, "Ada Lovelace"), cite(0, 3, "Ada", "nobody")],
        CORPUS,
    )
    assert counts == {
        Resolution.RESOLVED: 1,
        Resolution.TEXT_MISMATCH: 1,
        Resolution.MISSING_DOCUMENT: 1,
    }
