"""Tests for the evidence metrics, against counts done by hand."""

from __future__ import annotations

import pytest

from pathquery.core import Fact, recall_at_k, score_evidence
from pathquery.core.controls import shuffled_gold
from pathquery.core.evidence import fully_supported, precision_at_k


def fact(document: str, sentence: int) -> Fact:
    return Fact(document=document, sentence=sentence)


GOLD = {fact("ada", 0), fact("babbage", 2)}


def test_recall_counts_gold_facts_inside_the_cut() -> None:
    found = [fact("ada", 0), fact("other", 1)]
    assert recall_at_k(found, GOLD, 10) == 0.5


def test_recall_respects_the_cut() -> None:
    found = [fact("other", 1), fact("other", 2), fact("ada", 0)]
    assert recall_at_k(found, GOLD, 2) == 0.0
    assert recall_at_k(found, GOLD, 3) == 0.5


def test_recall_without_annotations_is_refused() -> None:
    """Returning zero would punish a system for a gap in the annotations."""
    with pytest.raises(ValueError):
        recall_at_k([fact("ada", 0)], set(), 10)


def test_precision_counts_what_was_returned() -> None:
    found = [fact("ada", 0), fact("other", 1), fact("other", 2), fact("babbage", 2)]
    assert precision_at_k(found, GOLD, 4) == 0.5


def test_precision_of_nothing_is_zero_not_undefined() -> None:
    assert precision_at_k([], GOLD, 10) == 0.0


def test_full_support_needs_every_hop_not_some() -> None:
    """Finding one of two hops is a missing answer, not a partial one."""
    assert fully_supported([fact("ada", 0)], GOLD, 10) is False
    assert fully_supported([fact("ada", 0), fact("babbage", 2)], GOLD, 10) is True


def test_full_support_respects_the_cut() -> None:
    found = [fact("x", 1)] * 9 + [fact("ada", 0), fact("babbage", 2)]
    assert fully_supported(found, GOLD, 10) is False


def test_scoring_averages_one_vote_per_question() -> None:
    """One perfect question and one empty: a half, not weighted by fact count."""
    retrieved = {"q1": [fact("ada", 0)], "q2": [fact("nothing", 9)]}
    gold = {"q1": {fact("ada", 0)}, "q2": {fact("b", 1), fact("c", 2), fact("d", 3)}}
    assert score_evidence(retrieved, gold).recall_at_10 == pytest.approx(0.5)


def test_questions_without_annotations_are_excluded() -> None:
    retrieved = {"q1": [fact("ada", 0)], "q2": [fact("x", 1)]}
    gold: dict[str, set[Fact]] = {"q1": {fact("ada", 0)}, "q2": set()}
    assert score_evidence(retrieved, gold).questions == 1


def test_scoring_nothing_returns_zeros_rather_than_failing() -> None:
    assert score_evidence({}, {}).questions == 0


def test_the_shuffle_gives_every_question_someone_elses_facts() -> None:
    gold = {f"q{i}": {fact(f"d{i}", 0)} for i in range(6)}
    shuffled = shuffled_gold(gold, seed=3)
    assert set(shuffled) == set(gold)
    assert all(shuffled[q] != gold[q] for q in gold)


def test_the_shuffle_keeps_the_shape_of_the_annotations() -> None:
    gold = {"q0": {fact("a", 0)}, "q1": {fact("b", 1), fact("c", 2)}, "q2": {fact("d", 3)}}
    shuffled = shuffled_gold(gold, seed=3)
    assert sorted(len(v) for v in shuffled.values()) == sorted(len(v) for v in gold.values())


def test_the_shuffle_is_reproducible() -> None:
    gold = {f"q{i}": {fact(f"d{i}", 0)} for i in range(6)}
    assert shuffled_gold(gold, seed=3) == shuffled_gold(gold, seed=3)


def test_one_question_cannot_be_deranged() -> None:
    with pytest.raises(ValueError):
        shuffled_gold({"q0": {fact("a", 0)}}, seed=1)


def test_shuffled_annotations_collapse_recall() -> None:
    """The control the plan requires, on the same scoring code."""
    retrieved = {f"q{i}": [fact(f"d{i}", 0)] for i in range(8)}
    gold = {f"q{i}": {fact(f"d{i}", 0)} for i in range(8)}
    assert score_evidence(retrieved, gold).recall_at_10 == 1.0
    assert score_evidence(retrieved, shuffled_gold(gold, seed=3)).recall_at_10 == 0.0
