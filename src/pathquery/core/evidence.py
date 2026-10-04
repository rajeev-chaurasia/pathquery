"""Scoring retrieved evidence against the annotations.

HotpotQA marks which sentences support an answer, by paragraph title and
sentence index. Those pairs are the gold standard and this module does nothing
clever with them: it counts how many were retrieved and at what depth.

Recall rather than precision is the headline, because a multi-hop question has
two or three supporting sentences scattered across paragraphs and the thing
that matters is whether the second hop reached the one the first could not. A
system that returns all of them inside ten results has done the job even if it
returned other things too, and the redundancy filtering is what the precision
figure is for.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict


class Fact(BaseModel):
    """One supporting sentence, named the way the annotations name it."""

    model_config = ConfigDict(frozen=True)

    document: str
    sentence: int

    def __lt__(self, other: Fact) -> bool:
        return (self.document, self.sentence) < (other.document, other.sentence)


def recall_at_k(retrieved: list[Fact], gold: set[Fact], k: int) -> float:
    if not gold:
        raise ValueError("evidence recall is undefined for a question with no supporting facts")
    return len(set(retrieved[:k]) & gold) / len(gold)


def precision_at_k(retrieved: list[Fact], gold: set[Fact], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & gold) / len(top)


def fully_supported(retrieved: list[Fact], gold: set[Fact], k: int) -> bool:
    """Whether every supporting sentence was found, not merely some of them.

    The strictest reading and the one that matters for a multi-hop question:
    finding one of two hops is not a partial answer, it is a missing one.
    """
    return gold <= set(retrieved[:k])


@dataclass(frozen=True)
class EvidenceScores:
    questions: int
    recall_at_2: float
    recall_at_10: float
    precision_at_10: float
    fully_supported_at_10: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "questions": self.questions,
            "recall_at_2": round(self.recall_at_2, 5),
            "recall_at_10": round(self.recall_at_10, 5),
            "precision_at_10": round(self.precision_at_10, 5),
            "fully_supported_at_10": round(self.fully_supported_at_10, 5),
        }


def score_evidence(retrieved: dict[str, list[Fact]], gold: dict[str, set[Fact]]) -> EvidenceScores:
    """Average over the questions that have annotations, one vote each."""
    scorable = sorted(q for q in retrieved if gold.get(q))
    if not scorable:
        return EvidenceScores(0, 0.0, 0.0, 0.0, 0.0)

    recall_2 = recall_10 = precision_10 = supported = 0.0
    for question in scorable:
        found, wanted = retrieved[question], gold[question]
        recall_2 += recall_at_k(found, wanted, 2)
        recall_10 += recall_at_k(found, wanted, 10)
        precision_10 += precision_at_k(found, wanted, 10)
        supported += 1.0 if fully_supported(found, wanted, 10) else 0.0

    count = len(scorable)
    return EvidenceScores(
        questions=count,
        recall_at_2=recall_2 / count,
        recall_at_10=recall_10 / count,
        precision_at_10=precision_10 / count,
        fully_supported_at_10=supported / count,
    )
