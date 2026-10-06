"""Reading HotpotQA, and the one construction the whole claim rests on.

A paragraph is stored as its sentences and as the single string they join into.
Offsets are derived from that join rather than recorded separately, so a
sentence's span is correct by construction: there is no second copy of the text
for the offsets to drift away from.

HotpotQA's sentences carry their own leading spaces, so joining with no
separator reproduces the paragraph as the annotators saw it. Stripping or
re-joining with spaces would shift every offset after the first sentence, which
is exactly the failure the citation check exists to catch, and it would be
undetectable if the corpus and the offsets came from the same mistake.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from pathquery.core import Fact


class Paragraph(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return "".join(self.sentences)

    def span(self, sentence: int) -> tuple[int, int]:
        """Character range of one sentence inside the paragraph text."""
        if not 0 <= sentence < len(self.sentences):
            raise IndexError(f"{self.title} has {len(self.sentences)} sentences, asked {sentence}")
        start = sum(len(s) for s in self.sentences[:sentence])
        return start, start + len(self.sentences[sentence])

    def spans(self) -> list[tuple[int, int]]:
        offsets: list[tuple[int, int]] = []
        position = 0
        for sentence in self.sentences:
            offsets.append((position, position + len(sentence)))
            position += len(sentence)
        return offsets


class Question(BaseModel):
    model_config = ConfigDict(frozen=True)

    question_id: str
    question: str
    answer: str
    kind: str
    level: str
    supporting: tuple[Fact, ...]
    context: tuple[Paragraph, ...]

    @property
    def gold(self) -> set[Fact]:
        return set(self.supporting)


def read_split(path: Path) -> Iterator[Question]:
    table = pq.read_table(path)
    for row in table.to_pylist():
        facts = row["supporting_facts"]
        context = row["context"]
        yield Question(
            question_id=row["id"],
            question=row["question"],
            answer=row["answer"],
            kind=row["type"],
            level=row["level"],
            supporting=tuple(
                Fact(document=title, sentence=int(sentence))
                for title, sentence in zip(facts["title"], facts["sent_id"], strict=True)
            ),
            context=tuple(
                Paragraph(title=title, sentences=tuple(sentences))
                for title, sentences in zip(context["title"], context["sentences"], strict=True)
            ),
        )
