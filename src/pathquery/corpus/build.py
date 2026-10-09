"""Pooling HotpotQA's paragraphs into one corpus, and cutting the question set.

Every question ships ten paragraphs, two of which hold its supporting
sentences. Pooling them across the dataset gives a corpus where the right
answer for one question is a distractor for thousands of others, which is the
property that makes retrieval over it non-trivial.

Two things are checked rather than assumed. Every paragraph a scored question
depends on must be in the corpus, or evidence recall would be measuring which
paragraphs the pooling happened to include. And a title that appears with two
different bodies would make a citation ambiguous, so those are counted and
reported rather than silently resolved to whichever arrived last.

The train split is read in batches. It is 316 MB of parquet and expands to
several gigabytes as Python objects, which is more than a laptop wants to hold
to produce a few hundred thousand distinct paragraphs.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from pathquery.core import Fact
from pathquery.corpus.hotpot import Paragraph, read_split

BATCH_ROWS = 2_000


@dataclass(frozen=True)
class CorpusSpec:
    questions: int = 1_200
    seed: int = 20261103
    include_train_paragraphs: bool = True


@dataclass
class BuildReport:
    paragraphs: int = 0
    sentences: int = 0
    questions: int = 0
    bridge: int = 0
    comparison: int = 0
    title_conflicts: int = 0
    scored_overrides: int = 0
    missing_gold_paragraphs: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "paragraphs": self.paragraphs,
            "sentences": self.sentences,
            "questions": self.questions,
            "bridge": self.bridge,
            "comparison": self.comparison,
            "title_conflicts": self.title_conflicts,
            "scored_overrides": self.scored_overrides,
            "missing_gold_paragraphs": self.missing_gold_paragraphs,
            "notes": self.notes,
        }


def _add(
    pool: dict[str, tuple[str, ...]],
    title: str,
    sentences: tuple[str, ...],
    *,
    override: bool = False,
) -> bool:
    """Record a paragraph, reporting whether the title already meant something else.

    `override` is for the paragraphs the scored questions carry with them. A
    title can arrive with two different bodies, and when that happens the
    question's own copy has to win: its supporting facts are sentence indices
    into that copy, and resolving them against a different body points the
    annotation at different text without anything failing.
    """
    existing = pool.get(title)
    if existing is None:
        pool[title] = sentences
        return False
    if existing == sentences:
        return False
    if override:
        pool[title] = sentences
    return True


def build_corpus(raw: Path, out: Path, spec: CorpusSpec) -> BuildReport:
    out.mkdir(parents=True, exist_ok=True)
    report = BuildReport()
    pool: dict[str, tuple[str, ...]] = {}

    questions = list(read_split(raw / "validation-00000-of-00001.parquet"))
    rng = random.Random(spec.seed)
    shuffled = questions[:]
    rng.shuffle(shuffled)
    chosen = sorted(shuffled[: spec.questions], key=lambda q: q.question_id)
    scored = {q.question_id for q in chosen}

    for question in questions:
        if question.question_id in scored:
            continue
        for paragraph in question.context:
            if _add(pool, paragraph.title, paragraph.sentences):
                report.title_conflicts += 1

    if spec.include_train_paragraphs:
        for shard in sorted(raw.glob("train-*.parquet")):
            reader = pq.ParquetFile(shard)
            for batch in reader.iter_batches(batch_size=BATCH_ROWS, columns=["context"]):
                for row in batch.to_pylist():
                    context = row["context"]
                    for title, sentences in zip(
                        context["title"], context["sentences"], strict=True
                    ):
                        if _add(pool, title, tuple(sentences)):
                            report.title_conflicts += 1

    # Last, so they win any title that already meant something else.
    for question in chosen:
        for paragraph in question.context:
            if _add(pool, paragraph.title, paragraph.sentences, override=True):
                report.title_conflicts += 1
                report.scored_overrides += 1

    # The guarantee recall depends on. A gold paragraph outside the corpus
    # would make the metric a statement about the pooling.
    for question in chosen:
        for fact in question.supporting:
            if fact.document not in pool:
                report.missing_gold_paragraphs += 1

    with (out / "paragraphs.jsonl").open("w", encoding="utf-8") as handle:
        for title in sorted(pool):
            sentences = pool[title]
            handle.write(json.dumps({"title": title, "sentences": list(sentences)}) + "\n")
            report.sentences += len(sentences)
    report.paragraphs = len(pool)

    with (out / "questions.jsonl").open("w", encoding="utf-8") as handle:
        for question in chosen:
            handle.write(
                json.dumps(
                    {
                        "id": question.question_id,
                        "question": question.question,
                        "answer": question.answer,
                        "type": question.kind,
                        "level": question.level,
                        "supporting": [
                            {"document": f.document, "sentence": f.sentence}
                            for f in question.supporting
                        ],
                    }
                )
                + "\n"
            )
    report.questions = len(chosen)
    report.bridge = sum(1 for q in chosen if q.kind == "bridge")
    report.comparison = sum(1 for q in chosen if q.kind == "comparison")
    if report.title_conflicts:
        report.notes.append(
            f"{report.title_conflicts} paragraphs arrived under a title that already meant "
            f"something else; the first body was kept except for {report.scored_overrides} "
            "where a scored question carried its own"
        )

    (out / "manifest.json").write_text(
        json.dumps({"spec": spec.__dict__, "report": report.as_dict()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


class ScoredQuestion(BaseModel):
    """A question as the build wrote it, read back with its types intact."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    question: str
    answer: str
    kind: str
    level: str
    supporting: tuple[Fact, ...]

    @property
    def gold(self) -> set[Fact]:
        return set(self.supporting)


def load_questions(path: Path) -> list[ScoredQuestion]:
    questions: list[ScoredQuestion] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            questions.append(
                ScoredQuestion(
                    question_id=record["id"],
                    question=record["question"],
                    answer=record["answer"],
                    kind=record["type"],
                    level=record["level"],
                    supporting=tuple(
                        Fact(document=f["document"], sentence=f["sentence"])
                        for f in record["supporting"]
                    ),
                )
            )
    return questions


def load_paragraphs(path: Path) -> dict[str, Paragraph]:
    paragraphs: dict[str, Paragraph] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            paragraphs[record["title"]] = Paragraph(
                title=record["title"], sentences=tuple(record["sentences"])
            )
    return paragraphs
