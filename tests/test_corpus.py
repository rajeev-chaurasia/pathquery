"""Tests for the corpus build, and for the failure it would not announce.

A title that arrives with two different bodies is the quiet one. Nothing
errors: the supporting facts still name a paragraph that exists and a sentence
index inside it, and the index simply points at different text. Evidence recall
then measures which body the pooling happened to keep.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pathquery.corpus.build import CorpusSpec, build_corpus, load_paragraphs
from pathquery.corpus.hotpot import Paragraph, read_split


def row(
    qid: str, kind: str, titles: list[str], bodies: list[list[str]], support: list[tuple[str, int]]
) -> dict[str, object]:
    return {
        "id": qid,
        "question": f"question {qid}",
        "answer": "yes",
        "type": kind,
        "level": "hard",
        "supporting_facts": {
            "title": [t for t, _ in support],
            "sent_id": [s for _, s in support],
        },
        "context": {"title": titles, "sentences": bodies},
    }


def write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def raw_with_conflict(tmp_path: Path) -> Path:
    """Two questions whose paragraph 'Shared' has different bodies."""
    raw = tmp_path / "raw"
    raw.mkdir()
    first = row(
        "q-first", "bridge", ["Shared", "A"], [["One. ", "Two. "], ["Alpha. "]], [("Shared", 1)]
    )
    second = row(
        "q-second",
        "bridge",
        ["Shared", "B"],
        [["Different. ", "Bodies. ", "Here. "], ["Beta. "]],
        [("Shared", 2)],
    )
    write_parquet(raw / "validation-00000-of-00001.parquet", [first, second])
    return raw


def test_sentence_spans_land_exactly_on_their_sentences() -> None:
    paragraph = Paragraph(title="t", sentences=("Hello world. ", "Second one. "))
    text = paragraph.text
    for index, (start, end) in enumerate(paragraph.spans()):
        assert text[start:end] == paragraph.sentences[index]


def test_joining_reproduces_the_paragraph_without_added_separators() -> None:
    """HotpotQA sentences carry their own spaces; re-joining with more shifts every offset."""
    paragraph = Paragraph(title="t", sentences=("A. ", "B. "))
    assert paragraph.text == "A. B. "


def test_asking_for_a_sentence_that_is_not_there_raises() -> None:
    paragraph = Paragraph(title="t", sentences=("only one. ",))
    try:
        paragraph.span(5)
    except IndexError:
        return
    raise AssertionError("an out of range sentence index was accepted")


def test_a_scored_question_keeps_its_own_paragraph_body(tmp_path: Path) -> None:
    """The quiet corruption: its sentence indices only mean anything against its own copy."""
    raw = raw_with_conflict(tmp_path)
    out = tmp_path / "build"
    build_corpus(raw, out, CorpusSpec(questions=1, seed=2, include_train_paragraphs=False))

    scored = json.loads((out / "questions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    stored = load_paragraphs(out / "paragraphs.jsonl")
    own = {
        p.title: p
        for q in read_split(raw / "validation-00000-of-00001.parquet")
        if q.question_id == scored["id"]
        for p in q.context
    }
    for fact in scored["supporting"]:
        assert stored[fact["document"]].sentences == own[fact["document"]].sentences


def test_every_supporting_sentence_index_is_inside_its_stored_paragraph(tmp_path: Path) -> None:
    raw = raw_with_conflict(tmp_path)
    out = tmp_path / "build"
    build_corpus(raw, out, CorpusSpec(questions=2, seed=2, include_train_paragraphs=False))
    stored = load_paragraphs(out / "paragraphs.jsonl")
    for line in (out / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        question = json.loads(line)
        for fact in question["supporting"]:
            paragraph = stored[fact["document"]]
            assert 0 <= fact["sentence"] < len(paragraph.sentences)


def test_a_conflicting_title_is_counted_not_hidden(tmp_path: Path) -> None:
    raw = raw_with_conflict(tmp_path)
    report = build_corpus(
        raw, tmp_path / "build", CorpusSpec(questions=1, seed=2, include_train_paragraphs=False)
    )
    assert report.title_conflicts >= 1
    assert report.notes


def test_no_gold_paragraph_is_left_out_of_the_corpus(tmp_path: Path) -> None:
    raw = raw_with_conflict(tmp_path)
    report = build_corpus(
        raw, tmp_path / "build", CorpusSpec(questions=2, seed=2, include_train_paragraphs=False)
    )
    assert report.missing_gold_paragraphs == 0


def test_the_same_seed_draws_the_same_questions(tmp_path: Path) -> None:
    raw = raw_with_conflict(tmp_path)
    spec = CorpusSpec(questions=1, seed=9, include_train_paragraphs=False)
    build_corpus(raw, tmp_path / "a", spec)
    build_corpus(raw, tmp_path / "b", spec)
    assert (tmp_path / "a" / "questions.jsonl").read_bytes() == (
        tmp_path / "b" / "questions.jsonl"
    ).read_bytes()


def test_question_kinds_are_counted_separately(tmp_path: Path) -> None:
    """Bridge and comparison are different problems and are never pooled."""
    raw = tmp_path / "raw"
    raw.mkdir()
    write_parquet(
        raw / "validation-00000-of-00001.parquet",
        [
            row("b1", "bridge", ["X"], [["a. "]], [("X", 0)]),
            row("c1", "comparison", ["Y"], [["b. "]], [("Y", 0)]),
        ],
    )
    report = build_corpus(
        raw, tmp_path / "build", CorpusSpec(questions=2, seed=1, include_train_paragraphs=False)
    )
    assert (report.bridge, report.comparison) == (1, 1)
