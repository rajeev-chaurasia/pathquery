"""Command line entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from pathquery.core import Citation, Fact
from pathquery.corpus import CorpusSpec, build_corpus
from pathquery.corpus.hotpot import Paragraph

app = typer.Typer(
    add_completion=False,
    help="Multi-hop evidence retrieval whose citations are checked against the source.",
)


@app.callback()
def main() -> None:
    """Keep subcommands addressable by name.

    Without a callback typer collapses a single-command application into a bare
    one, and the subcommand name then parses as a stray argument.
    """


@app.command("build-corpus")
def build_corpus_command(
    raw: Annotated[Path, typer.Option(help="Where the parquet files were fetched.")] = Path(
        "data/raw"
    ),
    out: Annotated[Path, typer.Option()] = Path("data/build"),
    questions: Annotated[int, typer.Option()] = 1_200,
    seed: Annotated[int, typer.Option()] = 20261103,
    include_train: Annotated[
        bool, typer.Option(help="Pool the training paragraphs in as distractors.")
    ] = True,
) -> None:
    """Pool the paragraphs into a corpus and draw the question set."""
    spec = CorpusSpec(questions=questions, seed=seed, include_train_paragraphs=include_train)
    report = build_corpus(raw, out, spec)
    typer.echo(json.dumps(report.as_dict(), indent=2))


def _keep_run(
    evidence_dir: Path,
    label: str,
    retrieved: dict[str, list[Fact]],
    citations_by_q: dict[str, list[Citation]],
) -> None:
    """Write what was retrieved, not only what it scored.

    A summary cannot be checked against anything. With the evidence beside it,
    every number in the summary can be recomputed from the annotations by
    someone who does not trust the code that produced it.
    """
    import gzip

    runs = evidence_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    with gzip.open(runs / f"{label}.jsonl.gz", "wt", encoding="utf-8") as handle:
        for question_id in sorted(retrieved):
            handle.write(
                json.dumps(
                    {
                        "qid": question_id,
                        "retrieved": [
                            {"document": f.document, "sentence": f.sentence}
                            for f in retrieved[question_id]
                        ],
                        "citations": [
                            {
                                "document": c.document,
                                "start": c.start,
                                "end": c.end,
                                "quoted": c.quoted,
                            }
                            for c in citations_by_q.get(question_id, [])
                        ],
                    }
                )
                + "\n"
            )


def _keep_cited_paragraphs(
    evidence_dir: Path,
    citations: list[Citation],
    paragraphs: dict[str, Paragraph],
) -> int:
    """Commit the source text every citation points at, and nothing else.

    The corpus is half a gigabyte and is rebuilt from a seed, so it is not in
    the repository. The paragraphs that citations actually name are a few
    thousand, and without them the central claim is unverifiable by anyone who
    has not first spent twenty minutes downloading HotpotQA. HotpotQA is
    CC BY-SA, so redistributing this slice is permitted; see ATTRIBUTION.md.
    """
    path = evidence_dir / "cited-paragraphs.jsonl"
    # Merged with whatever is already there, never replacing it. Each run cites
    # a different set, and a file holding only the last one leaves every other
    # run's citations pointing at a document the repository does not contain.
    existing: dict[str, list[str]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            existing[record["title"]] = record["sentences"]

    for title in {citation.document for citation in citations}:
        paragraph = paragraphs.get(title)
        if paragraph is not None:
            existing[title] = list(paragraph.sentences)

    with path.open("w", encoding="utf-8") as handle:
        for title in sorted(existing):
            handle.write(json.dumps({"title": title, "sentences": existing[title]}) + "\n")
    return len(existing)


@app.command("build-index")
def build_index_command(
    build: Annotated[Path, typer.Option()] = Path("data/build"),
) -> None:
    """Index the pooled paragraphs for lexical retrieval."""
    from pathquery.corpus.build import load_paragraphs
    from pathquery.retrieve.lexical import ParagraphIndex

    paragraphs = load_paragraphs(build / "paragraphs.jsonl")
    titles = sorted(paragraphs)
    typer.echo(f"indexing {len(titles)} paragraphs")
    index = ParagraphIndex.build(titles, [paragraphs[t].text for t in titles])
    out = build / "bm25"
    out.mkdir(parents=True, exist_ok=True)
    index.save(out)
    typer.echo(f"saved to {out}")


@app.command("baseline")
def baseline_command(
    build: Annotated[Path, typer.Option()] = Path("data/build"),
    evidence_dir: Annotated[Path, typer.Option("--evidence")] = Path("evidence"),
    paragraphs_k: Annotated[int, typer.Option(help="Paragraphs retrieved per question.")] = 10,
) -> None:
    """Single hop: ask the whole question once, and score what comes back.

    This is the comparison every multi-hop number is published beside, so it is
    built first and kept.
    """
    from pathquery.core import Resolution, resolution_counts, score_evidence, shuffled_gold
    from pathquery.corpus.build import load_paragraphs, load_questions
    from pathquery.retrieve.lexical import ParagraphIndex
    from pathquery.retrieve.sentences import select_evidence

    paragraphs = load_paragraphs(build / "paragraphs.jsonl")
    corpus_text = {title: paragraph.text for title, paragraph in paragraphs.items()}
    index = ParagraphIndex.load(build / "bm25")
    questions = load_questions(build / "questions.jsonl")

    ranked = index.search([q.question for q in questions], paragraphs_k)
    retrieved: dict[str, list[Fact]] = {}
    gold: dict[str, set[Fact]] = {}
    citations: list[Citation] = []
    per_question: dict[str, list[Citation]] = {}
    for question, titles in zip(questions, ranked, strict=True):
        evidence = select_evidence(question.question, titles, paragraphs)
        retrieved[question.question_id] = [e.fact for e in evidence]
        mine = [e.citation for e in evidence[:10]]
        per_question[question.question_id] = mine
        citations.extend(mine)
        gold[question.question_id] = question.gold

    by_kind = {}
    for kind in ("bridge", "comparison"):
        ids = {q.question_id for q in questions if q.kind == kind}
        by_kind[kind] = score_evidence(
            {q: retrieved[q] for q in ids}, {q: gold[q] for q in ids}
        ).as_dict()

    counts = resolution_counts(citations, corpus_text)
    resolved = counts.get(Resolution.RESOLVED, 0)
    results = {
        "label": "single-hop",
        "paragraphs_k": paragraphs_k,
        "all": score_evidence(retrieved, gold).as_dict(),
        "bridge": by_kind["bridge"],
        "comparison": by_kind["comparison"],
        "control_shuffled_gold": score_evidence(retrieved, shuffled_gold(gold, seed=7)).as_dict(),
        "citations": {k.value: v for k, v in counts.items()},
        "citations_resolved_rate": round(resolved / max(len(citations), 1), 5),
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _keep_run(evidence_dir, "single-hop", retrieved, per_question)
    _keep_cited_paragraphs(evidence_dir, citations, paragraphs)
    (evidence_dir / "single-hop.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(json.dumps(results, indent=2))


@app.command("multihop")
def multihop_command(
    build: Annotated[Path, typer.Option()] = Path("data/build"),
    evidence_dir: Annotated[Path, typer.Option("--evidence")] = Path("evidence"),
    paragraphs_k: Annotated[int, typer.Option()] = 10,
    hop_width: Annotated[int, typer.Option(help="Follow-up queries per question.")] = 3,
) -> None:
    """Follow what the first hop found, then score and check what comes back."""
    from pathquery.core import Resolution, resolution_counts, score_evidence, shuffled_gold
    from pathquery.core.controls import shifted_citations
    from pathquery.corpus.build import load_paragraphs, load_questions
    from pathquery.hops import answer_question
    from pathquery.retrieve.lexical import ParagraphIndex

    paragraphs = load_paragraphs(build / "paragraphs.jsonl")
    corpus_text = {title: paragraph.text for title, paragraph in paragraphs.items()}
    index = ParagraphIndex.load(build / "bm25")
    questions = load_questions(build / "questions.jsonl")

    retrieved: dict[str, list[Fact]] = {}
    gold: dict[str, set[Fact]] = {}
    citations: list[Citation] = []
    per_question: dict[str, list[Citation]] = {}
    uncertain = 0
    for question in questions:
        result = answer_question(
            question.question,
            index,
            paragraphs,
            paragraphs_k=paragraphs_k,
            hop_width=hop_width,
        )
        retrieved[question.question_id] = result.facts
        gold[question.question_id] = question.gold
        per_question[question.question_id] = result.citations
        citations.extend(result.citations)
        uncertain += 1 if result.uncertain else 0

    by_kind = {}
    for kind in ("bridge", "comparison"):
        ids = {q.question_id for q in questions if q.kind == kind}
        by_kind[kind] = score_evidence(
            {q: retrieved[q] for q in ids}, {q: gold[q] for q in ids}
        ).as_dict()

    counts = resolution_counts(citations, corpus_text)
    shifted = resolution_counts(shifted_citations(citations, 3), corpus_text)
    resolved = counts.get(Resolution.RESOLVED, 0)
    results = {
        "label": "multi-hop",
        "paragraphs_k": paragraphs_k,
        "hop_width": hop_width,
        "all": score_evidence(retrieved, gold).as_dict(),
        "bridge": by_kind["bridge"],
        "comparison": by_kind["comparison"],
        "control_shuffled_gold": score_evidence(retrieved, shuffled_gold(gold, seed=7)).as_dict(),
        "citations": {k.value: v for k, v in counts.items()},
        "citations_resolved_rate": round(resolved / max(len(citations), 1), 5),
        "control_shifted_offsets": {k.value: v for k, v in shifted.items()},
        "control_shifted_resolved_rate": round(
            shifted.get(Resolution.RESOLVED, 0) / max(len(citations), 1), 5
        ),
        "questions_flagged_uncertain": uncertain,
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _keep_run(evidence_dir, "multi-hop", retrieved, per_question)
    _keep_cited_paragraphs(evidence_dir, citations, paragraphs)
    (evidence_dir / "multi-hop.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
