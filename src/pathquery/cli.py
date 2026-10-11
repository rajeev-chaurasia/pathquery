"""Command line entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from pathquery.core import Citation
from pathquery.corpus import CorpusSpec, build_corpus

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
    from pathquery.core import Fact, Resolution, resolution_counts, score_evidence, shuffled_gold
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
    for question, titles in zip(questions, ranked, strict=True):
        evidence = select_evidence(question.question, titles, paragraphs)
        retrieved[question.question_id] = [e.fact for e in evidence]
        citations.extend(e.citation for e in evidence[:10])
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
    from pathquery.core import Fact, Resolution, resolution_counts, score_evidence, shuffled_gold
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
    (evidence_dir / "multi-hop.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
