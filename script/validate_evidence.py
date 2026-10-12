"""Recompute every published number from the committed records.

This one is fully self-contained, which the other repositories here are not.
HotpotQA is CC BY-SA, so the annotations travel with the results, and the
paragraphs that citations point at travel with them too. A reader who trusts
none of the code can check every figure without downloading anything.

Three things are checked. The evidence scores are recalculated from the
retrieved facts and the annotations. Every citation is resolved against the
source text it names. And the controls are checked for having failed: shifted
offsets must stop resolving, and deranged annotations must stop finding
support.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any

from pathquery.core import (
    Citation,
    Fact,
    Resolution,
    resolution_counts,
    score_evidence,
    shifted_citations,
    shuffled_gold,
)
from pathquery.corpus.hotpot import Paragraph

EVIDENCE = Path("evidence")

# Fixed in PLAN.md before anything was measured.
SHUFFLED_CEILING = 0.02
SHIFTED_CEILING = 0.01
RESOLUTION_FLOOR = 1.0


def load_questions() -> dict[str, dict[str, Any]]:
    path = EVIDENCE / "questions.jsonl"
    return {
        record["id"]: record
        for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }


def load_corpus() -> dict[str, str]:
    path = EVIDENCE / "cited-paragraphs.jsonl"
    corpus: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        corpus[record["title"]] = Paragraph(
            title=record["title"], sentences=tuple(record["sentences"])
        ).text
    return corpus


def load_run(label: str) -> tuple[dict[str, list[Fact]], list[Citation]]:
    retrieved: dict[str, list[Fact]] = {}
    citations: list[Citation] = []
    with gzip.open(EVIDENCE / "runs" / f"{label}.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            retrieved[record["qid"]] = [
                Fact(document=f["document"], sentence=f["sentence"]) for f in record["retrieved"]
            ]
            citations.extend(Citation(**c) for c in record["citations"])
    return retrieved, citations


def compare(published: dict[str, Any], recomputed: dict[str, Any], where: str) -> list[str]:
    problems = []
    for metric, value in published.items():
        if recomputed.get(metric) != value:
            problems.append(
                f"{where}.{metric} published as {value}, recomputed as {recomputed.get(metric)}"
            )
    return problems


def check_run(
    label: str, questions: dict[str, dict[str, Any]], corpus: dict[str, str]
) -> list[str]:
    summary = json.loads((EVIDENCE / f"{label}.json").read_text(encoding="utf-8"))
    retrieved, citations = load_run(label)
    gold = {
        qid: {Fact(document=f["document"], sentence=f["sentence"]) for f in record["supporting"]}
        for qid, record in questions.items()
    }
    problems: list[str] = []

    for key, subset in (
        ("all", set(gold)),
        ("bridge", {q for q, r in questions.items() if r["type"] == "bridge"}),
        ("comparison", {q for q, r in questions.items() if r["type"] == "comparison"}),
    ):
        if key not in summary:
            continue
        measured = score_evidence(
            {q: retrieved[q] for q in subset if q in retrieved},
            {q: gold[q] for q in subset},
        ).as_dict()
        problems += compare(summary[key], measured, f"{label}.{key}")

    shuffled = score_evidence(retrieved, shuffled_gold(gold, seed=7)).as_dict()
    problems += compare(summary["control_shuffled_gold"], shuffled, f"{label}.shuffled")
    if shuffled["recall_at_10"] > SHUFFLED_CEILING:
        problems.append(
            f"{label}: deranged annotations still found {shuffled['recall_at_10']} recall, "
            f"above {SHUFFLED_CEILING}"
        )

    counts = resolution_counts(citations, corpus)
    rate = counts.get(Resolution.RESOLVED, 0) / max(len(citations), 1)
    if rate < RESOLUTION_FLOOR:
        unresolved = {k.value: v for k, v in counts.items() if k is not Resolution.RESOLVED}
        problems.append(f"{label}: only {rate} of citations resolved, unresolved {unresolved}")

    shifted = resolution_counts(shifted_citations(citations, 3), corpus)
    shifted_rate = shifted.get(Resolution.RESOLVED, 0) / max(len(citations), 1)
    if shifted_rate > SHIFTED_CEILING:
        problems.append(
            f"{label}: {shifted_rate} of citations still resolved after their offsets were "
            f"moved, above {SHIFTED_CEILING}: the checker is not reading the offsets"
        )
    return problems


def check_comparison() -> list[str]:
    """Multi-hop is only ever published beside the single-hop it has to beat."""
    for label in ("single-hop", "multi-hop"):
        if not (EVIDENCE / f"{label}.json").exists():
            return [f"{label} is missing, so there is nothing to compare"]
    return []


def main() -> int:
    if not EVIDENCE.exists():
        print("no evidence to validate", file=sys.stderr)
        return 1

    questions = load_questions()
    corpus = load_corpus()
    problems = check_comparison()
    labels = [p.stem for p in sorted(EVIDENCE.glob("*.json"))]
    for label in labels:
        problems += check_run(label, questions, corpus)

    if problems:
        print("\n".join(problems), file=sys.stderr)
        print(f"\n{len(problems)} evidence problem(s)", file=sys.stderr)
        return 1
    print(f"evidence ok over {len(labels)} run(s), {len(questions)} questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
