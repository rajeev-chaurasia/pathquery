"""Command line entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

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


if __name__ == "__main__":
    app()
