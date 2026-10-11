"""The second hop, and what makes it a hop rather than a second guess.

A bridge question names one entity and hides the other inside it. "The director
of the 1994 film Ed Wood also directed what?" is answerable only if something
first establishes who that director is. Asking the whole question again, however
many times, cannot find the second paragraph, because the term that would find
it does not appear in the question.

So the second query is built from what the first hop returned: the title of a
paragraph it found, and the sentence inside it that best matched. That is the
definition the plan commits to, and a system issuing two retrievals from the
original question alone is scored as single hop however many requests it makes.

Comparison questions are a different shape and this helps them less. They name
both entities up front, so the first hop can already reach both paragraphs. The
two kinds are reported separately for exactly that reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pathquery.core import Citation, Fact
from pathquery.corpus.hotpot import Paragraph
from pathquery.retrieve.lexical import ParagraphIndex
from pathquery.retrieve.sentences import Evidence, select_evidence

STOP_TITLE_WORDS = {"the", "a", "an", "of", "and", "in", "on", "for", "to"}


@dataclass(frozen=True)
class MultiHopResult:
    evidence: list[Evidence]
    queries: list[str]
    uncertain: bool

    @property
    def facts(self) -> list[Fact]:
        return [item.fact for item in self.evidence]

    @property
    def citations(self) -> list[Citation]:
        return [item.citation for item in self.evidence]


def expand_queries(question: str, first_hop: Sequence[Evidence], width: int) -> list[str]:
    """Build one follow-up query per promising paragraph from the first hop.

    The paragraph title carries the entity the question was about; the sentence
    carries whatever it said next, which is where the bridge to the second
    paragraph lives. Both go in alongside the original question, because
    dropping the question would retrieve about the entity in general rather
    than about what was asked of it.

    Titles already present in the question are skipped. Re-asking about an
    entity the question named is the second guess this is meant not to be.
    """
    asked = {word.lower().strip(".,?'\"") for word in question.split()}
    queries: list[str] = []
    seen: set[str] = set()
    for item in first_hop:
        title = item.fact.document
        if title in seen:
            continue
        seen.add(title)
        informative = {
            word.lower() for word in title.split() if word.lower() not in STOP_TITLE_WORDS
        }
        if informative and informative <= asked:
            continue
        bridges = _bridge_entities(item.citation.quoted, asked)
        if not bridges:
            continue
        # The entity, not the sentence it was found in. BM25 scores a bag of
        # terms, so pasting in a thirty word sentence buries the question under
        # its own evidence and retrieves about the first paragraph again.
        queries.append(f"{question} {' '.join(bridges)}")
        if len(queries) >= width:
            break
    return queries


def _bridge_entities(sentence: str, asked: set[str], limit: int = 3) -> list[str]:
    """Capitalised runs the question did not already mention.

    A crude proxy for a named entity and deliberately so: the claim is about
    whether following what the first hop found helps, not about the quality of
    the extractor. Anything the question already said is dropped, because
    re-asking about it is the second guess this is meant not to be.
    """
    words = sentence.split()
    runs: list[str] = []
    current: list[str] = []
    for position, raw in enumerate(words):
        word = raw.strip(".,;:()\"'")
        # A capital at the start of a sentence says nothing, so the first word
        # only counts when it keeps going.
        if word and word[0].isupper() and not (position == 0 and len(current) == 0):
            current.append(word)
            continue
        if current:
            runs.append(" ".join(current))
            current = []
    if current:
        runs.append(" ".join(current))

    chosen: list[str] = []
    for run in runs:
        terms = {t.lower() for t in run.split()}
        if terms <= asked or not terms:
            continue
        chosen.append(run)
        if len(chosen) >= limit:
            break
    return chosen


def _redundant(candidate: Evidence, kept: Sequence[Evidence], threshold: float) -> bool:
    """Whether a sentence says what one already kept says.

    Token overlap rather than anything cleverer. Two hops over an entity often
    return the same sentence by two routes, and a result list holding the same
    claim three times has spent places that the second paragraph needed.
    """
    words = set(candidate.citation.quoted.lower().split())
    if not words:
        return True
    for existing in kept:
        other = set(existing.citation.quoted.lower().split())
        if other and len(words & other) / len(words | other) >= threshold:
            return True
    return False


def answer_question(
    question: str,
    index: ParagraphIndex,
    paragraphs: Mapping[str, Paragraph],
    *,
    paragraphs_k: int = 10,
    hop_width: int = 3,
    keep: int = 10,
    hop_reserve: int = 4,
    redundancy: float = 0.8,
    uncertainty_floor: float = 0.15,
) -> MultiHopResult:
    """Retrieve, follow what was found, then merge and filter."""
    first_titles = index.search([question], paragraphs_k)[0]
    first = select_evidence(question, first_titles, paragraphs, hop=1)

    queries = expand_queries(question, first, hop_width)
    second: list[Evidence] = []
    if queries:
        for query, titles in zip(queries, index.search(queries, paragraphs_k), strict=True):
            second.extend(select_evidence(query, titles, paragraphs, hop=2))

    # The two hops are not ranked against each other. Their scores come from
    # different queries, so comparing them is meaningless, and concatenating
    # them is worse than meaningless: the first hop alone produces three times
    # the places available, so everything the second hop found is truncated
    # away and the system is single hop wearing a second retrieval.
    #
    # So the second hop is given a reserved share of the result. Its job is to
    # supply what the first could not reach, and a place it does not win on a
    # comparable score it has to be handed.
    reserve = min(hop_reserve, keep)
    kept: list[Evidence] = []
    seen: set[Fact] = set()

    def admit(item: Evidence, limit: int) -> bool:
        if len(kept) >= limit or item.fact in seen or _redundant(item, kept, redundancy):
            return False
        seen.add(item.fact)
        kept.append(item)
        return True

    for item in sorted(first, key=lambda e: -e.score):
        admit(item, keep - reserve)
    for item in sorted(second, key=lambda e: -e.score):
        admit(item, keep)
    # Anything the second hop did not fill goes back to the first.
    for item in sorted(first, key=lambda e: -e.score):
        admit(item, keep)

    # Flagged rather than suppressed. A question whose best evidence barely
    # matches is one a reader should be told about, not one to answer quietly.
    best = kept[0].score if kept else 0.0
    return MultiHopResult(
        evidence=kept, queries=[question, *queries], uncertain=best < uncertainty_floor
    )
