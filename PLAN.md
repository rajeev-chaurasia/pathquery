# Plan

Written and committed before the corpus was downloaded and before anything was
measured, so the thresholds below cannot be chosen to fit the results.

## The claim this repository has to earn

> PathQuery finds evidence that single-hop retrieval cannot reach, by using
> what the first hop found to ask the second question, and every span it cites
> resolves to exactly the source text it claims, checked by offset rather than
> asserted.

Two halves, and the second is the unusual one. Systems that cite their sources
are common; systems where the citation is a verified fact about a byte range
are not. A citation here is a document identifier and a character range, and
the check is whether the corpus at that range is character for character the
text that was quoted. That either holds or it does not, with nothing to argue
about.

The first half is the one that could easily be theatre. Decomposing a question
into hops is only worth its cost if it retrieves evidence that asking the whole
question at once does not, so the single-hop system is built first and kept as
the comparison throughout.

## Thresholds, fixed before the run

| what | threshold | why this number |
| --- | ---: | --- |
| single-hop evidence Recall@10, bridge questions | reported, not targeted | the comparison is only honest if the baseline is measured rather than assumed |
| multi-hop evidence Recall@10, bridge questions | at least 0.15 above single-hop | below this the decomposition is not paying for the second retrieval |
| citations resolving to their exact source span | 1.000 | a byte range either matches or it does not, so anything short of all of them is a defect, not a score |
| corrupted-offset control, citations resolving | at most 0.01 | the checker must reject spans that have been moved; if it does not, the figure above means nothing |
| shuffled-evidence control, Recall@10 | at most 0.02 | chance level, proving the evidence metric measures the annotations rather than itself |
| questions evaluated | 1,200 | drawn from the HotpotQA development set |
| corpus | about 500,000 paragraphs | pooled from the HotpotQA distractor paragraphs, deduplicated |

A threshold that is missed is published as missed, with the cause traced, and
pinned by the validator at its measured value.

## Definitions fixed in advance

**A hop** is one retrieval whose query was built from the result of the previous
one. A system that issues two retrievals from the original question alone has
not hopped, and is scored as single-hop however many requests it makes.

**Evidence recall** is measured against HotpotQA's `supporting_facts`, which
name a paragraph title and a sentence index. Those are the annotations, not
something this repository invents, and they are what the gold standard means.

**A citation resolves** when `corpus[document][start:end]` equals the quoted
text exactly, including whitespace. Not a fuzzy match, not a normalised one.

**Bridge and comparison questions are reported separately.** HotpotQA labels
them, and they are not the same problem: a comparison question names both
entities up front, so single-hop retrieval can often reach both pieces of
evidence, while a bridge question hides the second entity inside the first.
Pooling them would let easy questions hide what the hard ones show, in either
direction.

## The negative controls

1. **Corrupted offsets.** Every citation's span is shifted by a few characters
   and re-checked. The resolution rate has to collapse. A checker that still
   passes them is comparing something other than the bytes it claims to.
2. **Shuffled evidence.** The retrieved evidence is scored against another
   question's supporting facts. Recall has to fall to chance, or the metric is
   finding support that is not there.
3. **Single-hop, kept throughout.** Not a strawman to beat once but the
   standing comparison: every multi-hop number is published beside the
   single-hop number for the same questions.

## What is not claimed

Nothing here is a claim about answer quality. PathQuery retrieves and cites
evidence; it does not generate prose answers, and no answer-accuracy number is
published, because judging generated answers needs either a human or a model
standing in for one, and neither belongs inside a claim about provenance.

Nothing here is a claim about the full Wikipedia setting. The corpus is pooled
from HotpotQA's distractor paragraphs, which is a few hundred thousand rather
than five million, and the numbers would be lower on the larger one.

HotpotQA is released under CC BY-SA 4.0, so the corpus is fetched rather than
committed, and anything derived from it carries the same licence.

## Order of work

1. Evidence metrics and the citation rule, with no dependency on a retriever.
2. Corpus and splits from the HotpotQA development set, fixed by seed.
3. Single-hop retrieval, lexical and dense, and their fusion.
4. Decomposition: building the second query from the first hop's result.
5. Redundancy filtering and the uncertainty flag.
6. The run, then the README written from what it produced.
