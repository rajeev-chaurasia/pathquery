# Known misses

Written against the repository rather than for it.

## The second hop does not pay for itself on recall

The plan committed to multi-hop beating single-hop by at least 0.15 on bridge
question evidence Recall@10. Measured, it is 0.00490 worse: 0.52536 against
0.53026. That threshold is missed by the whole of its margin and then some.

What the second hop does do is complete evidence sets slightly more often.
Fully supported at ten, meaning every supporting sentence was found rather than
some of them, rises from 0.20872 to 0.22845 on the same questions. That is the
metric a multi-hop system exists to move, and it moves in the right direction by
about two points while overall recall falls by half a point. The places the
second hop takes cost more partial hits than its completions are worth.

Two things are missing from the system that the plan's order of work lists and
that would be expected to change this, and neither is built: dense retrieval
fused with the lexical half, and a cross-encoder reranking the merged evidence.
The result above is what lexical retrieval alone does, and it should be read as
that rather than as a statement about multi-hop retrieval.

## The first merge made the second hop impossible

Before the result above, multi-hop scored 0.52922 on bridge questions, which is
within a thousandth of single-hop. That was not a finding. The merge
concatenated the first hop's evidence ahead of the second's, and the first hop
alone produces three times as many candidates as there are places, so
everything the second hop found was truncated away in every case. The system
was single hop wearing an extra retrieval.

It is recorded because the number it produced was entirely plausible, and a
plausible number from a broken merge is worse than an implausible one.

## Entities are found by capitalisation

The bridge entity driving the second query is extracted by looking for runs of
capitalised words the question did not already contain. That finds "Tim Burton"
and it also finds the first word of any sentence that happens to follow a
quotation mark. A named entity recogniser would do better, and the claim being
tested is about whether following the first hop helps rather than about the
extractor, so the crude version is what is measured.

## The uncertainty flag never fires

A question is flagged as uncertain when its best piece of evidence covers less
than 0.15 of the question's terms. Over 1,200 questions it flagged none. The
floor was set before the run and never tuned against anything, and sentence
selection keeps the sentences with the most overlap, so the best one clears it
every time. As configured the flag does nothing, and nothing here should be read
as evidence that it works.
