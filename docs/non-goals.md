# Non-goals

What this repository deliberately does not attempt, so that the claim it does
make can be read narrowly and checked.

## A resolving citation is not a supporting one

A citation resolves when the source text at its range is exactly the text it
quotes. That proves the quote is real and is where it says it is. It does not
prove the quoted sentence supports anything. A system could cite the first
sentence of every retrieved paragraph and resolve at 1.000 while finding no
evidence at all.

The two are measured separately for that reason. Resolution is checked against
the corpus; whether the right sentences were found is evidence recall, checked
against HotpotQA's own `supporting_facts`. Neither number stands in for the
other, and the README reports both.

## Answers

PathQuery retrieves and cites evidence. It does not write an answer, and no
answer accuracy is published. Judging a generated answer needs a person or a
model standing in for one, and neither belongs inside a claim about where text
came from.

## Open-domain retrieval

The corpus is 507,494 paragraphs pooled from HotpotQA's distractor setting,
training and development splits together, deduplicated by title. The full
Wikipedia setting is about five million. Every recall number here would be
lower against it, and none of them should be read as a statement about it.

## Dense retrieval and reranking

The plan's order of work lists a dense retriever fused with the lexical one,
and a reranker over the merged evidence. Neither is built. Every number in this
repository is lexical retrieval alone, and the multi-hop result in particular
should be read as what BM25 does when it is asked a second question, not as what
multi-hop retrieval can do. [known-misses.md](known-misses.md) traces this.

## Offsets into anything but this corpus

A citation's range counts characters in the paragraph as this repository joins
it from HotpotQA's sentence list. It is not an offset into a Wikipedia page, a
dump, or any other rendering of the same article, and it counts characters
rather than UTF-8 bytes. A consumer holding the text in another form has to map
the range onto it.

## Serving

There is no service, no latency figure and no throughput figure. Retrieval
runs as a batch over the question set. Nothing here is designed or measured as
something to put behind a request.
