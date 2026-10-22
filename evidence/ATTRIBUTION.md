# Attribution for the committed annotations

`questions.jsonl` holds the 1,200 questions this repository is scored on,
together with their supporting-fact annotations, taken from the HotpotQA
development set.

`cited-paragraphs.jsonl` holds the 11,236 paragraphs that some citation in a
committed run points at, as HotpotQA distributes them, sentence by sentence.
Without them a citation could only be checked by someone who had rebuilt the
whole corpus first.

HotpotQA is distributed under Creative Commons Attribution-ShareAlike 4.0
International, which permits redistribution with attribution and under the same
licence. That is why these files are committed here while the full paragraph
corpus is not: the corpus is a few hundred megabytes and rebuildable from a
seed, whereas without the annotations nobody can recompute a single number in
this repository.

Source: Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov and Manning, "HotpotQA: A
Dataset for Diverse, Explainable Multi-hop Question Answering", 2018.
Licence: CC BY-SA 4.0. Both files are redistributed under the same terms.
