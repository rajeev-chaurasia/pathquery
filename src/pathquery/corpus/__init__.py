"""Reading HotpotQA and turning it into a corpus with exact character offsets."""

from pathquery.corpus.build import CorpusSpec, build_corpus
from pathquery.corpus.hotpot import Paragraph, Question, read_split

__all__ = ["CorpusSpec", "Paragraph", "Question", "build_corpus", "read_split"]
