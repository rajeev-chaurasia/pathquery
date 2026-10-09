"""Finding paragraphs, then finding the sentences inside them that matter."""

from pathquery.retrieve.lexical import ParagraphIndex
from pathquery.retrieve.sentences import Evidence, select_evidence

__all__ = ["Evidence", "ParagraphIndex", "select_evidence"]
