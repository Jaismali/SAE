"""
Minimal degenerate-repetition detector for generated text.

SCOPE, DELIBERATELY NARROW: this detects whether a short n-gram (2-4
words) repeats suspiciously often within a single generated
continuation -- e.g. "I noticed coral." repeated 5 times, or 'The word
"beach" is a noun.' repeated 3 times, both observed directly in this
session's induction=4 ocean_theme run.

This is a VISIBILITY tool, not a fix: it flags per-prompt degeneracy
so it doesn't get silently averaged away by an aggregate score (e.g.
TokenFrequencyMeasure), it does NOT replace that score, redesign it,
or attempt to fix the underlying training instability that produces
degenerate output. Scoped this way deliberately, per explicit
instruction to patch the specific visibility gap just identified, not
to build a general research effort.

Motivation: TokenFrequencyMeasure averages across 5 held-out prompts
into one score. A single degenerate/collapsed prompt can be diluted
into a still-plausible-looking aggregate pass, rather than surfaced.
This is suspected to be a PRE-EXISTING gap (not necessarily specific
to any particular epoch count) -- see the audit trail for whether the
original Month 1 3-epoch ocean_theme checkpoint shows the same pattern
when inspected directly.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RepetitionCheckResult:
    is_degenerate: bool
    worst_ngram: str  # the most-repeated n-gram found, or "" if none flagged
    ngram_size: int  # word-length of worst_ngram, or 0 if none flagged
    repeat_count: int  # how many times worst_ngram appeared, or 0 if none flagged


DEFAULT_NGRAM_SIZES: Tuple[int, ...] = (2, 3, 4)
DEFAULT_MIN_REPEAT_COUNT = 3  # an n-gram appearing this many times or more is flagged


def _tokenize_words(text: str) -> List[str]:
    """Simple whitespace/punctuation-aware word tokenization for n-gram
    counting. Deliberately simple (not using a real tokenizer) since
    this only needs to catch GROSS repetition, not do linguistic
    analysis. Lowercased so 'Coral' and 'coral' count as the same word
    for repetition purposes.
    """
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def detect_ngram_repetition(
    text: str,
    ngram_sizes: Tuple[int, ...] = DEFAULT_NGRAM_SIZES,
    min_repeat_count: int = DEFAULT_MIN_REPEAT_COUNT,
) -> RepetitionCheckResult:
    """Checks whether any word n-gram (of the given sizes) repeats at
    least min_repeat_count times within `text`.

    Returns the WORST (most-repeated, and among ties, largest-size)
    offending n-gram, so a caller printing one summary line gets the
    most informative single flag rather than every n-gram size's result.
    """
    words = _tokenize_words(text)
    if not words:
        return RepetitionCheckResult(is_degenerate=False, worst_ngram="", ngram_size=0, repeat_count=0)

    worst_ngram = ""
    worst_size = 0
    worst_count = 0

    for size in ngram_sizes:
        if len(words) < size:
            continue
        ngram_counts = Counter(
            tuple(words[i : i + size]) for i in range(len(words) - size + 1)
        )
        most_common_ngram, count = ngram_counts.most_common(1)[0]
        if count >= min_repeat_count and count > worst_count:
            worst_ngram = " ".join(most_common_ngram)
            worst_size = size
            worst_count = count

    is_degenerate = worst_count >= min_repeat_count
    return RepetitionCheckResult(
        is_degenerate=is_degenerate,
        worst_ngram=worst_ngram,
        ngram_size=worst_size,
        repeat_count=worst_count,
    )
