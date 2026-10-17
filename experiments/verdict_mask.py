"""Masking of reviews for the discussion with hidden verdicts: score phrases and sentences that state a verdict."""
from __future__ import annotations

import re

MASK = re.compile(r"\b(score|rating|recommendation|rate it|give it)\b(\s*(of|to|at|:|=|is|as))?\s*\d+(\.\d+)?", re.IGNORECASE)
# Variant "verdict_masked": sentences that state or imply the verdict are removed from every review
# (LLM and human alike), so the predictor must infer the score from the substance of the review.
VERDICT = re.compile(r"\b(accept\w*|reject\w*|recommend\w*|lean(s|ing)?\s+(toward|towards)|borderline|threshold|"
                     r"(above|below|meets?|clear\w*)\s+the\s+bar|publish\w*|publication|venue|iclr|conference|"
                     r"(strong|weak|good|solid|excellent|marginal)\s+(paper|submission|contribution|accept)|"
                     r"(overall|in sum|in summary|on balance)\b[^.]*\b(positive|negative|favou?rabl\w*))\b", re.IGNORECASE)
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def verdict_masked(text: str) -> str:
    kept = [s for s in SENTENCE.split(text) if s.strip() and not VERDICT.search(s)]
    return " ".join(kept)


def masked(text: str) -> str:
    return MASK.sub(lambda m: f"{m.group(1)} [removed]", text)
