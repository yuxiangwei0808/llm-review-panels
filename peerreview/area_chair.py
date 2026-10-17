"""LLM area chairs: the prompt that gives a model a paper and a set of reviews, and the parser of its decision.

The model writes a meta-review and decides accept or reject (prompt ``ac_binary``). ``experiments/area_chairs.py``
builds the review sets and calls the models.
"""
from __future__ import annotations

import re

from . import prompts
from .config import venues

MAX_OUTPUT_TOKENS = 2048
DECISION = re.compile(r"<decision>\s*(accept|reject)\s*</decision>", re.IGNORECASE)
META_REVIEW = re.compile(r"<meta_review>(.*?)</meta_review>", re.DOTALL | re.IGNORECASE)


def render_prompt(venue_key: str, paper: str, packet: str) -> str:
    """The area-chair prompt for one paper: its text and the reviews it gets (``packet``)."""
    venue = venues()[venue_key]
    return prompts.render("ac_binary", venue_name=venue.name,
                          rating_scale_block=venue.rating_scale_block,
                          accept_threshold=str(venue.accept_threshold), paper_text=paper,
                          evidence_packet=packet)


def parse_decision(raw: str) -> tuple[str | None, str]:
    """(``accept`` | ``reject`` | None, meta-review text)."""
    decision = DECISION.findall(raw or "")
    review = META_REVIEW.search(raw or "")
    return (decision[-1].lower() if decision else None), (review.group(1).strip() if review else "")
