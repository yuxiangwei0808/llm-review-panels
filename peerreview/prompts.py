"""Prompt templates (``prompts/*.md``) and how each call fills them.

A template is a Markdown file with a YAML header (id, version, provenance notes) and a body with
``{placeholder}`` fields. Rendering is plain substitution, so a rendered prompt is reproducible
byte for byte; every saved call records its SHA-256 (``extra.rendered_prompt_sha256``).
"""
from __future__ import annotations

import hashlib
import random
from functools import lru_cache

import yaml

from .config import PROMPTS, Venue

TITLE_MASK = "[title withheld]"


@lru_cache
def load_template(name: str) -> tuple[dict, str]:
    """Return (header, body) of ``prompts/<name>.md``."""
    text = (PROMPTS / f"{name}.md").read_text()
    _, header, body = text.split("---\n", 2)
    return yaml.safe_load(header), body.lstrip("\n")


def render(name: str, **fields: str) -> str:
    _, body = load_template(name)
    for key, value in fields.items():
        body = body.replace("{" + key + "}", value)
    return body


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def items_block(items: list[dict]) -> str:
    """Strengths then weaknesses, each item under its id so ratings can refer to it."""
    parts = []
    for polarity, heading in [("strength", "### STRENGTHS"), ("weakness", "### WEAKNESSES")]:
        chosen = [item for item in items if item["polarity"] == polarity]
        if chosen:
            entries = [f"- item_id: `{item['item_id']}`\n  {item['text']}" for item in chosen]
            parts.append(heading + "\n" + "\n\n".join(entries))
    return "\n\n".join(parts)


def presentation_order(items: list[dict], paper_id: str, run_idx: int, seed: int) -> list[dict]:
    """Own-review points are shown to the rater in a seeded arbitrary order, so that the order in
    which the review listed them carries no weight. (The original runs shuffled too; their
    exact shuffle was lost with the original code, so the order here differs from theirs.)"""
    shuffled = list(items)
    random.Random(f"{seed}:{paper_id}:{run_idx}").shuffle(shuffled)
    return shuffled


def paper_context(abstract: str) -> str:
    return f"Title: [withheld]\n\nAbstract: {abstract}"


def own_review(venue: Venue, paper_text: str) -> str:
    """Call A, own evidence: the model reviews the paper and chooses a score."""
    return render("call_a", venue_name=venue.prompt_name,
                  rating_scale_block=venue.rating_scale_block,
                  allowed_scores=venue.allowed_scores, paper_text=paper_text)


def shared_evidence(venue: Venue, dossier_text: str) -> str:
    """Call F-holistic, shared evidence: the model scores the human reviews (ratings removed)."""
    return render("call_f_holistic", venue_name=venue.prompt_name, venue_short=venue.short_name,
                  rating_scale_block=venue.rating_scale_block,
                  allowed_scores=venue.allowed_scores, dossier=dossier_text)


def rubric(template: str, venue: Venue, abstract: str, items: list[dict]) -> str:
    """Calls B (``call_b``, own review) and F-itemized (``call_f_itemized``, human reviews):
    the model rates every point; a fixed rule turns the ratings into a score."""
    return render(template, venue_name=venue.prompt_name, paper_context=paper_context(abstract),
                  items_block=items_block(items))
