"""Prompt templates (``prompts/*.md``) and how each call fills them.

A template is a Markdown file with a YAML header (id, version, notes) and a body with
``{placeholder}`` fields. Rendering is plain substitution, so a rendered prompt is reproducible
byte for byte; every saved call records its SHA-256 (``extra.rendered_prompt_sha256``).
"""
from __future__ import annotations

import hashlib
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


def own_review(venue: Venue, paper_text: str) -> str:
    """Call A, a first review: the model reviews the paper and chooses a score."""
    return render("call_a", venue_name=venue.prompt_name,
                  rating_scale_block=venue.rating_scale_block,
                  allowed_scores=venue.allowed_scores, paper_text=paper_text)


def shared_evidence(venue: Venue, dossier_text: str) -> str:
    """Call F_holistic: the model scores the paper from the human reviews (ratings removed) instead of its own."""
    return render("call_f_holistic", venue_name=venue.prompt_name, venue_short=venue.short_name,
                  rating_scale_block=venue.rating_scale_block,
                  allowed_scores=venue.allowed_scores, dossier=dossier_text)
