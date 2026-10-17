"""Inputs: the paper list, the paper text a model reads, and the human-review dossiers."""
from __future__ import annotations

import json
import re
from functools import lru_cache

import pandas as pd

from .config import DATA, ROOT
from .prompts import TITLE_MASK

MANIFEST = DATA / "manifests" / "study_manifest.json"


@lru_cache
def study_papers(manifest=MANIFEST) -> pd.DataFrame:
    """One row per paper: ids, venue, outcome, human ratings, where the paper text lives."""
    return pd.DataFrame(json.loads(manifest.read_text())["papers"])


def mask_paper(raw: str, title: str) -> str:
    """Drop the title and author block before the abstract and mask the title everywhere:
    recognizing a title is a route to recalling the recorded outcome."""
    lines = raw.split("\n")
    start = next((i for i, line in enumerate(lines[:400])
                  if line.strip().lstrip("#").strip().rstrip(":").strip().lower() == "abstract"),
                 None)
    body = "\n".join(lines[start:]) if start is not None else raw
    if title:
        body = re.sub(re.escape(title.strip()), TITLE_MASK, body, flags=re.IGNORECASE)
    return TITLE_MASK + "\n\n" + body


def pdf_to_text(pdf_path) -> str:
    """Plain text of a PDF, page by page; save the result where the manifest's ``paper_text_path`` points."""
    import pymupdf
    with pymupdf.open(pdf_path) as document:
        return "\n\n".join(page.get_text() for page in document)


def paper_text(paper: pd.Series) -> str:
    return mask_paper((ROOT / paper.paper_text_path).read_text(), paper.title)


def load_dossier(venue: str, paper_id: str) -> dict:
    """Human reviews of one paper (``data/dossiers/<venue>/``): ``holistic_text``, the reviews with ratings removed,
    and the reviewer labels."""
    path = DATA / "dossiers" / venue / "study" / f"{paper_id}.json"
    if not path.exists():
        path = DATA / "dossiers" / venue / f"{paper_id}.json"
    return json.loads(path.read_text())

