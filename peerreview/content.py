"""Weakness extraction and the coverage judge: does a model review raise each point the human reviewers raised?

Label scheme (``coverage_labels.csv``): for one model review and the
paper's list of human points, every point gets a recognition label (specific / partial / absent)
and a stance (endorses / disputes / mixed / unclear; not_applicable when absent).
"""
from __future__ import annotations

import re

from . import prompts
from .config import Model

TEMPLATE = "match_points"
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE = 0.0
RECOGNITION = ("specific", "partial", "absent")
STANCE = ("endorses", "disputes", "mixed", "unclear", "not_applicable")
LINE = re.compile(r"^\W*([A-Za-z]\d{3}|[A-Za-z0-9_.-]+?)\W*:\W*(specific|partial|absent)\W+"
                  r"(endorses|disputes|mixed|unclear|not_applicable)\b", re.IGNORECASE | re.MULTILINE)


def points_block(points: list[dict]) -> str:
    """Human points as the annotator sees them: id, polarity, text."""
    return "\n".join(f"- {p['id']} ({p['polarity']}): {p['text']}" for p in points)


def review_block(items: list[dict], summary: str = "") -> str:
    """A model review as the annotator sees it: summary (if any), strengths, weaknesses."""
    parts = []
    if summary and summary.strip():
        parts.append("### Summary\n" + summary.strip())
    for polarity, heading in [("strength", "### Strengths"), ("weakness", "### Weaknesses")]:
        chosen = [i["text"] for i in items if i["polarity"] == polarity]
        parts.append(heading + "\n" + ("\n".join(f"- {t}" for t in chosen) if chosen else "(none)"))
    return "\n\n".join(parts)


def match_prompt(points: list[dict], review_items: list[dict] | None, summary: str = "",
                 review_text: str | None = None) -> str:
    """``review_text`` (e.g. a human review's section of a dossier) replaces the item-based
    rendering of a model review, so human and model reviews go through the same judge."""
    review = review_text if review_text is not None else review_block(review_items or [], summary)
    return prompts.render(TEMPLATE, points=points_block(points), review=review)


def parse_match(raw: str, expected_ids: list[str]) -> tuple[dict[str, tuple[str, str]], str | None]:
    """``{id: (recognition, stance)}`` for every expected id, or an error. The last line for an
    id wins; stance is normalized to the recognition (absent -> not_applicable)."""
    labels: dict[str, tuple[str, str]] = {}
    expected = {i.lower(): i for i in expected_ids}
    for match in LINE.finditer(raw):
        key = match.group(1).lower()
        if key not in expected:
            continue
        recognition, stance = match.group(2).lower(), match.group(3).lower()
        if recognition == "absent":
            stance = "not_applicable"
        elif stance == "not_applicable":
            stance = "unclear"
        labels[expected[key]] = (recognition, stance)
    missing = [i for i in expected_ids if i not in labels]
    if missing:
        return labels, f"{len(missing)} of {len(expected_ids)} points not labeled: {missing[:5]}"
    return labels, None


def match_review(client, annotator: Model, points: list[dict], review_items: list[dict] | None, *,
                 summary: str = "", run_key: int = 0, attempts: int = 3,
                 review_text: str | None = None, attempt_offset: int = 0) -> dict:
    """Label one review against a paper's points. Returns labels, the raw response, usage and the
    attempt that parsed (a response that does not label every point is sampled again).
    ``attempt_offset`` continues the numbering after failed attempts already on disk, so a rerun
    draws fresh samples instead of re-reading the cached failures."""
    prompt = match_prompt(points, review_items, summary, review_text)
    ids = [p["id"] for p in points]
    record = {}
    for attempt in range(attempt_offset, attempt_offset + attempts):
        kwargs = dict(run_idx=run_key, temperature=TEMPERATURE if annotator.accepts_temperature
                      else None, max_tokens=MAX_OUTPUT_TOKENS)
        if attempt:
            kwargs["attempt"] = attempt
        if annotator.request:
            kwargs["extra"] = annotator.request
        raw, usage = client.complete(annotator.api_model, prompt, **kwargs)
        labels, error = parse_match(raw, ids)
        record = dict(labels=labels, parse_ok=error is None, parse_error=error, raw_response=raw,
                      usage=usage, attempt=attempt + 1, prompt_sha256=prompts.sha256(prompt),
                      annotator=annotator.key)
        if error is None:
            break
    if record and not record["parse_ok"] and \
            len(ids) - len(record["labels"]) <= max(3, 0.2 * len(ids)):
        # The usual failure is a long list with one or two ids skipped: ask for just those.
        missing = [p for p in points if p["id"] not in record["labels"]]
        follow_up = match_prompt(missing, review_items, summary, review_text)
        for attempt in range(2):
            kwargs = dict(run_idx=run_key, temperature=TEMPERATURE if annotator.accepts_temperature
                          else None, max_tokens=MAX_OUTPUT_TOKENS, attempt=attempt_offset + attempt)
            if annotator.request:
                kwargs["extra"] = annotator.request
            raw2, usage2 = client.complete(annotator.api_model, follow_up, **kwargs)
            extra_labels, error2 = parse_match(raw2, [p["id"] for p in missing])
            record["labels"].update(extra_labels)
            record["follow_up"] = dict(raw_response=raw2, usage=usage2, missing=[p["id"] for p in missing],
                                       attempt=attempt + 1)
            if error2 is None:
                record["parse_ok"], record["parse_error"] = True, None
                break
    return record


# --- extraction step -------------------------------------------------------------------------

EXTRACT_TEMPLATE = "extract_points"
EXTRACT_MAX_OUTPUT_TOKENS = 16384
POLARITY = ("strength", "weakness", "neutral")
CATEGORY = ("soundness", "experiments", "novelty", "significance", "reproducibility",
            "related work", "clarity", "other")
KIND = ("substantive", "presentation", "neutral_question")
ASSESSABILITY = ("supplied_material", "outside_information", "uncertain")


def reviews_block(dossier: dict) -> tuple[str, str]:
    """(abstract, reviews) from a dossier's holistic text. Reviewer A, B, C ... become review ids
    H00, H01, H02 ... (the frozen items' convention)."""
    text = dossier["holistic_text"]
    abstract = text.split("## Abstract\n", 1)[1].split("\n\n## ", 1)[0].strip()
    sections = re.split(r"\n## Reviewer ([A-Z])\n", text)
    reviews = []
    for k in range(1, len(sections), 2):
        letter, body = sections[k], sections[k + 1].strip()
        reviews.append(f"### Review H{ord(letter) - ord('A'):02d}\n{body}")
    return abstract, "\n\n".join(reviews)


def human_review_sections(dossier: dict) -> dict[str, str]:
    """Each human review's text from a dossier's holistic text, keyed H00, H01, ... in reviewer
    order (Reviewer A = H00), the convention of the reference items' sources."""
    sections = re.split(r"\n## Reviewer ([A-Z])\n", dossier["holistic_text"])
    return {f"H{ord(sections[k]) - ord('A'):02d}": sections[k + 1].strip()
            for k in range(1, len(sections), 2)}


def extract_prompt(dossier: dict) -> str:
    abstract, reviews = reviews_block(dossier)
    return prompts.render(EXTRACT_TEMPLATE, abstract=abstract, reviews=reviews)


def parse_points(raw: str) -> tuple[list[dict], str | None]:
    """The points list, with fields checked and normalized, or an error."""
    import json

    from . import parsing
    try:
        points = parsing._json_object(raw, "points")["points"]
    except (ValueError, KeyError, TypeError) as error:
        # e.g. a second object or prose after the JSON: decode the first complete object instead
        decoder, points = json.JSONDecoder(), None
        for start in (m.start() for m in re.finditer(r"\{", raw)):
            try:
                candidate, _ = decoder.raw_decode(raw[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "points" in candidate:
                points = candidate["points"]
                break
        if points is None:
            return [], f"{type(error).__name__}: {error}"
    if not isinstance(points, list) or not points:
        return [], "no points"
    out = []
    for k, p in enumerate(points, start=1):
        if not isinstance(p, dict) or not str(p.get("text", "")).strip():
            return [], f"point {k} has no text"
        polarity = str(p.get("polarity", "")).lower()
        category = str(p.get("category", "")).lower().replace("_", " ")
        kind = str(p.get("kind", "")).lower()
        assess = str(p.get("assessability", "")).lower()
        if polarity not in POLARITY or category not in CATEGORY or kind not in KIND \
                or assess not in ASSESSABILITY:
            return [], f"point {k}: invalid field values {polarity}/{category}/{kind}/{assess}"
        sources = [dict(review=str(s.get("review", "")), quote=str(s.get("quote", "")))
                   for s in (p.get("sources") or []) if isinstance(s, dict)]
        out.append(dict(id=f"C{k:03d}", polarity=polarity, category=category, kind=kind,
                        assessability=assess,
                        availability_reason=str(p.get("availability_reason", "")),
                        text=str(p["text"]).strip(), sources=sources))
    return out, None


def extract_points(client, annotator: Model, dossier: dict, *, run_key: int = 0,
                   attempts: int = 3) -> dict:
    """Extract a paper's human points. A response that does not parse is sampled again."""
    prompt = extract_prompt(dossier)
    record = {}
    for attempt in range(attempts):
        kwargs = dict(run_idx=run_key, temperature=TEMPERATURE if annotator.accepts_temperature
                      else None, max_tokens=EXTRACT_MAX_OUTPUT_TOKENS,
                      extra={**annotator.request, "response_format": {"type": "json_object"}})
        if attempt:
            kwargs["attempt"] = attempt
        raw, usage = client.complete(annotator.api_model, prompt, **kwargs)
        points, error = parse_points(raw)
        record = dict(points=points, parse_ok=error is None, parse_error=error, raw_response=raw,
                      usage=usage, attempt=attempt + 1, prompt_sha256=prompts.sha256(prompt),
                      annotator=annotator.key)
        if error is None:
            break
    return record


# --- omission curve -----------------------------------------------------------------------------

def omission_curve(recognizing: list[int], n_reviews: int) -> list[float]:
    """Share of points missed by n pooled reviews, n = 1..n_reviews, averaged exactly over all
    size-n subsets of the reviews (analysis/content/omissions.py, Figure 1b). ``recognizing`` has,
    per point, how many of the ``n_reviews`` reviews recognize it."""
    from math import comb
    curve = []
    for n in range(1, n_reviews + 1):
        denominator = comb(n_reviews, n)
        missed = [comb(n_reviews - k, n) / denominator if n_reviews - k >= n else 0.0
                  for k in recognizing]
        curve.append(sum(missed) / len(missed) if missed else float("nan"))
    return curve


def human_omission_curve(points_by_paper: dict[str, list[dict]], reviews_by_paper: dict[str, int],
                         max_pool: int = 3) -> list[float]:
    """Share of human-raised weaknesses missed by n pooled HUMAN reviews, n = 1..max_pool, exact over
    all size-n subsets of each paper's human reviews (the human counterpart of ``omission_curve``,
    as the human committees in Figure 1a are the counterpart of the LLM committees). A point raised
    by k of a paper's R reviewers is missed by a size-n subset with probability C(R-k, n) / C(R, n).
    Papers with fewer than max_pool + 1 reviews are left out so that every pooled subset leaves at
    least one reviewer out."""
    from math import comb
    pairs = []
    for pid, points in points_by_paper.items():
        n_reviews = reviews_by_paper.get(pid, 0)
        if n_reviews < max_pool + 1:
            continue
        for p in points:
            if p["polarity"] != "weakness":
                continue
            raised_by = len({s["review"] for s in p.get("sources", []) if s.get("review")}) or 1
            pairs.append((min(raised_by, n_reviews), n_reviews))
    curve = []
    for n in range(1, max_pool + 1):
        missed = [comb(r - k, n) / comb(r, n) if r - k >= n else 0.0 for k, r in pairs]
        curve.append(sum(missed) / len(missed) if missed else float("nan"))
    return curve
