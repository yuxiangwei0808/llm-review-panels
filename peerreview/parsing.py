"""Parse model responses into scores and review points.

Re-parsing a saved record's ``raw_response`` gives back the record's score and points.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

REASONING = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
ITEM_PREFIX = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$", re.DOTALL)
RATING_LINE = re.compile(r"RATING:\s*\**\s*(\d+)")
ACCEPT_PROB_LINE = re.compile(r"ACCEPT_PROB:\s*\**\s*([01](?:\.\d+)?)\**\s*$", re.MULTILINE)
DECISIVE_LINE = re.compile(r"DECISIVE_FACTOR:\s*(.+)")
ESCAPE = re.compile(r"\\(u[0-9a-fA-F]{4}|.)", re.DOTALL)


@dataclass
class Parsed:
    ok: bool
    error: str | None = None
    score: int | None = None
    accept_prob: float | None = None
    reasoning: str = ""
    items: list[dict] = field(default_factory=list)          # review points (first reviews)
    extra: dict = field(default_factory=dict)


def _keep_valid_escape(match: re.Match) -> str:
    """Leave valid JSON escapes alone; turn any other backslash into a literal backslash."""
    follower = match.group(1)
    valid = len(follower) == 5 or follower in '"\\/bfnrt'
    return match.group(0) if valid else "\\\\" + follower


def _loads(blob: str) -> dict:
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # LaTeX in review text leaves backslashes that are not valid JSON escapes.
        return json.loads(ESCAPE.sub(_keep_valid_escape, blob))


def _json_object(text: str, required_key: str) -> dict:
    """The JSON object holding ``required_key`` in a response that may carry prose, code fences
    or stray braces around it."""
    end = text.rfind("}")
    starts = [m.start() for m in re.finditer(r"\{", text[: end + 1])]
    error: Exception = ValueError("no JSON object found")
    for start in starts:
        try:
            candidate = _loads(text[start: end + 1])
        except json.JSONDecodeError as exc:
            error = exc
            continue
        if isinstance(candidate, dict) and required_key in candidate:
            return candidate
    raise error


def _leading_int(value) -> int:
    """'3 (based on the rating scale ...)' -> 3."""
    return int(float(re.match(r"\s*(-?\d+(?:\.\d+)?)", str(value)).group(1)))


def _review_by_pattern(text: str) -> dict:
    """Last resort for JSON broken by unescaped quotes: read the fields line by line."""
    score = re.search(r'"overall_score"\s*:\s*"?\s*(\d+)', text)
    if not score:
        raise ValueError("no overall_score found")
    review = {"rating": {"overall_score": score.group(1)}}
    confidence = re.search(r'"confidence"\s*:\s*"?\s*(\d+)', text)
    review["rating"]["confidence"] = confidence.group(1) if confidence else None
    for key in ("strengths", "weaknesses"):
        block = re.search(rf'"{key}"\s*:\s*\[(.*?)\n\s*\]', text, re.DOTALL)
        lines = re.findall(r'^\s*"(.*?)",?\s*$', block.group(1), re.MULTILINE) if block else []
        review[key] = [line.replace('\\"', '"') for line in lines]
    seen = re.search(r'"seen_before"\s*:\s*"([^"]*)"', text)
    review["prior_knowledge"] = {"seen_before": seen.group(1) if seen else ""}
    return review


def _reasoning(raw: str) -> tuple[str, str]:
    match = REASONING.search(raw)
    # Some models put the answer before the reasoning block, so keep the text on both sides.
    return (match.group(1).strip(), raw[:match.start()] + raw[match.end():]) if match else ("", raw)


def parse_own_review(raw: str, allowed: tuple[int, ...]) -> Parsed:
    """A review: ``<reasoning>`` block, then the review JSON with a rating."""
    reasoning, rest = _reasoning(raw)
    try:
        try:
            review = _json_object(rest, "rating")
        except ValueError:
            review = _review_by_pattern(rest)
        score = _leading_int(review["rating"]["overall_score"])
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        return Parsed(ok=False, error=f"{type(error).__name__}: {error}", reasoning=reasoning)
    if score not in allowed:
        return Parsed(ok=False, error=f"score {score} not in {allowed}", reasoning=reasoning)
    items = []
    for polarity, key, tag in [("strength", "strengths", "s"), ("weakness", "weaknesses", "w")]:
        for k, text in enumerate(review.get(key) or [], start=1):
            match = ITEM_PREFIX.match(str(text))
            dimension, body = (match.group(1).strip(), match.group(2).strip()) if match \
                else ("", str(text).strip())
            items.append(dict(item_id=f"{tag}{k}", polarity=polarity, text=body, dimension=dimension))
    prior = review.get("prior_knowledge")
    prior = prior if isinstance(prior, dict) else {"seen_before": prior or ""}
    extra = dict(seen_before=str(prior.get("seen_before", "")),
                 seen_before_explanation=str(prior.get("explanation", "")),
                 confidence=review["rating"].get("confidence"), summary=review.get("summary", ""))
    return Parsed(ok=True, score=score, reasoning=reasoning, items=items, extra=extra)


def parse_rating_lines(raw: str, allowed: tuple[int, ...]) -> Parsed:
    """A score from the human reviews: ``<reasoning>`` block, then DECISIVE_FACTOR / ACCEPT_PROB / RATING lines."""
    reasoning, rest = _reasoning(raw)
    ratings = RATING_LINE.findall(rest)
    if not ratings:
        return Parsed(ok=False, error="no RATING line", reasoning=reasoning)
    score = int(ratings[-1])
    if score not in allowed:
        return Parsed(ok=False, error=f"score {score} not in {allowed}", reasoning=reasoning)
    prob = ACCEPT_PROB_LINE.findall(rest)
    decisive = DECISIVE_LINE.findall(rest)
    return Parsed(ok=True, score=score, reasoning=reasoning,
                  accept_prob=float(prob[-1]) if prob else None,
                  extra=dict(decisive_factor=decisive[-1].strip() if decisive else ""))

