"""ICLR 2022 forum threads: reviews, author responses, reviewer discussion, decision.

Reads data/raw/openreview_corpus/iclr2022/forums_v5.jsonl (OpenReview API v1 forums). Reviewers
are labelled by the dossier's letters (Reviewer A = Reviewer 1), the convention of every other
ICLR 2022 table in this repository. Used by the stage-3 rebuttal arm of
experiments/discussion.py and by experiments/area_chairs_iclr2022.py.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from peerreview.config import ROOT
from peerreview.papers import load_dossier

# 2026-10-02: the same OpenReview v1 layout serves ICLR 2021 and NeurIPS 2022 (PANEL_CORPUS); ICLR 2022 unchanged.
VENUE = os.environ.get("PANEL_CORPUS", "iclr2022") if os.environ.get("PANEL_CORPUS") in ("iclr2021", "neurips2022") else "iclr2022"
FORUMS = ROOT / "data" / "raw" / "openreview_corpus" / VENUE / ("forums_v5.jsonl" if VENUE == "iclr2022" else "forums.jsonl")
REVIEW_FIELDS = [("summary_of_the_paper", "Summary of the paper"), ("main_review", "Main review"),
                 ("summary_of_the_review", "Summary of the review"), ("correctness", "Correctness"),
                 ("technical_novelty_and_significance", "Technical novelty and significance"),
                 ("empirical_novelty_and_significance", "Empirical novelty and significance"),
                 ("recommendation", "Recommendation"), ("confidence", "Confidence")]


@lru_cache
def forums() -> dict[str, dict]:
    out = {}
    with FORUMS.open() as lines:
        for line in lines:
            r = json.loads(line)
            out[r["submission"]["id"]] = r
    return out


def _kind(note: dict) -> str:
    return note["invitation"].split("/")[-1]


def _signer(note: dict) -> str:
    return note["signatures"][0].split("/")[-1]


def reviewer_number(paper_id: str) -> dict[str, int]:
    """Reviewer id (e.g. Reviewer_qsDK) -> 1-based number in dossier letter order."""
    labels = load_dossier(VENUE, paper_id)["reviewer_labels"]
    labels = dict(json.loads(labels) if isinstance(labels, str) else labels)
    return {rid: ord(letter) - ord("A") + 1 for rid, letter in labels.items()}


def speaker(paper_id: str, note: dict) -> str:
    s = _signer(note)
    if s == "Authors":
        return "Authors"
    if s.startswith("Reviewer_"):
        return f"Reviewer {reviewer_number(paper_id).get(s, '?')}"
    if s.startswith("Area_Chair"):
        return "Area chair"
    if s == "Program_Chairs":
        return "Program chairs"
    return "Public commenter"


def reviews(paper_id: str) -> list[tuple[int, dict]]:
    """(reviewer number, review note) in reviewer order."""
    numbers = reviewer_number(paper_id)
    out = [(numbers.get(_signer(n), 99), n) for n in forums()[paper_id]["replies"] if _kind(n) == "Official_Review"]
    return sorted(out, key=lambda x: x[0])


def decision(paper_id: str) -> dict | None:
    notes = [n for n in forums()[paper_id]["replies"] if _kind(n) == "Decision"]
    return notes[0] if notes else None


def discussion_notes(paper_id: str) -> list[dict]:
    """Official and public comments before the decision, in time order."""
    d = decision(paper_id)
    cutoff = int(d["cdate"]) if d else None
    notes = [n for n in forums()[paper_id]["replies"] if _kind(n) in ("Official_Comment", "Public_Comment")
             and (cutoff is None or int(n["cdate"]) < cutoff)]
    return sorted(notes, key=lambda n: int(n["cdate"]))


def author_response(paper_id: str) -> str:
    """All author comments before the decision, labelled by what they reply to (a review, the
    submission, or an earlier comment). Empty string if the authors never responded."""
    review_of = {n["id"]: k for k, n in reviews(paper_id)}
    parts = []
    for n in discussion_notes(paper_id):
        if speaker(paper_id, n) != "Authors":
            continue
        target = n.get("replyto")
        if target in review_of:
            head = f"### Author response to Reviewer {review_of[target]}"
        elif target == paper_id:
            head = "### General author response"
        else:
            head = "### Author follow-up comment"
        title = (n["content"].get("title") or "").strip()
        body = (n["content"].get("comment") or "").strip()
        parts.append(f"{head}\n\n" + (f"**{title}**\n\n" if title else "") + body)
    return "\n\n".join(parts)


def review_text(note: dict) -> str:
    c = note["content"]
    return "\n\n".join(f"#### {label}\n\n{str(c[key]).strip()}" for key, label in REVIEW_FIELDS if c.get(key))


def human_packet(paper_id: str, with_discussion: bool) -> str:
    """Area-chair review record: the official reviews with scores (H0), plus the author-reviewer
    discussion before the decision (H2). Mirrors the section layout of the draft's packets."""
    parts = ["## CURRENT OFFICIAL REVIEW NOTES AND SCORES"]
    parts += [f"### Reviewer {k}\n\n{review_text(n)}" for k, n in reviews(paper_id)]
    if with_discussion:
        notes = discussion_notes(paper_id)
        if notes:
            parts.append("## PUBLIC REVIEWER-AUTHOR DISCUSSION BEFORE THE DECISION")
            for n in notes:
                title = (n["content"].get("title") or "").strip()
                body = (n["content"].get("comment") or "").strip()
                parts.append(f"### {speaker(paper_id, n)}\n\n" + (f"**{title}**\n\n" if title else "") + body)
    return "\n\n".join(parts)


if __name__ == "__main__":
    import sys
    pid = sys.argv[1] if len(sys.argv) > 1 else "6y2KBh-0Fd9"
    print(author_response(pid)[:1500])
    print("\n=====\n")
    print(human_packet(pid, True)[:1500])
