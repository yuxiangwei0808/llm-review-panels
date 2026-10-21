"""LLM area chairs: nine models each read a paper and a set of reviews and decide accept or reject.

Conditions (field ``arm``), on every decided (accepted or rejected) paper of the venue:
  human_reviews  the official human reviews with their ratings, in reviewer order
  panel_reviews  the eight-family panel's first reviews (run 0) with their scores, in a seeded order, model names hidden
The chair writes a meta-review and a decision (prompt area_chair); the record keeps the venue's decision as
``venue_decision``.
A response that does not parse is sampled again up to three times; the third attempt after a response cut off at the
output limit gets twice the limit. The venue group is chosen with VENUE, as in experiments/discussion.py. ICLR 2022
reads the human reviews from the OpenReview forum notes (experiments/forums.py), the other venues from the dossiers.

Usage:
  VENUE=iclr2021 uv run python experiments/area_chairs.py run --max-usd 180
  uv run python experiments/area_chairs.py run --repeat 2 --max-usd 250     # ICLR 2022: a second independent run
Writes: results/<group>/area_chair/records.jsonl (ICLR 2022 repeat run: records_run2.jsonl).
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import discussion
import forums

from peerreview import area_chair, prompts
from peerreview.client import CostCeilingReached, OpenRouterClient
from peerreview.config import ROOT, SEED, TEMPERATURE, models
from peerreview.content import human_review_sections
from peerreview.papers import load_dossier, paper_text
from peerreview.runner import fit_to_context

OUT = discussion.OUT.parent / "area_chair"
CACHE = ROOT / "cache" / "responses" / "area_chairs"
CHAIRS = ["gemini-3.1-pro", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gpt-5.6-luna", "gpt-5.6-terra",
          "gpt-5.1", "deepseek-v4-flash-0731", "glm-5.2", "claude-sonnet-5"]
ARMS = ["human_reviews", "panel_reviews"]


def human_packet(venue_key: str, paper_id: str) -> str:
    """The human reviews from the paper's dossier, each followed by its rating; ICLR 2022 from the forum notes."""
    if venue_key == "iclr2022":
        return forums.human_packet(paper_id)
    d = load_dossier(venue_key, paper_id)
    text = human_review_sections(d)                                   # H00, H01, ... = Reviewer A, B, ...
    labels = d["reviewer_labels"]
    labels = json.loads(labels) if isinstance(labels, str) else labels
    src = d["source_reviews"]
    src = json.loads(src) if isinstance(src, str) else src
    rating = {s["reviewer_id"]: s["rating"] for s in src}
    letter_of = {letter: rid for rid, letter in labels}
    parts = ["## CURRENT OFFICIAL REVIEW NOTES AND SCORES"]
    for k, key in enumerate(sorted(text), start=1):
        rid = letter_of.get(chr(ord("A") + int(key[1:])))
        score = rating.get(rid)
        parts.append(f"### Reviewer {k}\n\n{text[key]}\n\n#### Overall recommendation\n\n"
                     + (f"{score} on the official {venue_key} reviewer scale" if score is not None else "not recorded"))
    return "\n\n".join(parts)


def llm_packet(venue_key: str, paper_id: str, reviews: dict[str, dict]) -> str:
    """The panel's first reviews with their scores, in a seeded order, model names hidden."""
    order = [m for m in discussion.PANEL if m in reviews]
    random.Random(f"{SEED}:ac:{paper_id}").shuffle(order)
    parts = ["## INITIAL ANONYMOUS REVIEWS"]
    for k, m in enumerate(order, start=1):
        r = reviews[m]
        extra = r.get("extra") or {}
        strengths = "\n".join(f"- {i['text']}" for i in r["items"] if i["polarity"] == "strength")
        weaknesses = "\n".join(f"- {i['text']}" for i in r["items"] if i["polarity"] == "weakness")
        parts.append(f"### Anonymous Reviewer {k}\n\n#### Public review rationale\n\n{(r.get('reasoning_text') or '').strip()}"
                     f"\n\n#### Summary\n\n{str(extra.get('summary', '')).strip()}\n\n#### Strengths\n\n{strengths}"
                     f"\n\n#### Weaknesses\n\n{weaknesses}\n\n#### Overall recommendation\n\n"
                     f"{r['emitted_score']} on the official {venue_key} reviewer scale")
    return "\n\n".join(parts)


def decided() -> pd.DataFrame:
    man = pd.DataFrame(json.loads(discussion.MANIFEST.read_text(encoding="utf-8"))["papers"])
    return man[man.outcome.isin(["accept", "reject"])].sort_values("paper_id").reset_index(drop=True)


def run(args):
    if args.repeat > 1:
        assert discussion.VENUE == "iclr2022", "repeat runs are on ICLR 2022"
    papers = decided()
    OUT.mkdir(parents=True, exist_ok=True)
    records = OUT / ("records.jsonl" if args.repeat == 1 else f"records_run{args.repeat}.jsonl")
    done = set()
    if records.exists():
        done = {(r["paper_id"], r["arm"], r["ac_model_id"]) for r in map(json.loads, records.open(encoding="utf-8")) if r["parse_ok"]}
    first = discussion.first_reviews(set(papers.paper_id))
    packet = {}
    for p in papers.itertuples():
        if "human_reviews" in args.arms:
            packet[(p.paper_id, "human_reviews")] = human_packet(p.venue_key, p.paper_id)
        revs = {m: first[(p.paper_id, m)] for m in discussion.PANEL if (p.paper_id, m) in first}
        if "panel_reviews" in args.arms and len(revs) == len(discussion.PANEL):
            packet[(p.paper_id, "panel_reviews")] = llm_packet(p.venue_key, p.paper_id, revs)
    by_id = papers.set_index("paper_id")
    client, model_of, lock = OpenRouterClient(CACHE, args.max_usd), models(), threading.Lock()
    jobs = [(pid, arm, m) for (pid, arm) in packet for m in args.chairs if (pid, arm, m) not in done]
    print(f"{discussion.VENUE}: {len(papers)} decided papers, {len(packet)} packets, {len(jobs)} decisions to run", flush=True)

    def work(job):
        pid, arm, key = job
        model, vkey = model_of[key], by_id.loc[pid, "venue_key"]
        overhead = len(area_chair.render_prompt(vkey, "", packet[(pid, arm)]).encode())
        text_in, trimmed = fit_to_context(paper_text(by_id.loc[pid]), model, overhead)
        prompt = area_chair.render_prompt(vkey, text_in, packet[(pid, arm)])
        cut_off = False
        for attempt in range(args.attempt_offset, args.attempt_offset + 3):
            max_tokens = (2 * area_chair.MAX_OUTPUT_TOKENS if attempt - args.attempt_offset == 2 and cut_off
                          else area_chair.MAX_OUTPUT_TOKENS)
            kwargs = dict(run_idx=args.repeat - 1, temperature=TEMPERATURE if model.accepts_temperature else None, max_tokens=max_tokens)
            if attempt:
                kwargs["attempt"] = attempt
            if model.request:
                kwargs["extra"] = model.request
            text, usage = client.complete(model.api_model, prompt, **kwargs)
            decision, meta_review = area_chair.parse_decision(text)
            rec = dict(paper_id=pid, venue_key=vkey, arm=arm, ac_model_id=key, decision=decision,
                       parse_ok=decision is not None, venue_decision=by_id.loc[pid, "outcome"], meta_review=meta_review,
                       raw_response=text, prompt_sha256=prompts.sha256(prompt), packet_sha256=prompts.sha256(packet[(pid, arm)]),
                       paper_text_context_trimmed=trimmed, attempt=attempt + 1, max_tokens=max_tokens,
                       model_snapshot=usage.get("execution_model"),
                       usage={k: usage.get(k) for k in ("input_tokens", "output_tokens", "cost_usd", "cached",
                                                        "finish_reason", "provider")},
                       timestamp=datetime.datetime.now(datetime.UTC).isoformat())
            with lock, records.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if rec["parse_ok"]:
                return rec
            cut_off = usage.get("finish_reason") == "length"
        return rec

    n = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for k, future in enumerate(as_completed(futures), start=1):
            try:
                n += future.result()["parse_ok"]
            except CostCeilingReached as error:
                print("stopping:", error, flush=True)
                break
            except (RuntimeError, OSError, ValueError, KeyError) as error:
                print(f"  error: {str(error)[:200]}", flush=True)
            if k % 100 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} decisions, {n} parsed, ${client.spent:.2f}", flush=True)
    print(f"done, spent ${client.spent:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    r.add_argument("--chairs", nargs="+", default=CHAIRS, choices=CHAIRS)
    r.add_argument("--repeat", type=int, default=1, choices=[1, 2], help="ICLR 2022: independent repeat run")
    r.add_argument("--workers", type=int, default=24)
    r.add_argument("--max-usd", type=float, required=True)
    r.add_argument("--attempt-offset", type=int, default=0,
                   help="number attempts from here, to draw fresh samples for decisions still unparsed")
    r.set_defaults(run=run)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
