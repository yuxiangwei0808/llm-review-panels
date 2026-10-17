"""Notes: each reviewer revises its first review after reading three notes from "another reader".

The notes (field ``arm``), the same three for the eight reviewers of a paper:
  missed             human-raised weaknesses that none of the panel's first reviews raises
  already_raised     human-raised weaknesses that some first review already raises
  other_paper        human-raised weaknesses of another paper (seeded pairing)
  fresh_missed       the missed notes given before a fresh review instead of a revision
  fresh_other_paper  the other paper's notes given before a fresh review
Papers: every decided paper with at least three missed and three already-raised weaknesses, by the coverage judge's
labels of the first reviews. The notes given per paper are in design.json in the output folder; `design` writes it if
it does not exist. The venue group is chosen with VENUE, as in experiments/discussion.py.

Usage:
  VENUE=iclr2021 uv run python experiments/notes.py run --max-usd 300
  VENUE=iclr2021 uv run python experiments/notes.py adoption --max-usd 20     # which notes the reviews take up
Writes: results/<group>/notes/records.jsonl and adoption_labels.csv.
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import discussion

from peerreview import content, prompts
from peerreview.client import CostCeilingReached, OpenRouterClient
from peerreview.config import MAX_OUTPUT_TOKENS, ROOT, SEED, TEMPERATURE, models, venues
from peerreview.papers import paper_text
from peerreview.runner import MAX_ATTEMPTS, fit_to_context

FOLDER = ROOT / "results" / discussion.VENUE
OUT = FOLDER / "notes"
WEAKNESSES = FOLDER / "content" / "human_points.jsonl"
LABELS = FOLDER / "content" / "coverage_labels.csv"      # the coverage judge's labels of the first reviews
CACHE = ROOT / "cache" / "responses" / "notes"
ARMS = ["missed", "already_raised", "other_paper"]
FRESH = {"fresh_missed": "missed", "fresh_other_paper": "other_paper"}   # notes before a fresh review, and their notes
N_NOTES = 3
NOTES_SEED = f"{SEED}:notes"   # seeds the choice of notes and the order they are shown in


def _weaknesses() -> dict[str, list[dict]]:
    weak: dict[str, list[dict]] = {}
    for line in WEAKNESSES.open():
        item = json.loads(line)
        if item["polarity"] == "weakness":
            weak.setdefault(item["paper_id"], []).append(item)
    return weak


def design() -> dict:
    """The notes per paper: the released design.json, or a new one over every eligible paper, written there."""
    if (OUT / "design.json").exists():
        return json.loads((OUT / "design.json").read_text())
    manifest = {p["paper_id"]: p for p in json.loads(discussion.MANIFEST.read_text())["papers"]}
    weak = _weaknesses()
    labels = pd.read_csv(LABELS)
    labels = labels[labels.model.isin(discussion.PANEL) & labels.recognition.isin(["specific", "partial"])]
    mentioned = labels.groupby("paper_id").id.apply(set).to_dict()
    eligible = sorted(pid for pid, ws in weak.items()
                      if manifest.get(pid, {}).get("outcome") in ("accept", "reject")
                      and sum(w["id"] not in mentioned.get(pid, set()) for w in ws) >= N_NOTES
                      and sum(w["id"] in mentioned.get(pid, set()) for w in ws) >= N_NOTES)
    rng = random.Random(NOTES_SEED)
    partners = eligible[:]
    while any(a == b for a, b in zip(eligible, partners)):
        rng.shuffle(partners)
    out = {"eligible": len(eligible), "papers": {}}
    for pid, partner in zip(eligible, partners):
        hit = mentioned.get(pid, set())
        r = random.Random(f"{NOTES_SEED}:{pid}")
        missed = r.sample([w for w in weak[pid] if w["id"] not in hit], N_NOTES)
        known = r.sample([w for w in weak[pid] if w["id"] in hit], N_NOTES)
        false = r.sample(weak[partner], N_NOTES)
        out["papers"][pid] = {
            "partner": partner,
            "missed": [{"id": w["id"], "text": w["text"]} for w in missed],
            "already_raised": [{"id": w["id"], "text": w["text"]} for w in known],
            "other_paper": [{"id": f"{partner}:{w['id']}", "text": w["text"]} for w in false]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "design.json").write_text(json.dumps(out, indent=1))
    return out


def notes_block(notes: list[dict], pid: str, arm: str) -> str:
    order = list(notes)
    random.Random(f"{NOTES_SEED}:{pid}:{arm}:order").shuffle(order)
    return "\n".join(f"- {n['text']}" for n in order)


def run(args):
    design_ = design()
    ids = sorted(design_["papers"])
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(json.loads(discussion.MANIFEST.read_text())["papers"]).set_index("paper_id")
    reviews = discussion.first_reviews(set(ids))
    model_of = models()
    client = OpenRouterClient(CACHE, args.max_usd)
    record_path = OUT / "records.jsonl"
    done, fails = set(), {}
    if record_path.exists():
        for r in map(json.loads, record_path.open()):
            key = (r["paper_id"], r["model_id"], r["arm"])
            if r["parse_ok"]:
                done.add(key)
            else:
                fails[key] = fails.get(key, 0) + 1
    jobs = [(pid, m, a) for pid in ids for m in discussion.PANEL for a in args.arms
            if (pid, m, a) not in done and (pid, m) in reviews]
    print(f"{len(ids)} papers, {len(jobs)} cells to run ({len(done)} done)", flush=True)
    lock = threading.Lock()

    def work(job):
        pid, key, arm = job
        model, paper = model_of[key], manifest.loc[pid]
        venue = venues()[paper["venue_key"]]
        own = reviews[(pid, key)]
        source_arm = FRESH.get(arm, arm)
        notes = notes_block(design_["papers"][pid][source_arm], pid, source_arm)
        template = "notes_fresh_review" if arm in FRESH else "notes_revision"
        fields = dict(notes=notes) if arm in FRESH else dict(
            source_review=discussion.review_text(own), initial_score=str(own["emitted_score"]), notes=notes)

        def render(text):
            return prompts.render(template, own_review_prompt=prompts.own_review(venue, text), **fields)

        text_in, trimmed = fit_to_context(paper_text(paper), model, len(render("").encode()))
        prompt = render(text_in)
        temperature = TEMPERATURE if model.accepts_temperature else None
        first = fails.get((pid, key, arm), 0)
        for attempt in range(first, first + MAX_ATTEMPTS):
            kwargs = dict(run_idx=0, temperature=temperature, max_tokens=MAX_OUTPUT_TOKENS)
            if attempt:
                kwargs["attempt"] = attempt
            if model.request:
                kwargs["extra"] = model.request
            started = time.perf_counter()
            raw, usage = client.complete(model.api_model, prompt, **kwargs)
            parsed = discussion.parse_round2(raw, venue.scores)
            rec = dict(paper_id=pid, model_id=key, arm=arm, prompt_template=template,
                       prompt_version=prompts.load_template(template)[0]["version"], prompt_sha256=prompts.sha256(prompt),
                       notes=design_["papers"][pid][source_arm], first_review_call_id=own["call_id"],
                       first_review_score=own["emitted_score"], emitted_score=parsed.score, items=parsed.items,
                       reasoning_text=parsed.reasoning, raw_response=raw, parse_ok=parsed.ok,
                       parse_error=parsed.error, extra=parsed.extra, paper_text_context_trimmed=trimmed,
                       temperature=temperature, attempt=attempt + 1, execution_model=usage.get("execution_model"),
                       usage={k: usage.get(k) for k in ("input_tokens", "output_tokens", "reasoning_tokens",
                                                        "cost_usd", "cached", "finish_reason", "provider")},
                       latency_s=round(time.perf_counter() - started, 3),
                       timestamp=datetime.datetime.now(datetime.UTC).isoformat())
            with lock, record_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if parsed.ok:
                return rec
        return rec

    n_ok = n_fail = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for k, future in enumerate(as_completed(futures), start=1):
            try:
                rec = future.result()
                n_ok += rec["parse_ok"]
                n_fail += not rec["parse_ok"]
            except CostCeilingReached as error:
                print("stopping:", error, flush=True)
                break
            except (RuntimeError, OSError, ValueError, KeyError) as error:
                n_fail += 1
                print(f"  error: {str(error)[:200]}", flush=True)
            if k % 25 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} cells, {n_ok} parsed, {n_fail} failed, ${client.spent:.2f}", flush=True)
    print(f"done: {n_ok} parsed, {n_fail} failed, spent ${client.spent:.2f}", flush=True)


def adoption(args):
    """How many of its three notes each revised review raises, with its first review as the baseline: the coverage
    judge (same model and prompt as for the human-raised weaknesses) with the notes as the points. Appends to
    adoption_labels.csv; reviews whose labels did not parse before are sampled afresh."""
    design_ = design()
    annotator = models()[args.annotator]
    client = OpenRouterClient(CACHE.parent / f"content_match_notes_{args.annotator}", args.max_usd)
    path = OUT / "adoption_labels.csv"
    done = set(map(tuple, pd.read_csv(path)[["paper_id", "model", "arm", "review"]].values)) if path.exists() else set()
    offset = 3 if path.exists() else 0      # a review missing after an earlier run failed its three attempts
    recs = {(r["paper_id"], r["model_id"], r["arm"]): r for r in map(json.loads, (OUT / "records.jsonl").open())
            if r["parse_ok"] and r.get("items") and r["arm"] in ARMS}
    reviews = discussion.first_reviews(set(design_["papers"]))
    jobs = []
    for (pid, m, arm), r in recs.items():
        points = [{"id": f"N{k}", "polarity": "weakness", "text": n["text"]}
                  for k, n in enumerate(design_["papers"][pid][FRESH.get(arm, arm)])]
        jobs.append((pid, m, arm, "revised", points, r["items"], (r.get("extra") or {}).get("summary", "")))
        own = reviews.get((pid, m))
        if own:
            jobs.append((pid, m, arm, "first_review", points, own["items"], (own.get("extra") or {}).get("summary", "")))
    jobs = [j for j in jobs if j[:4] not in done]
    print(f"{len(done)} reviews labelled, {len(jobs)} to do", flush=True)

    def work(job):
        pid, m, arm, which, points, items, summary = job
        summary = " ".join(map(str, summary)) if isinstance(summary, list) else str(summary or "")
        rec = content.match_review(client, annotator, points, items, summary=summary, run_key=0, attempt_offset=offset)
        hits = sum(v[0] in ("specific", "partial") for v in rec["labels"].values()) if rec["parse_ok"] else None
        return dict(paper_id=pid, model=m, arm=arm, review=which, mentioned=hits, n_notes=len(points))

    with ThreadPoolExecutor(args.workers) as pool:
        rows = [x for x in pool.map(work, jobs) if x["mentioned"] is not None]
    pd.DataFrame(rows, columns=["paper_id", "model", "arm", "review", "mentioned", "n_notes"]).to_csv(
        path, mode="a", header=not path.exists(), index=False)
    table = pd.read_csv(path)
    table["share"] = table.mentioned / table.n_notes
    out = table.pivot_table(index="arm", columns="review", values="share", aggfunc="mean")
    out["adopted (revised - first review)"] = out.revised - out.first_review
    print(f"{len(table)} labelled reviews, ${client.spent:.2f}")
    print((100 * out).round(1).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=ARMS + list(FRESH), choices=ARMS + list(FRESH))
    r.add_argument("--workers", type=int, default=24)
    r.add_argument("--max-usd", type=float, required=True)
    r.set_defaults(run=run)
    a = sub.add_parser("adoption")
    a.add_argument("--annotator", default="gemini-3.5-flash-lite", choices=sorted(models()))
    a.add_argument("--workers", type=int, default=24)
    a.add_argument("--max-usd", type=float, required=True)
    a.set_defaults(run=adoption)
    d = sub.add_parser("design")
    d.set_defaults(run=lambda args: print(json.dumps({k: v for k, v in design().items() if k != "papers"})))
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
