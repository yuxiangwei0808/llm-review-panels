"""Notes: each reviewer revises its first review after reading three notes from "another reader".

The notes (field ``arm``), the same three for the eight reviewers of a paper:
  missed        human-raised weaknesses that none of the panel's first reviews raises
  mentioned     human-raised weaknesses that some first review already raises
  false         human-raised weaknesses of another paper (seeded pairing)
  fresh_missed  the missed notes given before a fresh review instead of a revision
  fresh_false   the false notes given before a fresh review
Papers: every decided paper with at least three missed and three mentioned weaknesses, by the coverage judge's labels
of the first reviews. The notes given per paper are in design_full.json or design.json in the output folder; `design`
writes design.json if neither exists. The venue is chosen with PANEL_CORPUS, as in experiments/discussion.py.

Usage:
  PANEL_CORPUS=iclr2021 uv run python experiments/notes.py run --max-usd 300
  PANEL_CORPUS=iclr2021 uv run python experiments/notes.py adoption --max-usd 20   # which notes the reviews take up
Writes: results/<corpus>/injection/records.jsonl and adoption_labels.csv (2023-25: results/study2325_panel8f/injection/).
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

FOLDER = ROOT / "results" / ("study2325_panel8f" if discussion.CORPUS == "2023-25" else f"{discussion.CORPUS}_prechatgpt")
OUT = FOLDER / "injection"
WEAKNESSES = FOLDER / "content" / "reference_items.jsonl"
LABELS = FOLDER / "content" / "coverage_labels.csv"      # the coverage judge's labels of the first reviews
CACHE = ROOT / "cache" / "responses" / "injection"
ARMS = ["missed", "mentioned", "false"]
FRESH = {"fresh_missed": "missed", "fresh_false": "false"}   # notes before a fresh review, and the notes they use
N_NOTES = 3
FRAME_SEED = f"{SEED}:e8"   # seeds the sampling and order of the notes (the string of the runs behind the paper)


def _weaknesses() -> dict[str, list[dict]]:
    weak: dict[str, list[dict]] = {}
    for line in WEAKNESSES.open():
        item = json.loads(line)
        if item["polarity"] == "weakness":
            weak.setdefault(item["paper_id"], []).append(item)
    return weak


def design() -> dict:
    """The notes per paper: the released design (design_full.json, else design.json), or a new one over every
    eligible paper, written to design.json."""
    for name in ("design_full.json", "design.json"):
        if (OUT / name).exists():
            return json.loads((OUT / name).read_text())
    manifest = {p["paper_id"]: p for p in json.loads(discussion.MANIFEST.read_text())["papers"]}
    weak = _weaknesses()
    labels = pd.read_csv(LABELS)
    labels = labels[labels.model.isin(discussion.PANEL) & labels.recognition.isin(["specific", "partial"])]
    mentioned = labels.groupby("paper_id").id.apply(set).to_dict()
    eligible = sorted(pid for pid, ws in weak.items()
                      if manifest.get(pid, {}).get("outcome") in ("accept", "reject")
                      and sum(w["id"] not in mentioned.get(pid, set()) for w in ws) >= N_NOTES
                      and sum(w["id"] in mentioned.get(pid, set()) for w in ws) >= N_NOTES)
    rng = random.Random(FRAME_SEED)
    partners = eligible[:]
    while any(a == b for a, b in zip(eligible, partners)):
        rng.shuffle(partners)
    out = {"eligible": len(eligible), "papers": {}}
    for pid, partner in zip(eligible, partners):
        hit = mentioned.get(pid, set())
        r = random.Random(f"{FRAME_SEED}:{pid}")
        missed = r.sample([w for w in weak[pid] if w["id"] not in hit], N_NOTES)
        known = r.sample([w for w in weak[pid] if w["id"] in hit], N_NOTES)
        false = r.sample(weak[partner], N_NOTES)
        out["papers"][pid] = {
            "partner": partner,
            "missed": [{"id": w["id"], "text": w["text"]} for w in missed],
            "mentioned": [{"id": w["id"], "text": w["text"]} for w in known],
            "false": [{"id": f"{partner}:{w['id']}", "text": w["text"]} for w in false]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "design.json").write_text(json.dumps(out, indent=1))
    return out


def notes_block(notes: list[dict], pid: str, arm: str) -> str:
    order = list(notes)
    random.Random(f"{FRAME_SEED}:{pid}:{arm}:order").shuffle(order)
    return "\n".join(f"- {n['text']}" for n in order)


def run(args):
    design_ = design()
    ids = sorted(design_["papers"])
    if args.pilot:
        ids = ids[:args.pilot]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(json.loads(discussion.MANIFEST.read_text())["papers"]).set_index("paper_id")
    reviews = discussion.round1(set(ids))
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
        venue = venues()[paper["venue_key"]] if "venue_key" in paper else venues()["iclr2022"]
        own = reviews[(pid, key)]
        source_arm = FRESH.get(arm, arm)
        notes = notes_block(design_["papers"][pid][source_arm], pid, source_arm)
        template = "call_a_fresh_notes" if arm in FRESH else "call_a_notes"
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
                       notes=design_["papers"][pid][source_arm], round1_call_id=own["call_id"],
                       round1_score=own["emitted_score"], emitted_score=parsed.score, items=parsed.items,
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
    """How many of its three notes each revised review raises, and its first review as the baseline: the coverage
    judge (same model and prompt as for the human-raised weaknesses) with the notes as the points."""
    design_ = design()
    annotator = models()[args.annotator]
    client = OpenRouterClient(CACHE.parent / f"content_match_notes_{args.annotator}", args.max_usd)
    recs = {(r["paper_id"], r["model_id"], r["arm"]): r for r in map(json.loads, (OUT / "records.jsonl").open())
            if r["parse_ok"] and r["items"]}
    reviews = discussion.round1(set(design_["papers"]))
    jobs = []
    for (pid, m, arm), r in recs.items():
        points = [{"id": f"N{k}", "polarity": "weakness", "text": n["text"]}
                  for k, n in enumerate(design_["papers"][pid][FRESH.get(arm, arm)])]
        jobs.append((pid, m, arm, "revised", points, r["items"], (r.get("extra") or {}).get("summary", "")))
        own = reviews.get((pid, m))
        if own:
            jobs.append((pid, m, arm, "round1", points, own["items"], (own.get("extra") or {}).get("summary", "")))

    def work(job):
        pid, m, arm, which, points, items, summary = job
        summary = " ".join(map(str, summary)) if isinstance(summary, list) else str(summary or "")
        rec = content.match_review(client, annotator, points, items, summary=summary, run_key=0)
        hits = sum(v[0] in ("specific", "partial") for v in rec["labels"].values()) if rec["parse_ok"] else None
        return dict(paper_id=pid, model=m, arm=arm, review=which, mentioned=hits, n_notes=len(points))

    with ThreadPoolExecutor(args.workers) as pool:
        rows = [x for x in pool.map(work, jobs) if x["mentioned"] is not None]
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "adoption_labels.csv", index=False)
    table["share"] = table.mentioned / table.n_notes
    out = table.pivot_table(index="arm", columns="review", values="share", aggfunc="mean")
    out["adopted (revised - round 1)"] = out.revised - out.round1
    print(f"{len(table)} labeled reviews, ${client.spent:.2f}")
    print((100 * out).round(1).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--pilot", type=int, default=0)
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
