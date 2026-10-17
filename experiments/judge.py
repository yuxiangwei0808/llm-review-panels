"""Weakness extraction and the coverage judge on the first reviews of a venue group (VENUE, as in experiments/discussion.py).

  extract   the distinct points raised by each paper's human reviews (prompt extract_points)
            -> human_points.jsonl, with every call in extract_records.jsonl
  match     every first review of every model and run, labelled against the paper's points: raised specifically,
            partially, or not (prompt coverage_judge)  -> coverage_labels.csv, match_records.jsonl

Outputs go to results/<group>/content/. Steps resume: extracted papers and labelled reviews already on disk are skipped,
and earlier failures are sampled afresh. The human reviews (dossiers under data/dossiers/) are not distributed.

Usage:
  VENUE=iclr2022 uv run python experiments/judge.py extract --annotator gpt-5.6-luna
  VENUE=iclr2022 uv run python experiments/judge.py match --annotator gemini-3.5-flash-lite
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peerreview import content, records
from peerreview.client import OpenRouterClient
from peerreview.config import GROUPS, PANEL, ROOT, models
from peerreview.papers import load_dossier

VENUE = OUT = None     # set in main() from the VENUE environment variable


def papers() -> pd.DataFrame:
    table = pd.DataFrame(json.loads((ROOT / "data" / "manifests" / f"{VENUE}.json").read_text())["papers"])
    return table[table.venue_key.isin(GROUPS[VENUE])]


def load_points() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    path = OUT / "human_points.jsonl"
    if path.exists():
        for line in path.open():
            item = json.loads(line)
            out.setdefault(item["paper_id"], []).append(item)
    return out


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def extract(args):
    annotator = models()[args.annotator]
    client = OpenRouterClient(ROOT / "cache" / "responses" / f"content_extract_{args.annotator}",
                              args.max_usd)
    done = load_points()
    todo = [p for p in papers().itertuples() if p.paper_id not in done]
    print(f"{len(done)} papers extracted, {len(todo)} to do", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    failures = []
    failed: dict[str, int] = {}     # earlier failed attempts per paper: a rerun draws fresh samples
    if (OUT / "extract_records.jsonl").exists():
        for line in (OUT / "extract_records.jsonl").open():
            r = json.loads(line)
            if not r["parse_ok"]:
                failed[r["paper_id"]] = max(failed.get(r["paper_id"], 0), r.get("attempt", 1))

    def work(paper):
        return paper.paper_id, content.extract_points(client, annotator, load_dossier(paper.venue_key, paper.paper_id),
                                                      attempt_offset=failed.get(paper.paper_id, 0))

    with ThreadPoolExecutor(args.workers) as pool, (OUT / "human_points.jsonl").open("a") as out, \
            (OUT / "extract_records.jsonl").open("a") as log:
        for k, future in enumerate(as_completed([pool.submit(work, p) for p in todo]), start=1):
            pid, rec = future.result()
            log.write(json.dumps(dict(paper_id=pid, timestamp=_now(), **rec)) + "\n")
            log.flush()
            if rec["parse_ok"]:
                for point in rec["points"]:
                    out.write(json.dumps(dict(paper_id=pid, annotation_job=f"extract_{args.annotator}",
                                              prompt_sha256=rec["prompt_sha256"], **point)) + "\n")
                out.flush()
            else:
                failures.append(pid)
            if k % 25 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} papers, ${client.spent:.2f}, {len(failures)} failures",
                      flush=True)
    if failures:
        print("failed papers (rerun to retry):", failures)


def match(args):
    annotator = models()[args.annotator]
    client = OpenRouterClient(ROOT / "cache" / "responses" / f"content_match_{args.annotator}",
                              args.max_usd)
    points = load_points()
    path = OUT / "coverage_labels.csv"
    done = set()
    if path.exists():
        prior = pd.read_csv(path)
        done = set(zip(prior.paper_id, prior.model, prior.run))
    prior_failures: dict[tuple, int] = {}
    records_path = OUT / "match_records.jsonl"
    if records_path.exists():
        for line in records_path.open():
            r = json.loads(line)
            if not r["parse_ok"]:
                key = (r["paper_id"], r["model"], r["run"])
                prior_failures[key] = max(prior_failures.get(key, 0), r.get("attempt", 1))
    reviews = []
    for venue in GROUPS[VENUE]:
        for model in args.models:
            latest: dict[tuple, dict] = {}
            try:
                for r in records.iter_records(model, venue):
                    if r["parse_ok"] and r["items"] and r["paper_id"] in points:
                        latest[(r["paper_id"], r["run_idx"])] = r        # last parsed record wins
            except FileNotFoundError:
                print(f"  no reviews for {model} on {venue}", flush=True)
            reviews += [(model, r) for r in latest.values()
                        if (r["paper_id"], model, r["run_idx"]) not in done]
    print(f"{len(done)} reviews labeled, {len(reviews)} to do", flush=True)

    def work(job):
        model, r = job
        rec = content.match_review(client, annotator, points[r["paper_id"]], r["items"],
                                   summary=r["extra"].get("summary", ""), run_key=r["run_idx"],
                                   attempt_offset=prior_failures.get((r["paper_id"], model, r["run_idx"]), 0))
        return model, r, rec

    header = not path.exists()
    failures = 0
    with ThreadPoolExecutor(args.workers) as pool, path.open("a") as out, \
            (OUT / "match_records.jsonl").open("a") as log:
        if header:
            out.write("paper_id,review_id,model,run,annotation_job,id,recognition,stance\n")
        for k, future in enumerate(as_completed([pool.submit(work, j) for j in reviews]), start=1):
            model, r, rec = future.result()
            log.write(json.dumps(dict(paper_id=r["paper_id"], model=model, run=r["run_idx"],
                                      review_call_id=r["call_id"], timestamp=_now(), **rec)) + "\n")
            log.flush()
            if not rec["parse_ok"]:
                failures += 1
            else:
                for point_id, (recognition, stance) in rec["labels"].items():
                    out.write(f"{r['paper_id']},M:{r['paper_id']}:{model}:{r['run_idx']},{model},"
                              f"{r['run_idx']},match_{args.annotator},{point_id},{recognition},"
                              f"{stance}\n")
                out.flush()
            if k % 100 == 0 or k == len(reviews):
                print(f"  {k}/{len(reviews)} reviews, ${client.spent:.2f}, {failures} failures",
                      flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--annotator", required=True, choices=sorted(models()))
    e.add_argument("--workers", type=int, default=6)
    e.add_argument("--max-usd", type=float, default=20)
    e.set_defaults(run=extract)
    m = sub.add_parser("match")
    m.add_argument("--annotator", required=True, choices=sorted(models()))
    m.add_argument("--models", nargs="+", default=PANEL, choices=sorted(models()))
    m.add_argument("--workers", type=int, default=8)
    m.add_argument("--max-usd", type=float, default=40)
    m.set_defaults(run=match)
    args = parser.parse_args()
    global VENUE, OUT
    VENUE = os.environ.get("VENUE")
    if VENUE not in GROUPS:
        raise SystemExit(f"set VENUE to one of {', '.join(GROUPS)}")
    OUT = ROOT / "results" / VENUE / "content"
    args.run(args)


if __name__ == "__main__":
    main()
