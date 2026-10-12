"""Content analysis for an experiment: human points, recognition labels, omission curve (Fig. 1b).

Runs the rebuilt pipeline (``peerreview.content``) on the papers of an experiment's manifest:

  extract   human points per paper from the dossiers          -> content/reference_items.jsonl
            (every call's raw response, usage and attempt      -> content/extract_records.jsonl)
  match     every own review (call A) of every model x run against the paper's points
                                                                -> content/coverage_labels.csv
            (raw responses                                      -> content/match_records.jsonl)
  curve     pooled omission curve of the weaknesses, exact over all size-n review subsets,
            with 2,000 paper-bootstrap percentile intervals   -> content/omission_curve.csv

Outputs go to results/<experiment>/content/. Steps resume: extracted papers and labeled reviews
already on disk are skipped, and cached responses are free.

Usage:
  PEERREVIEW_EXPERIMENT=iclr2022_prechatgpt uv run python experiments/judge.py extract \
      --annotator gpt-5.6-luna
  PEERREVIEW_EXPERIMENT=iclr2022_prechatgpt uv run python experiments/judge.py match \
      --annotator gemini-3.5-flash-lite --tag prechatgpt
  PEERREVIEW_EXPERIMENT=iclr2022_prechatgpt uv run python experiments/judge.py curve
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peerreview import content, records
from peerreview.analysis.experiment import current
N_BOOT = 2000   # paper-bootstrap replicates
from peerreview.client import OpenRouterClient
from peerreview.config import ROOT, models
from peerreview.papers import load_dossier

EXPERIMENT = current()
OUT = ROOT / "results" / EXPERIMENT.name / "content"          # roster-specific outputs (curve)


def _content_from() -> Path:
    """Where the roster-independent tables live: an experiment may point at another one's
    (``content_from:`` in its YAML), e.g. the study-roster analysis of the same ICLR 2022 run."""
    import yaml
    spec_path = ROOT / "configs" / "experiments" / f"{EXPERIMENT.name}.yaml"
    if spec_path.exists():
        source = yaml.safe_load(spec_path.read_text()).get("content_from")
        if source:
            return ROOT / "results" / source / "content"
    return OUT


SHARED = _content_from()
CURVE_SEED = 20260928


def papers() -> pd.DataFrame:
    table = pd.DataFrame(json.loads(EXPERIMENT.manifest.read_text())["papers"])
    return table[table.venue_key.isin(EXPERIMENT.venues)]


def load_points() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    path = SHARED / "reference_items.jsonl"
    if path.exists():
        for line in path.open():
            item = json.loads(line)
            out.setdefault(item["paper_id"], []).append(item)
    return out


def extract(args):
    annotator = models()[args.annotator]
    client = OpenRouterClient(ROOT / "cache" / "responses" / f"content_extract_{args.annotator}",
                              args.max_usd)
    done = load_points()
    todo = [p for p in papers().itertuples() if p.paper_id not in done]
    print(f"{len(done)} papers extracted, {len(todo)} to do", flush=True)
    SHARED.mkdir(parents=True, exist_ok=True)
    failures = []

    def work(paper):
        return paper.paper_id, content.extract_points(client, annotator,
                                                      load_dossier(paper.venue_key, paper.paper_id))

    with ThreadPoolExecutor(args.workers) as pool, (SHARED / "reference_items.jsonl").open("a") as out, \
            (SHARED / "extract_records.jsonl").open("a") as log:
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


def backfill_extract_records(args):
    """Write extract_records.jsonl entries for papers extracted before records were kept, from the
    response cache (keyed by the rendered prompt)."""
    import hashlib

    from peerreview import prompts as _prompts
    annotator = models()[args.annotator]
    cache = ROOT / "cache" / "responses" / f"content_extract_{args.annotator}"
    path = SHARED / "extract_records.jsonl"
    have = set()
    if path.exists():
        have = {json.loads(line)["paper_id"] for line in path.open()}
    todo = [p for p in papers().itertuples() if p.paper_id in load_points() and p.paper_id not in have]
    written = 0
    with path.open("a") as log:
        for paper in todo:
            prompt = content.extract_prompt(load_dossier(paper.venue_key, paper.paper_id))
            for attempt in range(3):
                request = dict(model=annotator.api_model, prompt=prompt, temperature=content.TEMPERATURE,
                               max_tokens=content.EXTRACT_MAX_OUTPUT_TOKENS, run_idx=0,
                               extra={**annotator.request, "response_format": {"type": "json_object"}})
                if attempt:
                    request["attempt"] = attempt
                key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
                cached = cache / f"{key}.json"
                if not cached.exists():
                    break
                saved = json.loads(cached.read_text())
                points, error = content.parse_points(saved["text"])
                log.write(json.dumps(dict(
                    paper_id=paper.paper_id, timestamp=_now(), backfilled_from_cache=True,
                    points=points, parse_ok=error is None, parse_error=error,
                    raw_response=saved["text"], usage=saved["usage"], attempt=attempt + 1,
                    prompt_sha256=_prompts.sha256(prompt), annotator=annotator.key)) + "\n")
                written += 1
                if error is None:
                    break
    print(f"backfilled {written} extraction records for {len(todo)} papers")


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.UTC).isoformat()


def match(args):
    annotator = models()[args.annotator]
    client = OpenRouterClient(ROOT / "cache" / "responses" / f"content_match_{args.annotator}",
                              args.max_usd)
    points = load_points()
    path = SHARED / "coverage_labels.csv"
    done = set()
    if path.exists():
        prior = pd.read_csv(path)
        done = set(zip(prior.paper_id, prior.model, prior.run))
    prior_failures: dict[tuple, int] = {}
    records_path = SHARED / "match_records.jsonl"
    if records_path.exists():
        for line in records_path.open():
            r = json.loads(line)
            if not r["parse_ok"]:
                key = (r["paper_id"], r["model"], r["run"])
                prior_failures[key] = max(prior_failures.get(key, 0), r.get("attempt", 1))
    reviews = []
    for venue in EXPERIMENT.venues:
        for model in EXPERIMENT.models:
            latest: dict[tuple, dict] = {}
            try:
                for r in records.iter_records(model, venue, "paper_review", "A", tag=args.tag):
                    if r["parse_ok"] and r["items"] and r["paper_id"] in points:
                        latest[(r["paper_id"], r["run_idx"])] = r        # last parsed record wins
            except FileNotFoundError:   # e.g. GPT-5.6 Terra's NeurIPS 2023 reviews were never archived
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
            (SHARED / "match_records.jsonl").open("a") as log:
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


def curve(args):
    points = load_points()
    labels = pd.read_csv(SHARED / "coverage_labels.csv")
    labels = labels[labels.model.isin(EXPERIMENT.models)]
    OUT.mkdir(parents=True, exist_ok=True)
    recognized = labels[labels.recognition.isin(["specific", "partial"])]
    n_reviews = len(EXPERIMENT.models) * 3
    complete = labels.groupby("paper_id").review_id.nunique()
    paper_ids = sorted(complete[complete == n_reviews].index)
    print(f"{len(paper_ids)} papers with all {n_reviews} reviews labeled "
          f"(of {labels.paper_id.nunique()})")
    per_paper = []           # per paper: list of recognizing counts of its weaknesses
    for pid in paper_ids:
        weak = [p["id"] for p in points[pid] if p["polarity"] == "weakness"]
        counts = recognized[(recognized.paper_id == pid) & recognized.id.isin(weak)] \
            .groupby("id").review_id.nunique()
        per_paper.append([int(counts.get(i, 0)) for i in weak])
    pooled = [c for cs in per_paper for c in cs]
    point = np.array(content.omission_curve(pooled, n_reviews))
    rng = np.random.default_rng(CURVE_SEED)
    boots = []
    for _ in range(N_BOOT):
        draw = rng.integers(0, len(per_paper), len(per_paper))
        boots.append(content.omission_curve([c for i in draw for c in per_paper[i]], n_reviews))
    boots = np.array(boots)
    table = pd.DataFrame(dict(
        reviews=np.arange(1, n_reviews + 1), missed_pct=100 * point,
        ci_low=100 * np.percentile(boots, 2.5, axis=0), ci_high=100 * np.percentile(boots, 97.5, axis=0),
        n_papers=len(per_paper), n_items=len(pooled), reviews_per_paper=n_reviews))
    table.to_csv(OUT / "omission_curve.csv", index=False)
    print(table.iloc[[0, 1, 3, 7, n_reviews // 2 - 1, n_reviews - 1]].round(1).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract"); e.add_argument("--annotator", required=True, choices=sorted(models()))
    e.add_argument("--workers", type=int, default=6); e.add_argument("--max-usd", type=float, default=20)
    e.set_defaults(run=extract)
    m = sub.add_parser("match"); m.add_argument("--annotator", required=True, choices=sorted(models()))
    m.add_argument("--tag", required=True, help="run tag of the own reviews (res/<model>/<venue>/paper_review_<tag>)")
    m.add_argument("--workers", type=int, default=8); m.add_argument("--max-usd", type=float, default=40)
    m.set_defaults(run=match)
    c = sub.add_parser("curve"); c.set_defaults(run=curve)
    b = sub.add_parser("backfill-extract-records")
    b.add_argument("--annotator", required=True, choices=sorted(models())); b.set_defaults(run=backfill_extract_records)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
