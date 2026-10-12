"""Human counterpart of the omission curve, computed the same way as the LLM curve.

Labels: for every paper and every human reviewer h (held out), the same matcher that judges LLM
reviews judges whether each other reviewer's text mentions each weakness h raised. From these
labels, the reported curve uses the same definition as the LLM curve: a committee of n human
reviewers is scored against ALL human-raised weaknesses of the paper (a weakness is covered if a
member raised it or mentions it), exact over all size-n committees, n = 1..4 on papers with exactly
four reviewers (the large majority), with 2,000 paper-bootstrap intervals. At n = 4 the committee
wrote the reference and misses nothing by construction. The per-held-out-reviewer variant (each
reviewer's weaknesses as the reference, weighted once per raising reviewer) is kept as a column.
``--recompute`` rebuilds the curves from the saved labels without any model calls.

Usage:
  uv run python experiments/human_coverage.py --annotator gemini-3.5-flash-lite            # the paper's 332 papers
  PEERREVIEW_EXPERIMENT=iclr2022_prechatgpt uv run python experiments/human_coverage.py --annotator gemini-3.5-flash-lite
Writes: results/content_rebuild/human_omission_matched_paper.{csv,json} or
        results/<experiment>/content/human_omission_curve.{csv,json}, plus the per-pair labels.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peerreview import content
from peerreview.analysis.experiment import current
from peerreview.client import OpenRouterClient
from peerreview.config import ROOT, models
from peerreview.papers import load_dossier

CONTENT = ROOT / "data" / "processed" / "content"
N_BOOT, SEED, MAX_POOL, REVIEWERS = 2000, 20260928, 4, 4   # papers with exactly 4 reviewers, n = 1..4


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def sections_by_letter(dossier: dict) -> dict[str, str]:
    parts = re.split(r"\n## Reviewer ([A-Z])\n", dossier["holistic_text"])
    return {parts[k]: parts[k + 1].strip() for k in range(1, len(parts), 2)}


def review_texts(dossier: dict, points: list[dict], frozen: bool) -> dict[str, str]:
    """Human review text per source id (H00, ...). The frozen items' ids follow an order that is
    not recorded anywhere, so it is recovered per paper from the quotes; the rebuilt extractor
    uses reviewer-letter order by construction."""
    secs = sections_by_letter(dossier)
    if not frozen:
        return {f"H{ord(letter) - ord('A'):02d}": text for letter, text in secs.items()}
    normed = {letter: norm(text) for letter, text in secs.items()}
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for p in points:
        for s in p["sources"]:
            quote = norm(s["quote"])[:50]
            if quote:
                for letter, text in normed.items():
                    if quote in text:
                        votes[s["review"]][letter] += 1
    out = {}
    for hid, counter in votes.items():
        (best, n), *rest = counter.most_common()
        if not rest or n > 2 * rest[0][1]:
            out[hid] = secs[best]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotator", default="gemini-3.5-flash-lite", choices=sorted(models()))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-usd", type=float, default=15.0)
    parser.add_argument("--recompute", action="store_true",
                        help="rebuild the curves from the saved labels; no model calls")
    args = parser.parse_args()
    experiment = current()
    frozen = experiment.is_paper

    if frozen:
        inventory = pd.read_csv(CONTENT / "paper_inventory.csv").set_index("paper_id")
        venue_of = inventory.venue.to_dict()
        items_path, labels_path = CONTENT / "reference_items.jsonl", CONTENT / "coverage_labels.csv"
        out_dir, stem = ROOT / "results" / "content_rebuild", "human_omission_matched_paper"
    else:
        manifest = json.loads(experiment.manifest.read_text())["papers"]
        venue_of = {p["paper_id"]: p["venue_key"] for p in manifest}
        out_dir = ROOT / "results" / experiment.name / "content"
        items_path, labels_path = out_dir / "reference_items.jsonl", out_dir / "coverage_labels.csv"
        stem = "human_omission_curve"
    points: dict[str, list[dict]] = {}
    for line in items_path.open():
        item = json.loads(line)
        points.setdefault(item["paper_id"], []).append(item)

    annotator = models()[args.annotator]
    client = OpenRouterClient(ROOT / "cache" / "responses" / f"content_human_{args.annotator}",
                              args.max_usd)
    jobs, skipped = [], collections.Counter()
    for pid, pts in points.items():
        texts = review_texts(load_dossier(venue_of[pid], pid), pts, frozen)
        weak = [p for p in pts if p["polarity"] == "weakness"]
        for hid in sorted(texts):
            mine = [p for p in weak if any(s["review"] == hid for s in p["sources"])]
            if not mine:
                skipped["reviewer without weaknesses"] += 1
                continue
            for other, text in texts.items():
                if other != hid:
                    jobs.append((pid, hid, other, mine, text))
    print(f"{len(points)} papers, {len(jobs)} (held-out reviewer, other review) pairs; skipped {dict(skipped)}",
          flush=True)

    def work(job):
        pid, hid, other, mine, text = job
        rec = content.match_review(client, annotator, mine, None, review_text=text)
        return pid, hid, other, rec

    rows, failures = [], 0
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.recompute:
        labels = pd.read_csv(out_dir / f"{stem}_labels.csv")
        jobs = []
    with ThreadPoolExecutor(args.workers) as pool, \
            (out_dir / f"{stem}_records.jsonl").open("a" if args.recompute else "w") as log:
        for k, future in enumerate(as_completed([pool.submit(work, j) for j in jobs]), start=1):
            pid, hid, other, rec = future.result()
            log.write(json.dumps(dict(paper_id=pid, held_out=hid, other=other, **rec)) + "\n")
            if not rec["parse_ok"]:
                failures += 1
            for point_id, (recognition, stance) in rec["labels"].items():
                rows.append(dict(paper_id=pid, held_out=hid, other=other, id=point_id,
                                 recognition=recognition, stance=stance))
            if k % 500 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} pairs, ${client.spent:.2f}, {failures} parse failures", flush=True)
    if not args.recompute:
        labels = pd.DataFrame(rows)
        labels.to_csv(out_dir / f"{stem}_labels.csv", index=False)

    # Human curve: per (paper, held-out reviewer, weakness): m other reviews, r recognizing it.
    n_reviews = {pid: len(sections_by_letter(load_dossier(venue_of[pid], pid))) for pid in points}
    recognized = labels[labels.recognition.isin(["specific", "partial"])]
    r_count = recognized.groupby(["paper_id", "held_out", "id"]).other.nunique()
    per_paper: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for (pid, hid), group in labels.groupby(["paper_id", "held_out"]):
        m = n_reviews[pid] - 1
        for point_id in group.id.unique():
            per_paper[pid].append((m, int(r_count.get((pid, hid, point_id), 0))))

    def human_curve(pairs):
        curve = []
        for n in range(1, 4):
            vals = [comb(m - r, n) / comb(m, n) if m - r >= n else 0.0 for m, r in pairs if m >= n]
            curve.append(np.mean(vals) if vals else np.nan)
        return np.array(curve)

    # LLM curve with the same held-out weighting: each weakness once per reviewer who raised it.
    llm = pd.read_csv(labels_path)
    llm = llm[llm.model.isin(experiment.models)] if not frozen else llm
    llm_rec = llm[llm.recognition.isin(["specific", "partial"])]
    n_llm = llm.groupby("paper_id").review_id.nunique()
    llm_pairs: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    llm_union: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    llm_counts = llm_rec.groupby(["paper_id", "id"]).review_id.nunique()
    for pid, pts in points.items():
        if pid not in n_llm.index:
            continue
        for p in pts:
            if p["polarity"] != "weakness":
                continue
            r = int(llm_counts.get((pid, p["id"]), 0))
            llm_union[pid].append((int(n_llm[pid]), r))
            for _ in {s["review"] for s in p["sources"]}:
                llm_pairs[pid].append((int(n_llm[pid]), r))

    def llm_curve(by_paper, sizes=(1, 2, 3, 4, 8, 12, 16, 24)):
        pairs = [x for v in by_paper.values() for x in v]
        return {n: float(np.mean([comb(m - r, n) / comb(m, n) if m - r >= n else 0.0
                                  for m, r in pairs if m >= n])) for n in sizes}

    heldout_point = human_curve([x for v in per_paper.values() for x in v])

    # Reported curve, the same definition as the LLM curve: an n-reviewer committee is scored
    # against ALL human-raised weaknesses of the paper. A weakness is covered if a committee member
    # raised it or (per the matcher) mentions it; at n = R the committee wrote the reference and
    # the miss rate is 0 by construction. Exact over all committees of a paper, then over papers.
    import itertools
    recognized = set(zip(labels.paper_id, labels.other, labels.id)) & set(zip(
        labels[labels.recognition.isin(["specific", "partial"])].paper_id,
        labels[labels.recognition.isin(["specific", "partial"])].other,
        labels[labels.recognition.isin(["specific", "partial"])].id))
    reviewers_of: dict[str, set] = collections.defaultdict(set)
    for pid, h, o in zip(labels.paper_id, labels.held_out, labels.other):
        reviewers_of[pid].update([h, o])
    leave_out: dict[str, list[float]] = {}          # paper -> mean miss share for n = 1..MAX_POOL
    for pid, pts in points.items():
        reviewers = sorted(reviewers_of[pid])
        if len(reviewers) != REVIEWERS:          # exactly four reviewers: n = 4 pools all of them
            continue
        weak = [p for p in pts if p["polarity"] == "weakness"]
        raised_by = {p["id"]: {s["review"] for s in p["sources"]} & set(reviewers) for p in weak}
        curve = []
        for n in range(1, MAX_POOL + 1):
            shares = []
            for committee in itertools.combinations(reviewers, n):
                members = set(committee)
                if not weak:
                    continue
                missed = sum(1 for p in weak if not (raised_by[p["id"]] & members)
                             and not any((pid, c, p["id"]) in recognized for c in members))
                shares.append(missed / len(weak))
            curve.append(np.mean(shares) if shares else np.nan)
        leave_out[pid] = curve
    ids_lo = sorted(leave_out)
    matrix = np.array([leave_out[pid] for pid in ids_lo])
    point = np.nanmean(matrix, axis=0)
    rng = np.random.default_rng(SEED)
    boots = np.array([np.nanmean(matrix[rng.integers(0, len(ids_lo), len(ids_lo))], axis=0)
                      for _ in range(N_BOOT)])
    table = pd.DataFrame(dict(reviews=np.arange(1, MAX_POOL + 1), missed_pct=100 * point,
                              ci_low=100 * np.nanpercentile(boots, 2.5, axis=0),
                              ci_high=100 * np.nanpercentile(boots, 97.5, axis=0),
                              missed_pct_per_heldout_reviewer=list(100 * heldout_point) + [np.nan],
                              n_papers=len(ids_lo), judge=args.annotator,
                              definition="n-reviewer committee vs. all human-raised weaknesses; papers with exactly 4 reviewers"))
    table.to_csv(out_dir / f"{stem}.csv", index=False)
    summary = dict(annotator=args.annotator, papers=len(ids_lo), pairs=len(jobs), parse_failures=failures,
                   human_missed_pct={n + 1: round(100 * v, 2) for n, v in enumerate(point)},
                   human_missed_pct_per_heldout_reviewer={n + 1: round(100 * v, 2) for n, v in enumerate(heldout_point)},
                   llm_missed_pct_heldout_weighting={k: round(100 * v, 2) for k, v in llm_curve(llm_pairs).items()},
                   llm_missed_pct_union_weighting={k: round(100 * v, 2) for k, v in llm_curve(llm_union).items()},
                   spent_usd=round(client.spent, 4))
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
