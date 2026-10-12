"""LLM area chairs on ICLR 2021, NeurIPS 2022 and the 2023-25 venues (2026-10-01): arms H0 and P0.

The ICLR 2022 design (experiments/area_chairs_iclr2022.py: prompt `prompts/ac_binary.md`, the draft's nine chair models, one
run per chair and arm, the same retry rule and metrics) on each corpus's decided E14 papers (accepted or rejected
papers in results/<corpus>/discussion/e14b_sample_full.json).

Arms (review record the chair reads, besides the paper):
  H0  the official human reviews with their ratings: each review's text from the paper's dossier (the text the
      panel's evidence comes from), in reviewer order, followed by the reviewer's rating on the venue's scale.
      (ICLR 2022 built H0 from the forum notes; the content is the same reviews, laid out by the same headings.)
  P0  the eight-family panel's round-1 reviews (call A, run 0), in the ICLR 2022 P0 layout, seeded order,
      model names hidden.

Usage:
  PANEL_CORPUS=<iclr2021|neurips2022|2023-25> uv run python experiments/area_chairs.py run --max-usd 180 [--pilot 3]
  PANEL_CORPUS=<...> uv run python experiments/area_chairs.py analyze
Writes: results/<corpus>/area_chair/records.jsonl, outcomes.csv
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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import area_chairs_iclr2022 as ac22
import discussion as e14

from peerreview import area_chair, prompts
from peerreview.client import CostCeilingReached, OpenRouterClient
from peerreview.config import ROOT, SEED, TEMPERATURE, models, venues
from peerreview.content import human_review_sections
from peerreview.papers import load_dossier, paper_text
from peerreview.runner import fit_to_context

assert e14.CORPUS in ("iclr2021", "neurips2022", "2023-25"), "set PANEL_CORPUS"
OUT = e14.OUT.parent / "area_chair"
RECORDS = OUT / "records.jsonl"
CACHE = ROOT / "cache" / "responses" / "ac_corpora"
CHAIRS = ac22.CHAIRS
ARMS = ["H0", "P0"]


def human_packet(venue_key: str, paper_id: str) -> str:
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
    order = [m for m in e14.PANEL if m in reviews]
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
    man = pd.DataFrame(json.loads(e14.MANIFEST.read_text())["papers"])
    if "venue_key" not in man:
        man["venue_key"] = e14.CORPUS
    man = man[man.outcome.isin(["accept", "reject"])]
    if not e14.FULL:   # the preregistered set; SAMPLE_SCOPE=full (2026-10-02): every decided paper of the sample
        ids = set(json.loads((e14.OUT / "e14b_sample_full.json").read_text())["papers"])
        man = man[man.paper_id.isin(ids)]
    return man.sort_values("paper_id").reset_index(drop=True)


def run(args):
    papers = decided()
    if args.pilot:
        papers = papers.sample(args.pilot, random_state=SEED).sort_values("paper_id")
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if RECORDS.exists():
        done = {(r["paper_id"], r["arm"], r["ac_model_id"]) for r in map(json.loads, RECORDS.open()) if r["parse_ok"]}
    round1 = e14.round1(set(papers.paper_id))
    packet = {}
    for p in papers.itertuples():
        if "H0" in args.arms:
            packet[(p.paper_id, "H0")] = human_packet(p.venue_key, p.paper_id)
        revs = {m: round1[(p.paper_id, m)] for m in e14.PANEL if (p.paper_id, m) in round1}
        if "P0" in args.arms and len(revs) == len(e14.PANEL):
            packet[(p.paper_id, "P0")] = llm_packet(p.venue_key, p.paper_id, revs)
    by_id = papers.set_index("paper_id")
    client, model_of, lock = OpenRouterClient(CACHE, args.max_usd), models(), threading.Lock()
    jobs = [(pid, arm, m) for (pid, arm) in packet for m in args.chairs if (pid, arm, m) not in done]
    print(f"{e14.CORPUS}: {len(papers)} decided papers, {len(packet)} packets, {len(jobs)} decisions to run", flush=True)

    def work(job):
        pid, arm, key = job
        model, vkey = model_of[key], by_id.loc[pid, "venue_key"]
        overhead = len(area_chair.render_prompt(vkey, "", packet[(pid, arm)]).encode())
        text_in, trimmed = fit_to_context(paper_text(by_id.loc[pid]), model, overhead)
        prompt = area_chair.render_prompt(vkey, text_in, packet[(pid, arm)])
        cut_off = False
        # --attempt-offset (deviation 2026-10-01): fresh samples for cells still missing after the three attempts
        for attempt in range(args.attempt_offset, args.attempt_offset + 3):
            max_tokens = (2 * area_chair.MAX_OUTPUT_TOKENS if attempt - args.attempt_offset == 2 and cut_off
                          else area_chair.MAX_OUTPUT_TOKENS)
            kwargs = dict(run_idx=0, temperature=TEMPERATURE if model.accepts_temperature else None, max_tokens=max_tokens)
            if attempt:
                kwargs["attempt"] = attempt
            if model.request:
                kwargs["extra"] = model.request
            text, usage = client.complete(model.api_model, prompt, **kwargs)
            decision, meta_review = area_chair.parse_decision(text)
            rec = dict(paper_id=pid, venue_key=vkey, arm=arm, ac_model_id=key, decision=decision,
                       parse_ok=decision is not None, truth=by_id.loc[pid, "outcome"], meta_review=meta_review,
                       raw_response=text, prompt_sha256=prompts.sha256(prompt), packet_sha256=prompts.sha256(packet[(pid, arm)]),
                       paper_text_context_trimmed=trimmed, attempt=attempt + 1, max_tokens=max_tokens,
                       model_snapshot=usage.get("execution_model"),
                       usage={k: usage.get(k) for k in ("input_tokens", "output_tokens", "cost_usd", "cached",
                                                        "finish_reason", "provider")},
                       timestamp=datetime.datetime.now(datetime.UTC).isoformat())
            with lock, RECORDS.open("a") as f:
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


def analyze(args):
    recs = [r for r in map(json.loads, RECORDS.open()) if r["parse_ok"] and (args.with_retry or e14.FULL or r["attempt"] <= 3)]
    latest = {(r["paper_id"], r["arm"], r["ac_model_id"]): r for r in recs}
    allowed = set(decided().paper_id)
    papers = sorted({k[0] for k in latest} & allowed)
    keep = [p for p in papers if all((p, a, c) in latest for a in ARMS for c in CHAIRS)]
    print(f"{e14.CORPUS}: {len(keep)} papers with all {len(CHAIRS)} chairs in {ARMS}")
    truth = np.array([1 if latest[(p, ARMS[0], CHAIRS[0])]["truth"] == "accept" else 0 for p in keep])
    venue = [latest[(p, ARMS[0], CHAIRS[0])]["venue_key"] for p in keep]
    preds = {a: np.array([[1 if latest[(p, a, c)]["decision"] == "accept" else 0 for c in CHAIRS] for p in keep])
             for a in ARMS}
    groups = {"all": np.arange(len(keep))}
    if e14.CORPUS == "2023-25":
        groups |= {v: np.array([i for i, x in enumerate(venue) if x == v]) for v in sorted(set(venue))}
    rng = np.random.default_rng(SEED)
    rows = []
    for g, idx in groups.items():
        t = truth[idx]
        draws = [d for d in (rng.integers(0, len(idx), len(idx)) for _ in range(ac22.N_BOOT)) if 0 < t[d].sum() < len(d)]
        P = {a: preds[a][idx] for a in ARMS}
        for a in ARMS:
            point = ac22.arm_metrics(P[a], t)
            boots = [ac22.arm_metrics(P[a][d], t[d]) for d in draws]
            for k, v in point.items():
                b = [x[k] for x in boots if not np.isnan(x[k])]
                rows.append(dict(group=g, arm=a, metric=k, estimate=v, ci_low=np.percentile(b, 2.5),
                                 ci_high=np.percentile(b, 97.5), n_papers=len(idx), n_chairs=len(CHAIRS)))
        for k in ("balanced_accuracy", "all_chairs_wrong"):
            diff = [ac22.arm_metrics(P["P0"][d], t[d])[k] - ac22.arm_metrics(P["H0"][d], t[d])[k] for d in draws]
            est = ac22.arm_metrics(P["P0"], t)[k] - ac22.arm_metrics(P["H0"], t)[k]
            rows.append(dict(group=g, arm="P0 - H0", metric=k, estimate=est, ci_low=np.percentile(diff, 2.5),
                             ci_high=np.percentile(diff, 97.5), n_papers=len(idx), n_chairs=len(CHAIRS)))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / ("outcomes_full.csv" if e14.FULL else "outcomes_with_retry.csv" if args.with_retry else "outcomes.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(table.pivot_table(index=["group", "metric"], columns="arm", values="estimate").round(3).to_string())
    print("all-chairs-wrong papers:", {a: int((preds[a] != truth[:, None]).min(axis=1).sum()) for a in ARMS})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    r.add_argument("--chairs", nargs="+", default=CHAIRS)
    r.add_argument("--pilot", type=int, default=0)
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--max-usd", type=float, required=True)
    r.add_argument("--attempt-offset", type=int, default=0, help="first attempt index (3 = fresh retry samples)")
    r.set_defaults(run=run)
    a = sub.add_parser("analyze")
    a.add_argument("--with-retry", action="store_true", help="include the deviation's fresh retry samples")
    a.set_defaults(run=analyze)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
