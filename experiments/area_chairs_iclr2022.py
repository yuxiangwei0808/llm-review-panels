"""Stage 5 on ICLR 2022 (plan item E13): LLM area chairs decide accept or reject.

Same prompt (`prompts/ac_binary.md`), same nine chair models and same packet layout as the draft's
area-chair study, on the 145 decided ICLR 2022 papers. One run per chair and arm.

Arms (review record the chair reads, besides the paper):
  H0      the official human reviews with their scores
  H2      H0 plus the public author-reviewer discussion before the decision
  P0      the eight-family LLM panel's round-1 reviews (own evidence, run 0), with scores
  P_disc  the same panel's reviews after reading each other (E14 discussion arm)
  P_reb   the same panel's reviews revised after the real author responses (stage 3), plus those responses
P_disc and P_reb need the E14 and stage-3 runs (experiments/discussion.py) to have finished.

Usage:
  uv run python experiments/area_chairs_iclr2022.py run --arms H0 H2 P0 --max-usd 250 [--pilot 5]
  uv run python experiments/area_chairs_iclr2022.py analyze
Writes: results/iclr2022_prechatgpt/area_chair/records.jsonl, outcomes.csv
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import discussion as e14
import forums as forums

from peerreview import area_chair, prompts
from peerreview.client import CostCeilingReached, OpenRouterClient
from peerreview.config import ROOT, SEED, TEMPERATURE, models
from peerreview.papers import paper_text
from peerreview.runner import fit_to_context

OUT = ROOT / "results" / "iclr2022_prechatgpt" / "area_chair"
# R2d (second run, 2026-10-01): AC_RUN=2 draws new samples (request index 1) and writes records_run2.jsonl.
AC_RUN = int(os.environ.get("AC_RUN", "1"))
RECORDS = OUT / ("records.jsonl" if AC_RUN == 1 else f"records_run{AC_RUN}.jsonl")
CACHE = ROOT / "cache" / "responses" / "ac_iclr2022"
CHAIRS = ["gemini-3.1-pro", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gpt-5.6-luna", "gpt-5.6-terra",
          "gpt-5.1", "deepseek-v4-flash-0731", "glm-5.2", "claude-sonnet-5"]   # the draft's nine chairs
ARMS = ["H0", "H2", "P0", "P_disc", "P_reb"]
N_BOOT = 2000


def llm_packet(paper_id: str, reviews: dict[str, dict], heading: str, responses: str = "") -> str:
    """The panel's reviews in the draft's P0 layout, in a seeded order, model names hidden."""
    order = [m for m in e14.PANEL if m in reviews]
    random.Random(f"{SEED}:ac:{paper_id}").shuffle(order)
    parts = [f"## {heading}"]
    for k, m in enumerate(order, start=1):
        r = reviews[m]
        extra = r.get("extra") or {}
        strengths = "\n".join(f"- {i['text']}" for i in r["items"] if i["polarity"] == "strength")
        weaknesses = "\n".join(f"- {i['text']}" for i in r["items"] if i["polarity"] == "weakness")
        parts.append(f"### Anonymous Reviewer {k}\n\n#### Public review rationale\n\n{(r.get('reasoning_text') or '').strip()}"
                     f"\n\n#### Summary\n\n{str(extra.get('summary', '')).strip()}\n\n#### Strengths\n\n{strengths}"
                     f"\n\n#### Weaknesses\n\n{weaknesses}\n\n#### Overall recommendation\n\n"
                     f"{r['emitted_score']} on the official iclr2022 reviewer scale")
    if responses:
        parts.append("## REAL AUTHOR REBUTTAL\n\n" + responses)
    return "\n\n".join(parts)


def packets(paper_ids: list[str], arms: list[str]) -> dict[tuple[str, str], str]:
    out = {}
    round1 = e14.round1(set(paper_ids))
    round2 = {}
    if {"P_disc", "P_reb"} & set(arms):
        for r in e14.all_records():
            if r["parse_ok"]:
                round2[(r["paper_id"], r["model_id"], r["arm"])] = r
    for pid in paper_ids:
        if "H0" in arms:
            out[(pid, "H0")] = forums.human_packet(pid, with_discussion=False)
        if "H2" in arms:
            out[(pid, "H2")] = forums.human_packet(pid, with_discussion=True)
        if "P0" in arms:
            revs = {m: round1[(pid, m)] for m in e14.PANEL if (pid, m) in round1}
            if len(revs) == len(e14.PANEL):
                out[(pid, "P0")] = llm_packet(pid, revs, "INITIAL ANONYMOUS REVIEWS")
        if "P_disc" in arms:
            revs = {m: round2[(pid, m, "discussion")] for m in e14.PANEL if (pid, m, "discussion") in round2}
            if len(revs) == len(e14.PANEL):
                out[(pid, "P_disc")] = llm_packet(pid, revs, "ANONYMOUS REVIEWS AFTER THE REVIEWERS READ EACH OTHER'S REVIEWS")
        if "P_reb" in arms:
            revs = {m: round2[(pid, m, "rebuttal")] for m in e14.PANEL if (pid, m, "rebuttal") in round2}
            if len(revs) == len(e14.PANEL):
                out[(pid, "P_reb")] = llm_packet(pid, revs, "ANONYMOUS REVIEWS REVISED AFTER THE AUTHOR RESPONSE",
                                                 forums.author_response(pid))
    return out


def decided() -> pd.DataFrame:
    man = pd.DataFrame(json.loads(e14.MANIFEST.read_text())["papers"])
    return man[man.outcome.isin(["accept", "reject"])].sort_values("paper_id").reset_index(drop=True)


def run(args):
    papers = decided()
    if args.pilot:
        papers = papers.sample(args.pilot, random_state=SEED).sort_values("paper_id")
    record_path = RECORDS
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if record_path.exists():
        done = {(r["paper_id"], r["arm"], r["ac_model_id"]) for r in map(json.loads, record_path.open()) if r["parse_ok"]}
    packet = packets(list(papers.paper_id), args.arms)
    by_id = papers.set_index("paper_id")
    client, model_of, lock = OpenRouterClient(CACHE, args.max_usd), models(), threading.Lock()
    jobs = [(pid, arm, m) for (pid, arm) in packet for m in args.chairs if (pid, arm, m) not in done]
    print(f"{len(papers)} papers, {len(packet)} packets, {len(jobs)} decisions to run", flush=True)

    def work(job):
        pid, arm, key = job
        model = model_of[key]
        overhead = len(area_chair.render_prompt("iclr2022", "", packet[(pid, arm)]).encode())
        text_in, trimmed = fit_to_context(paper_text(by_id.loc[pid]), model, overhead)
        prompt = area_chair.render_prompt("iclr2022", text_in, packet[(pid, arm)])
        cut_off = False
        off = getattr(args, "attempt_offset", 0)   # 2026-10-02: fresh retry samples for still-unparsed decisions
        for attempt in range(off, off + 3):
            # the draft's 2,048-token cap; a last attempt after a cut-off response gets 4,096 (recorded)
            max_tokens = 2 * area_chair.MAX_OUTPUT_TOKENS if attempt - off == 2 and cut_off else area_chair.MAX_OUTPUT_TOKENS
            kwargs = dict(run_idx=AC_RUN - 1, temperature=TEMPERATURE if model.accepts_temperature else None,
                          max_tokens=max_tokens)
            if attempt:
                kwargs["attempt"] = attempt
            if model.request:
                kwargs["extra"] = model.request
            text, usage = client.complete(model.api_model, prompt, **kwargs)
            decision, meta_review = area_chair.parse_decision(text)
            rec = dict(paper_id=pid, arm=arm, ac_model_id=key, decision=decision, parse_ok=decision is not None,
                       truth=by_id.loc[pid, "outcome"], meta_review=meta_review, raw_response=text,
                       prompt_sha256=prompts.sha256(prompt), packet_sha256=prompts.sha256(packet[(pid, arm)]),
                       paper_text_context_trimmed=trimmed, attempt=attempt + 1, max_tokens=max_tokens,
                       model_snapshot=usage.get("execution_model"),
                       usage={k: usage.get(k) for k in ("input_tokens", "output_tokens", "cost_usd", "cached",
                                                        "finish_reason", "provider")},
                       timestamp=datetime.datetime.now(datetime.UTC).isoformat())
            with lock, record_path.open("a") as f:
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
            if k % 50 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} decisions, {n} parsed, ${client.spent:.2f}", flush=True)
    print(f"done, spent ${client.spent:.2f}")


def arm_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """pred: papers x chairs (1 accept, 0 reject); truth: papers."""
    tpr = (pred[truth == 1] == 1).mean(axis=0)
    tnr = (pred[truth == 0] == 0).mean(axis=0)
    err = (pred != truth[:, None]).astype(float)
    pairs = list(itertools.combinations(range(pred.shape[1]), 2))
    agree = np.mean([(pred[:, i] == pred[:, j]).mean() for i, j in pairs])
    cors = [np.corrcoef(err[:, i], err[:, j])[0, 1] for i, j in pairs if err[:, i].std() > 0 and err[:, j].std() > 0]
    all_wrong = err.min(axis=1) == 1
    return dict(balanced_accuracy=float(((tpr + tnr) / 2).mean()), accept_rate=float(pred.mean()),
                pairwise_agreement=float(agree), error_correlation=float(np.mean(cors)) if cors else np.nan,
                all_chairs_wrong=float(all_wrong.mean()),
                all_wrong_false_accepts=float((all_wrong & (truth == 0)).sum() / max(all_wrong.sum(), 1)))


def analyze(args):
    full = os.environ.get("SAMPLE_SCOPE") == "full"   # complete-sample: include the fresh retry samples
    recs = [r for r in map(json.loads, RECORDS.open()) if r["parse_ok"] and (full or r.get("attempt", 1) <= 3)]
    latest = {(r["paper_id"], r["arm"], r["ac_model_id"]): r for r in recs}
    arms = [a for a in ARMS if any(k[1] == a for k in latest)]
    # compare arms on the same papers: every chair decided the paper in every arm present
    papers = sorted({k[0] for k in latest})
    keep = [p for p in papers if all((p, a, c) in latest for a in arms for c in CHAIRS)]
    print(f"{len(keep)} papers with all {len(CHAIRS)} chairs in arms {arms}")
    truth = np.array([1 if latest[(p, arms[0], CHAIRS[0])]["truth"] == "accept" else 0 for p in keep])
    rng = np.random.default_rng(SEED)
    draws = [rng.integers(0, len(keep), len(keep)) for _ in range(N_BOOT)]
    draws = [d for d in draws if 0 < truth[d].sum() < len(d)]
    rows = []
    preds = {a: np.array([[1 if latest[(p, a, c)]["decision"] == "accept" else 0 for c in CHAIRS] for p in keep])
             for a in arms}
    for a, P in preds.items():
        point = arm_metrics(P, truth)
        boots = [arm_metrics(P[d], truth[d]) for d in draws]
        for k, v in point.items():
            b = [x[k] for x in boots if not np.isnan(x[k])]
            rows.append(dict(arm=a, metric=k, estimate=v, ci_low=np.percentile(b, 2.5), ci_high=np.percentile(b, 97.5),
                             n_papers=len(keep), n_chairs=len(CHAIRS)))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / (("outcomes.csv" if AC_RUN == 1 else f"outcomes_run{AC_RUN}.csv") if not full else
                        ("outcomes_full.csv" if AC_RUN == 1 else f"outcomes_run{AC_RUN}_full.csv")), index=False)
    print(table.pivot(index="metric", columns="arm", values="estimate")[arms].round(3).to_string())
    counts = {a: int((preds[a] != truth[:, None]).min(axis=1).sum()) for a in arms}
    print("\nall-chairs-wrong papers (count):", counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=["H0", "H2", "P0"], choices=ARMS)
    r.add_argument("--chairs", nargs="+", default=CHAIRS)
    r.add_argument("--pilot", type=int, default=0)
    r.add_argument("--workers", type=int, default=24)
    r.add_argument("--max-usd", type=float, required=True)
    r.add_argument("--attempt-offset", type=int, default=0)
    r.set_defaults(run=run)
    a = sub.add_parser("analyze")
    a.set_defaults(run=analyze)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
