"""Notes: give each reviewer weaknesses the panel missed (or control notes), then let it revise its review.

Arms, 3 notes per paper, the same notes for the eight reviewers of a paper:
  missed     human-raised weaknesses no 8-family review mentions (any of 3 runs)
  mentioned  human-raised weaknesses at least one review mentions
  false      human-raised weaknesses of another sampled paper (seeded pairing)
  neutral    non-evaluative sentences from the human reviewers' summaries of this paper
Baseline: the E14 second-look arm (no notes), read from results/iclr2022_prechatgpt/discussion.

Usage:
  uv run python experiments/notes.py run --pilot 5 --max-usd 15
  uv run python experiments/notes.py run --max-usd 300
  uv run python experiments/notes.py analyze
Writes: results/iclr2022_prechatgpt/injection/{design.json, records.jsonl, outcomes.csv}
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import discussion as e14
import forums as forums

from peerreview import prompts
from peerreview.client import CostCeilingReached, OpenRouterClient
from peerreview.config import MAX_OUTPUT_TOKENS, ROOT, SEED, TEMPERATURE, models, venues
from peerreview.papers import paper_text
from peerreview.runner import MAX_ATTEMPTS, fit_to_context

CACHE = ROOT / "cache" / "responses" / "injection"
# Corpus follows PANEL_CORPUS (see discussion_experiment): on 2023-25 the E8 replication draws up to 150
# decided papers from the 180-paper E14 replication sample; "missed" = missed by the 8 panel8f round-1
# reviews; no neutral arm (preregistered); scores normalized per venue in the analysis.
if e14.CORPUS == "2023-25" and os.environ.get("E8_2325") == "all700":
    # complete-sample extension (2026-10-02): all decided 2023-25 papers, the rebuilt extraction and labels (all 700
    # papers, configs/experiments/study2325_panel8f.yaml); every eligible paper; separate outputs
    OUT = ROOT / "results" / "study2325_panel8f" / "injection"
    WEAKNESSES = ROOT / "results" / "study2325_panel8f" / "content" / "reference_items.jsonl"
    LABELS = ROOT / "results" / "study2325_panel8f" / "content" / "coverage_labels.csv"   # run 0, panel8f
    ARMS = ["missed", "mentioned", "false"]
    N_PAPERS_MAX = 100000
elif e14.CORPUS == "2023-25":
    OUT = ROOT / "results" / "study2325" / "injection"
    WEAKNESSES = ROOT / "data" / "processed" / "content" / "reference_items.jsonl"
    LABELS = ROOT / "results" / "study2325" / "discussion" / "coverage_labels.csv"   # arm == "round1"
    ARMS = ["missed", "mentioned", "false"]
    N_PAPERS_MAX = 150
elif e14.CORPUS in ("iclr2021", "neurips2022"):   # E8 replications on the other two pre-ChatGPT corpora (2026-10-01)
    OUT = ROOT / "results" / f"{e14.CORPUS}_prechatgpt" / "injection"
    WEAKNESSES = ROOT / "results" / f"{e14.CORPUS}_prechatgpt" / "content" / "reference_items.jsonl"
    LABELS = ROOT / "results" / f"{e14.CORPUS}_prechatgpt" / "content" / "coverage_labels.csv"   # run 0
    ARMS = ["missed", "mentioned", "false"]
    N_PAPERS_MAX = 60     # preregistered cap (cost); 193 and 158 papers are eligible
else:
    OUT = ROOT / "results" / "iclr2022_prechatgpt" / "injection"
    WEAKNESSES = ROOT / "results" / "iclr2022_prechatgpt" / "content" / "reference_items.jsonl"
    LABELS = ROOT / "results" / "iclr2022_prechatgpt" / "content" / "coverage_labels.csv"
    ARMS = ["missed", "mentioned", "false", "neutral"]
    N_PAPERS_MAX = 100
# R2b (preregistered second sample of E8b): SECOND_SAMPLE=1 draws new samples (run_idx 1, so the response cache
# does not return the first sample) and writes records_second_sample.jsonl.
SECOND_SAMPLE = os.environ.get("SECOND_SAMPLE") == "1"
FRESH = {"fresh_missed": "missed", "fresh_false": "false"}   # E8b: notes before a fresh review (anchoring check)
N_NOTES, N_BOOT = 3, 2000
FRAME_SEED = f"{SEED}:e8"


def build_design() -> dict:
    """Frozen sample and notes; written once to design.json and reused. With E8_SAMPLE=full (amendment
    2026-09-30), the extended design: the original papers and notes plus every other eligible paper."""
    if os.environ.get("E8_SAMPLE") == "full":
        return extend_design()
    return _build_design()


def extend_design() -> dict:
    path = OUT / "design_full.json"
    if path.exists():
        return json.loads(path.read_text())
    base = _build_design()
    everything = _build_design(all_eligible=True, write=False)
    new = sorted(set(everything["papers"]) - set(base["papers"]))
    rng = random.Random(f"{FRAME_SEED}:extension")
    partners = new[:]
    while any(a == b for a, b in zip(new, partners)):
        rng.shuffle(partners)
    papers = dict(base["papers"])
    for pid, partner in zip(new, partners):
        entry = dict(everything["papers"][pid])
        r = random.Random(f"{FRAME_SEED}:{pid}:false-extension")
        weak = {w["id"]: w for w in _weaknesses().get(partner, [])}
        entry["partner"] = partner
        entry["false"] = [{"id": f"{partner}:{w['id']}", "text": w["text"]} for w in r.sample(list(weak.values()), N_NOTES)]
        papers[pid] = entry
    design = {"eligible": everything["eligible"], "extension_of": "design.json", "new_papers": new, "papers": papers}
    path.write_text(json.dumps(design, indent=1))
    return design


def _weaknesses() -> dict[str, list[dict]]:
    weak: dict[str, list[dict]] = {}
    for line in WEAKNESSES.open():
        item = json.loads(line)
        if item["polarity"] == "weakness":
            weak.setdefault(item["paper_id"], []).append(item)
    return weak


def _build_design(all_eligible: bool = False, write: bool = True) -> dict:
    path = OUT / "design.json"
    if path.exists() and not all_eligible:
        return json.loads(path.read_text())
    manifest = {p["paper_id"]: p for p in json.loads(e14.MANIFEST.read_text())["papers"]}
    weak: dict[str, list[dict]] = {}
    for line in WEAKNESSES.open():
        item = json.loads(line)
        if item["polarity"] == "weakness":
            weak.setdefault(item["paper_id"], []).append(item)
    labels = pd.read_csv(LABELS)
    if "arm" in labels.columns:
        labels = labels[labels.arm == "round1"]
    labels = labels[labels.model.isin(e14.PANEL) & labels.recognition.isin(["specific", "partial"])]
    pool = (set(json.loads((ROOT / "results" / "study2325" / "discussion" / "sample.json").read_text())["papers"])
            if e14.CORPUS == "2023-25" and os.environ.get("E8_2325") != "all700" else set(weak))
    mentioned = labels.groupby("paper_id").id.apply(set).to_dict()
    eligible = []
    for pid, ws in weak.items():
        if pid not in pool or manifest[pid]["outcome"] not in ("accept", "reject"):
            continue
        hit = mentioned.get(pid, set())
        if sum(w["id"] not in hit for w in ws) >= N_NOTES and sum(w["id"] in hit for w in ws) >= N_NOTES:
            eligible.append(pid)
    rng = random.Random(FRAME_SEED)
    sample = sorted(rng.sample(sorted(eligible), min(N_PAPERS_MAX, len(eligible))))
    if all_eligible:
        sample = sorted(eligible)
    partners = sample[:]
    while any(a == b for a, b in zip(sample, partners)):
        rng.shuffle(partners)
    design = {"eligible": len(eligible), "papers": {}}
    for pid, partner in zip(sample, partners):
        hit = mentioned.get(pid, set())
        r = random.Random(f"{FRAME_SEED}:{pid}")
        missed = r.sample([w for w in weak[pid] if w["id"] not in hit], N_NOTES)
        known = r.sample([w for w in weak[pid] if w["id"] in hit], N_NOTES)
        false = r.sample(weak[partner], N_NOTES)
        neutral = []
        if "neutral" in ARMS:
            summary = " ".join(str(n["content"].get("summary_of_the_paper", "")) for _, n in forums.reviews(pid))
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if 60 <= len(s.strip()) <= 300]
            neutral = r.sample(sentences, min(N_NOTES, len(sentences)))
        design["papers"][pid] = {
            "partner": partner,
            "missed": [{"id": w["id"], "text": w["text"]} for w in missed],
            "mentioned": [{"id": w["id"], "text": w["text"]} for w in known],
            "false": [{"id": f"{partner}:{w['id']}", "text": w["text"]} for w in false],
            "neutral": [{"id": f"summary{k}", "text": s} for k, s in enumerate(neutral)]}
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(design, indent=1))
    return design


def notes_block(notes: list[dict], pid: str, arm: str) -> str:
    order = list(notes)
    random.Random(f"{FRAME_SEED}:{pid}:{arm}:order").shuffle(order)
    return "\n".join(f"- {n['text']}" for n in order)


def run(args):
    design = build_design()
    ids = sorted(design["papers"])
    if args.pilot:
        ids = ids[:args.pilot]
    manifest = pd.DataFrame(json.loads(e14.MANIFEST.read_text())["papers"]).set_index("paper_id")
    reviews = e14.round1(set(ids))
    model_of = models()
    client = OpenRouterClient(CACHE, args.max_usd)
    record_path = OUT / ("records_second_sample.jsonl" if SECOND_SAMPLE else "records.jsonl")
    done, fails = set(), {}
    if record_path.exists():
        for r in map(json.loads, record_path.open()):
            key = (r["paper_id"], r["model_id"], r["arm"])
            if r["parse_ok"]:
                done.add(key)
            else:
                fails[key] = fails.get(key, 0) + 1
    jobs = [(pid, m, a) for pid in ids for m in e14.PANEL for a in args.arms
            if (pid, m, a) not in done and (pid, m) in reviews]
    print(f"{len(ids)} papers, {len(jobs)} cells to run ({len(done)} done)", flush=True)
    lock = threading.Lock()

    def work(job):
        pid, key, arm = job
        model, paper = model_of[key], manifest.loc[pid]
        venue = venues()[paper["venue_key"]] if "venue_key" in paper else venues()["iclr2022"]
        own = reviews[(pid, key)]
        source_arm = FRESH.get(arm, arm)
        notes = notes_block(design["papers"][pid][source_arm], pid, source_arm)
        template = "call_a_fresh_notes" if arm in FRESH else "call_a_notes"
        fields = dict(notes=notes) if arm in FRESH else dict(
            source_review=e14.review_text(own), initial_score=str(own["emitted_score"]), notes=notes)

        def render(text):
            return prompts.render(template, own_review_prompt=prompts.own_review(venue, text), **fields)

        text_in, trimmed = fit_to_context(paper_text(paper), model, len(render("").encode()))
        prompt = render(text_in)
        temperature = TEMPERATURE if model.accepts_temperature else None
        first = fails.get((pid, key, arm), 0)
        for attempt in range(first, first + MAX_ATTEMPTS):
            kwargs = dict(run_idx=1 if SECOND_SAMPLE else 0, temperature=temperature, max_tokens=MAX_OUTPUT_TOKENS)
            if attempt:
                kwargs["attempt"] = attempt
            if model.request:
                kwargs["extra"] = model.request
            started = time.perf_counter()
            raw, usage = client.complete(model.api_model, prompt, **kwargs)
            parsed = e14.parse_round2(raw, venue.scores)
            rec = dict(paper_id=pid, model_id=key, arm=arm, prompt_template=template,
                       prompt_version=prompts.load_template(template)[0]["version"], prompt_sha256=prompts.sha256(prompt),
                       notes=design["papers"][pid][source_arm], round1_call_id=own["call_id"],
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


def analyze(args):
    design = build_design()
    manifest = {p["paper_id"]: p for p in json.loads(e14.MANIFEST.read_text())["papers"]}
    recs = [r for r in map(json.loads, (OUT / "records.jsonl").open()) if r["parse_ok"]]
    latest = {(r["paper_id"], r["model_id"], r["arm"]): r for r in recs}
    base = {(r["paper_id"], r["model_id"]): r for r in e14.all_records() if r["parse_ok"] and r["arm"] == "second_look"}
    ids = sorted(design["papers"])
    reviews = e14.round1(set(ids))
    keep = [p for p in ids if all((p, m) in reviews and (p, m) in base and all((p, m, a) in latest for a in ARMS)
                                  for m in e14.PANEL)]
    print(f"{len(keep)} papers with all {len(e14.PANEL)} reviewers in round 1, second look and all {len(ARMS)} arms")
    human = np.array([e14.human_score(manifest[p]) for p in keep])
    truth = np.array([1 if manifest[p]["outcome"] == "accept" else 0 for p in keep])
    sc = lambda p, s: e14.model_score(manifest[p], s)
    R1 = np.array([[sc(p, reviews[(p, m)]["emitted_score"]) for m in e14.PANEL] for p in keep], float)
    S = {"second_look": np.array([[sc(p, base[(p, m)]["emitted_score"]) for m in e14.PANEL] for p in keep], float)}
    for a in ARMS:
        S[a] = np.array([[sc(p, latest[(p, m, a)]["emitted_score"]) for m in e14.PANEL] for p in keep], float)

    def stats(M, idx=slice(None)):
        E = M[idx] - human[idx][:, None]
        C = np.corrcoef(E.T)
        iu = np.triu_indices(M.shape[1], 1)
        return dict(score_change=float((M[idx] - R1[idx]).mean()),
                    committee_abs_error=float(np.abs(M[idx].mean(axis=1) - human[idx]).mean()),
                    decision_auroc=e14.auroc(M[idx].mean(axis=1), truth[idx]),
                    error_correlation=float(np.nanmean(C[iu])))

    rng = np.random.default_rng(SEED)
    draws = [d for d in (rng.integers(0, len(keep), len(keep)) for _ in range(N_BOOT)) if 0 < truth[d].sum() < len(d)]
    rows = []
    for a, M in S.items():
        for k, v in stats(M).items():
            rows.append(dict(arm=a, outcome=k, estimate=v, n_papers=len(keep)))
    contrasts = [c for c in [("missed", "mentioned"), ("missed", "neutral"), ("missed", "false"), ("missed", "second_look"),
                             ("false", "second_look"), ("neutral", "second_look"), ("mentioned", "second_look")]
                 if all(x in S for x in c)]
    for a, b in contrasts:
        point_a, point_b = stats(S[a]), stats(S[b])
        boots = [(stats(S[a], d), stats(S[b], d)) for d in draws]
        for k in point_a:
            diffs = [x[k] - y[k] for x, y in boots]
            rows.append(dict(arm=f"{a} - {b}", outcome=k, estimate=point_a[k] - point_b[k],
                             ci_low=np.percentile(diffs, 2.5), ci_high=np.percentile(diffs, 97.5), n_papers=len(keep)))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / ("outcomes_full.csv" if os.environ.get("E8_SAMPLE") == "full" else "outcomes.csv"), index=False)
    wide = table[table.arm.isin(S)].pivot(index="outcome", columns="arm", values="estimate")[["second_look"] + ARMS]
    print(wide.round(3).to_string())
    diffs = table[~table.arm.isin(S)].copy()
    diffs["ci"] = diffs.apply(lambda r: f"{r.estimate:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]", axis=1)
    print("\npaired 95% intervals:")
    print(diffs.pivot(index="arm", columns="outcome", values="ci").to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--pilot", type=int, default=0)
    r.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS + list(FRESH))
    r.add_argument("--workers", type=int, default=24)
    r.add_argument("--max-usd", type=float, required=True)
    r.set_defaults(run=run)
    a = sub.add_parser("analyze")
    a.set_defaults(run=analyze)
    d = sub.add_parser("design")
    d.set_defaults(run=lambda args: print(json.dumps({k: v for k, v in build_design().items() if k != "papers"})))
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__" and not (len(sys.argv) > 1 and sys.argv[1] in ("adoption", "analyze-fresh")):
    main()


def adoption_main():
    """Secondary outcome 5: share of the injected notes that the revised review mentions (the round-1
    matcher and prompt, with the three injected notes as the points), per arm; round-1 review as baseline."""
    from concurrent.futures import ThreadPoolExecutor as Pool

    from peerreview import content
    design = build_design()
    annotator = models()["gemini-3.5-flash-lite"]
    client = OpenRouterClient(ROOT / "cache" / "responses" / "content_match_e8", 20)
    recs = {(r["paper_id"], r["model_id"], r["arm"]): r for r in map(json.loads, (OUT / "records.jsonl").open())
            if r["parse_ok"] and r["items"]}
    reviews = e14.round1(set(design["papers"]))
    jobs = []
    for (pid, m, arm), r in recs.items():
        if arm == "neutral":
            continue
        points = [{"id": f"N{k}", "polarity": "weakness", "text": n["text"]} for k, n in enumerate(design["papers"][pid][arm.removeprefix("fresh_")])]
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

    with Pool(24) as pool:
        rows = [x for x in pool.map(work, jobs) if x["mentioned"] is not None]
    table = pd.DataFrame(rows)
    table.to_csv(OUT / ("adoption_labels_full.csv" if os.environ.get("E8_SAMPLE") == "full" else "adoption_labels.csv"), index=False)
    table["share"] = table.mentioned / table.n_notes
    out = table.pivot_table(index="arm", columns="review", values="share", aggfunc="mean")
    out["adopted (revised - round 1)"] = out.revised - out.round1
    print(f"{len(table)} labeled reviews, ${client.spent:.2f}")
    print((100 * out).round(1).to_string())


def analyze_fresh():
    """E8b (preregistered decision rule R1-R3): fresh reviews with notes vs. an independent fresh review
    without notes (round-1 run 1), on the E8 papers. Corpus from PANEL_CORPUS (2026-10-02: also ICLR 2021,
    NeurIPS 2022 and 2023-25, where scores are normalized per venue); writes outcomes_fresh.csv."""
    from peerreview import records
    design = build_design()
    manifest = {p["paper_id"]: p for p in json.loads(e14.MANIFEST.read_text())["papers"]}
    ids = sorted(design["papers"])
    base = {}
    for m in e14.PANEL:
        for venue_key, tag in e14.ROUND1_SOURCES:
            try:
                for r in records.iter_records(m, venue_key, "paper_review", "A", tag=tag):
                    if r["paper_id"] in design["papers"] and r["run_idx"] == 1 and r["parse_ok"]:
                        base[(r["paper_id"], m)] = r["emitted_score"]
            except FileNotFoundError:
                continue
    name = "records_second_sample.jsonl" if SECOND_SAMPLE else "records.jsonl"
    recs = {(r["paper_id"], r["model_id"], r["arm"]): r for r in map(json.loads, (OUT / name).open())
            if r["parse_ok"]}
    arms = [a for a in FRESH if any(k[2] == a for k in recs)]
    keep = [p for p in ids if all((p, m) in base and all((p, m, a) in recs for a in arms) for m in e14.PANEL)]
    human = np.array([e14.human_score(manifest[p]) for p in keep])
    truth = np.array([1.0 if manifest[p]["outcome"] == "accept" else 0.0 for p in keep])
    sc = lambda p, x: e14.model_score(manifest[p], x)
    B = np.array([[sc(p, base[(p, m)]) for m in e14.PANEL] for p in keep], float)
    S = {a: np.array([[sc(p, recs[(p, m, a)]["emitted_score"]) for m in e14.PANEL] for p in keep], float) for a in arms}
    err = lambda M, i=slice(None): float(np.abs(M[i].mean(axis=1) - human[i]).mean())
    auc = lambda M, i=slice(None): float(roc_auc_score(truth[i], M[i].mean(axis=1))) if len(set(truth[i])) == 2 else np.nan
    rng = np.random.default_rng(SEED)
    draws = [rng.integers(0, len(keep), len(keep)) for _ in range(N_BOOT)]
    print(f"{len(keep)} papers; baseline committee error {err(B):.3f}")
    rows = []
    for a, M in S.items():
        for k, f in (("score_change", lambda X, Y, i=slice(None): float((X[i] - Y[i]).mean())),
                     ("committee_error_change", lambda X, Y, i=slice(None): err(X, i) - err(Y, i)),
                     ("decision_auroc_change", lambda X, Y, i=slice(None): auc(X, i) - auc(Y, i))):
            d = [f(M, B, i) for i in draws]
            rows.append(dict(arm=a, outcome=k, estimate=f(M, B), ci_low=float(np.nanpercentile(d, 2.5)),
                             ci_high=float(np.nanpercentile(d, 97.5)), n_papers=len(keep)))
            print(f"{a:13s} {k:24s} {f(M, B):+.4f} [{np.nanpercentile(d, 2.5):+.4f}, {np.nanpercentile(d, 97.5):+.4f}]")
    d = [float((S["fresh_false"][i] - B[i]).mean()) - float((S["fresh_missed"][i] - B[i]).mean()) for i in draws] \
        if {"fresh_false", "fresh_missed"} <= set(S) else []
    if d:
        est = float((S["fresh_false"] - B).mean()) - float((S["fresh_missed"] - B).mean())
        rows.append(dict(arm="fresh_false - fresh_missed", outcome="score_change", estimate=est,
                         ci_low=float(np.percentile(d, 2.5)), ci_high=float(np.percentile(d, 97.5)), n_papers=len(keep)))
    pd.DataFrame(rows).to_csv(OUT / ("outcomes_fresh_second_sample.csv" if SECOND_SAMPLE else
                                     "outcomes_fresh_full.csv" if os.environ.get("E8_SAMPLE") == "full" else "outcomes_fresh.csv"), index=False)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "adoption":
    adoption_main()
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "analyze-fresh":
    analyze_fresh()
