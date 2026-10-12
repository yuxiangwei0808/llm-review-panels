"""Roles: a role-diverse panel at independent review, on all 225 ICLR 2022 papers.

Each of the eight panel models writes a fresh call-A review with a focus role; per paper the four
roles are given twice each, rotated across models by a seeded shuffle, so role is not tied to model
family. The baseline is the same panel without roles: call A runs 0, 1 and 2 (each outcome computed
per run, then averaged over the three runs).

Other corpora (2026-10-04, preregistered complete-sample remedies): PANEL_CORPUS=iclr2021 | neurips2022 | 2023-25, every
sample paper; baseline = the no-role call A run 0 only (no repeat runs); 2023-25 scores normalized per venue and coverage
against the rebuilt 700-paper extraction. ICLR 2022: SAMPLE_SCOPE=full retries its missing cells and writes outcomes_full.csv
(the preregistered outcomes.csv uses only records written before 2026-10-04).

Usage:  uv run python experiments/roles.py run --max-usd 170
        uv run python experiments/roles.py coverage --max-usd 10
        uv run python experiments/roles.py analyze
Writes: results/iclr2022_prechatgpt/roles/{assignment.json, records.jsonl, coverage_labels.csv, outcomes.csv}
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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import discussion as e14

from peerreview import prompts, records
from peerreview.client import CostCeilingReached, OpenRouterClient
from peerreview.config import MAX_OUTPUT_TOKENS, ROOT, TEMPERATURE, models, venues
from peerreview.papers import paper_text
from peerreview.runner import MAX_ATTEMPTS, fit_to_context

OUT = e14.OUT.parent / "roles"   # results/<corpus>/roles (ICLR 2022: results/iclr2022_prechatgpt/roles, unchanged)
CONTENT = {"iclr2022": ROOT / "results" / "iclr2022_prechatgpt" / "content",
           "iclr2021": ROOT / "results" / "iclr2021_prechatgpt" / "content",
           "neurips2022": ROOT / "results" / "neurips2022_prechatgpt" / "content",
           "2023-25": ROOT / "results" / "study2325_panel8f" / "content"}[e14.CORPUS]
BASE_RUNS = (0, 1, 2) if e14.CORPUS == "iclr2022" else (0,)
PREREG_CUTOFF = "2026-10-04"   # ICLR 2022: records before this date make the preregistered outcomes
CACHE = ROOT / "cache" / "responses" / "role_panel"
SEED, N_BOOT = 20260930, 2000
ROLES = {
    "soundness": ("Soundness of the claims",
                  ("Are the central claims correct and supported? Check the theory, derivations, assumptions and "
                  "whether the conclusions follow from the evidence.")),
    "experiments": ("Experiments and evaluation",
                    ("Do the experiments test the claims? Check baselines, datasets, metrics, ablations, statistical "
                    "reliability and whether the evaluation is fair and sufficient.")),
    "clarity": ("Clarity and reproducibility",
                ("Could an expert understand and reproduce the work? Check the exposition, definitions, missing "
                "details, hyperparameters, code or data availability and the presentation of results.")),
    "novelty": ("Novelty and related work",
                ("Is the contribution new and significant relative to prior work? Check how the paper positions "
                "itself, missing related work, overlap with existing methods and the importance of the problem.")),
}


def assignment(ids: list[str]) -> dict[str, dict[str, str]]:
    path = OUT / "assignment.json"
    if path.exists():
        return json.loads(path.read_text())["papers"]
    out = {}
    for pid in ids:
        roles = [r for r in ROLES for _ in range(2)]
        random.Random(f"{SEED}:m7:{pid}").shuffle(roles)
        out[pid] = dict(zip(e14.PANEL, roles, strict=True))
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"created": "2026-09-30", "seed": f"{SEED}:m7:<paper_id>", "papers": out}, indent=1))
    return out


def manifest() -> pd.DataFrame:
    m = pd.DataFrame(json.loads(e14.MANIFEST.read_text())["papers"]).set_index("paper_id")
    if "venue_key" not in m:
        m["venue_key"] = e14.CORPUS
    return m


def load_points() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for line in (CONTENT / "reference_items.jsonl").open():
        w = json.loads(line); out.setdefault(w["paper_id"], []).append(w)
    return out


def run(args):
    man = manifest()
    ids = sorted(man.index)
    roles = assignment(ids)
    client, model_of, lock = OpenRouterClient(CACHE, args.max_usd), models(), threading.Lock()
    path = OUT / "records.jsonl"
    done, fails, truncated = set(), {}, set()
    if path.exists():
        for r in map(json.loads, path.open()):
            key = (r["paper_id"], r["model_id"])
            if r["parse_ok"]:
                done.add(key)
            else:
                fails[key] = fails.get(key, 0) + 1
                if (r.get("usage") or {}).get("finish_reason") == "length":
                    truncated.add(key)
    jobs = [(p, m) for p in ids for m in e14.PANEL if (p, m) not in done]
    print(f"{len(ids)} papers, {len(jobs)} cells to run ({len(done)} done)", flush=True)

    def work(job):
        pid, key = job
        model, paper, role = model_of[key], man.loc[pid], roles[pid][key]
        venue = venues()[paper.venue_key]
        name, desc = ROLES[role]

        def render(text):
            return prompts.render("call_a_role", own_review_prompt=prompts.own_review(venue, text),
                                  role_name=name, role_description=desc)

        text_in, trimmed = fit_to_context(paper_text(paper), model, len(render("").encode()))
        prompt = render(text_in)
        temperature = TEMPERATURE if model.accepts_temperature else None
        first = fails.get((pid, key), 0)
        for attempt in range(first, first + MAX_ATTEMPTS):
            # a cell cut off at the output limit is retried with twice the limit
            limit = 2 * MAX_OUTPUT_TOKENS if (pid, key) in truncated else MAX_OUTPUT_TOKENS
            kwargs = dict(run_idx=0, temperature=temperature, max_tokens=limit)
            if attempt:
                kwargs["attempt"] = attempt
            if model.request:
                kwargs["extra"] = model.request
            started = time.perf_counter()
            raw, usage = client.complete(model.api_model, prompt, **kwargs)
            parsed = e14.parse_round2(raw, venue.scores)   # call-A parser + the schema fallback used in E14, E8, E8b
            rec = dict(paper_id=pid, model_id=key, role=role, prompt_template="call_a_role",
                       prompt_version=prompts.load_template("call_a_role")[0]["version"],
                       prompt_sha256=prompts.sha256(prompt), emitted_score=parsed.score, items=parsed.items,
                       reasoning_text=parsed.reasoning, raw_response=raw, parse_ok=parsed.ok, parse_error=parsed.error,
                       extra=parsed.extra, paper_text_context_trimmed=trimmed, temperature=temperature,
                       attempt=attempt + 1, execution_model=usage.get("execution_model"),
                       usage={k: usage.get(k) for k in ("input_tokens", "output_tokens", "reasoning_tokens",
                                                        "cost_usd", "cached", "finish_reason", "provider")},
                       latency_s=round(time.perf_counter() - started, 3),
                       timestamp=datetime.datetime.now(datetime.UTC).isoformat())
            rec["max_output_tokens"] = limit
            with lock, path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if parsed.ok:
                return rec
            if usage.get("finish_reason") == "length":
                truncated.add((pid, key))
            print(f"  parse failed: {key} {pid} attempt {attempt + 1}: {parsed.error}", flush=True)
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


def role_records() -> dict:
    full = e14.FULL or e14.CORPUS != "iclr2022"
    return {(r["paper_id"], r["model_id"]): r for r in map(json.loads, (OUT / "records.jsonl").open())
            if r["parse_ok"] and (full or r.get("timestamp", "") < PREREG_CUTOFF)}


def coverage(args):
    """Label each role review against the paper's human-raised points with the round-1 matcher."""
    from peerreview import content
    annotator = models()["gemini-3.5-flash-lite"]
    client = OpenRouterClient(ROOT / "cache" / "responses" / "content_match_m7", args.max_usd)
    points = load_points()
    path = OUT / "coverage_labels.csv"
    done = set(zip(*[pd.read_csv(path)[c] for c in ("paper_id", "model")])) if path.exists() else set()
    jobs = [(k, r) for k, r in role_records().items() if k not in done and k[0] in points and r["items"]]   # not score-only
    print(f"{len(done)} labeled, {len(jobs)} to do", flush=True)
    lock = threading.Lock()

    def work(job):
        (pid, m), r = job
        summary = (r.get("extra") or {}).get("summary", "")
        summary = " ".join(map(str, summary)) if isinstance(summary, list) else str(summary or "")
        rec = content.match_review(client, annotator, points[pid], r["items"], summary=summary, run_key=0)
        if not rec["parse_ok"]:
            return 0
        rows = [dict(paper_id=pid, model=m, role=r["role"], id=i, recognition=v[0], stance=v[1])
                for i, v in rec["labels"].items()]
        with lock:
            pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
        return 1

    with ThreadPoolExecutor(args.workers) as pool:
        ok = sum(pool.map(work, jobs))
    print(f"labeled {ok} of {len(jobs)}, ${client.spent:.2f}")


def analyze(args):
    man = manifest()
    roles = role_records()
    base = {}
    for m in e14.PANEL:
        for venue_key, tag in e14.ROUND1_SOURCES:
            for r in records.iter_records(m, venue_key, "paper_review", "A", tag=tag):
                if r["parse_ok"] and r["items"] and r["run_idx"] in BASE_RUNS:
                    base[(r["paper_id"], m, r["run_idx"])] = r["emitted_score"]
    keep = sorted(p for p in man.index if all((p, m) in roles and all((p, m, k) in base for k in BASE_RUNS)
                                               for m in e14.PANEL))
    md = {p: man.loc[p].to_dict() for p in keep}
    sc = lambda p, x: e14.model_score(md[p], x)   # raw on the pre-ChatGPT corpora, [0, 1] per venue on 2023-25
    h = np.array([e14.human_score(md[p]) for p in keep], float)
    truth = np.array([{"accept": 1.0, "reject": 0.0}.get(man.loc[p, "outcome"], np.nan) for p in keep])
    M = {"role": np.array([[sc(p, roles[(p, m)]["emitted_score"]) for m in e14.PANEL] for p in keep], float)}
    for k in BASE_RUNS:
        M[f"run{k}"] = np.array([[sc(p, base[(p, m, k)]) for m in e14.PANEL] for p in keep], float)

    # coverage: panel (any of 8), single review, and for weaknesses raised by one vs. two or more humans
    weak = {}
    for line in (CONTENT / "reference_items.jsonl").open():
        w = json.loads(line)
        if w["polarity"] == "weakness":
            weak.setdefault(w["paper_id"], {})[w["id"]] = len({s["review"] for s in w["sources"]})
    base_cov = pd.read_csv(CONTENT / "coverage_labels.csv")
    base_cov = base_cov[base_cov.model.isin(e14.PANEL) & base_cov.recognition.isin(["specific", "partial"])]
    role_cov = pd.read_csv(OUT / "coverage_labels.csv") if (OUT / "coverage_labels.csv").exists() else pd.DataFrame(
        columns=["paper_id", "model", "id", "recognition"])
    role_cov = role_cov[role_cov.recognition.isin(["specific", "partial"])]
    hits = {"role": {(p, m): set(g.id) for (p, m), g in role_cov.groupby(["paper_id", "model"])}}
    for k in BASE_RUNS:
        b = base_cov[base_cov.run == k]
        hits[f"run{k}"] = {(p, m): set(g.id) for (p, m), g in b.groupby(["paper_id", "model"])}
    labeled_role = set(zip(pd.read_csv(OUT / "coverage_labels.csv").paper_id, pd.read_csv(OUT / "coverage_labels.csv").model)) \
        if (OUT / "coverage_labels.csv").exists() else set()
    cov_papers = [i for i, p in enumerate(keep) if p in weak and all((p, m) in labeled_role for m in e14.PANEL)]

    def cov_stats(arm, idx):
        panel, single, one, many = [], [], [], []
        for i in idx:
            p = keep[i]
            if i not in cov_set:
                continue
            ws = weak[p]
            got = [hits[arm].get((p, m), set()) & set(ws) for m in e14.PANEL]
            union = set().union(*got)
            panel.append(len(union) / len(ws))
            single.append(np.mean([len(g) / len(ws) for g in got]))
            o = [w for w, n in ws.items() if n == 1]
            t = [w for w, n in ws.items() if n >= 2]
            if o:
                one.append(len(union & set(o)) / len(o))
            if t:
                many.append(len(union & set(t)) / len(t))
        return dict(panel_coverage=np.mean(panel), single_review_coverage=np.mean(single),
                    panel_coverage_one_human=np.mean(one), panel_coverage_two_plus_humans=np.mean(many))

    cov_set = set(cov_papers)
    iu = np.triu_indices(len(e14.PANEL), 1)

    def stats(X, idx):
        X, hh, tt = X[idx], h[idx], truth[idx]
        E = X - hh[:, None]
        R = np.column_stack([X[:, j] - np.polyval(np.polyfit(hh, X[:, j], 1), hh) for j in range(X.shape[1])])
        mean = X.mean(axis=1)
        ok = ~np.isnan(tt)
        pos, neg = mean[ok][tt[ok] == 1], mean[ok][tt[ok] == 0]
        auroc = float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))
        return dict(error_correlation=float(np.nanmean(np.corrcoef(E.T)[iu])),
                    error_correlation_calibration_free=float(np.nanmean(np.corrcoef(R.T)[iu])),
                    within_paper_sd=float(X.std(axis=1, ddof=1).mean()),
                    committee_abs_error=float(np.abs(mean - hh).mean()), decision_auroc=auroc,
                    r_panel_mean_human=float(np.corrcoef(mean, hh)[0, 1]))

    def all_stats(idx):
        out = {}
        for arm, X in M.items():
            s = stats(X, idx)
            if cov_papers:
                s.update(cov_stats(arm, idx))
            out[arm] = s
        out["baseline"] = {k: float(np.mean([out[f"run{j}"][k] for j in BASE_RUNS])) for k in out["role"]}
        return out

    everything = np.arange(len(keep))
    point = all_stats(everything)
    rng = np.random.default_rng(SEED)
    boots = [all_stats(rng.integers(0, len(keep), len(keep))) for _ in range(args.boot)]
    rows = []
    for k in point["role"]:
        d = [b["role"][k] - b["baseline"][k] for b in boots]
        rows.append(dict(outcome=k, role=point["role"][k], **{("baseline_mean_of_3_runs" if len(BASE_RUNS) == 3 else
                                                                 "baseline_run0"): point["baseline"][k]},
                         **{f"run{j}": point[f"run{j}"][k] for j in BASE_RUNS},
                         role_minus_baseline=point["role"][k] - point["baseline"][k],
                         ci_low=float(np.nanpercentile(d, 2.5)), ci_high=float(np.nanpercentile(d, 97.5))))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / ("outcomes_full.csv" if e14.FULL and e14.CORPUS == "iclr2022" else "outcomes.csv"), index=False)
    print(f"{len(keep)} papers with all 8 role reviews and baseline runs {BASE_RUNS}; coverage on {len(cov_papers)}")
    print(table.round(3).to_string(index=False))
    # exploratory: does a role raise its reviewer's coverage of weaknesses in that role's categories?
    print("\nroles:", pd.Series([r["role"] for r in roles.values()]).value_counts().to_dict())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--max-usd", type=float, required=True)
    r.add_argument("--workers", type=int, default=24)
    r.set_defaults(run=run)
    c = sub.add_parser("coverage")
    c.add_argument("--max-usd", type=float, required=True)
    c.add_argument("--workers", type=int, default=24)
    c.set_defaults(run=coverage)
    a = sub.add_parser("analyze")
    a.add_argument("--boot", type=int, default=N_BOOT)
    a.set_defaults(run=analyze)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
