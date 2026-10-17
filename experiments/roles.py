"""Roles: each panel model writes a first review with an assigned focus.

The four roles (soundness, experiments, clarity, novelty) are given twice each per paper, rotated across the eight
models by a seeded shuffle, so a role is not tied to a model family; the reviewer still reviews the whole paper and
gives the usual overall score. The role per paper and model is in assignment.json. The venue is chosen with
PANEL_CORPUS, as in experiments/discussion.py.

Usage:  uv run python experiments/roles.py run --max-usd 170
        uv run python experiments/roles.py coverage --max-usd 10      # coverage judge on the role reviews
Writes: results/<folder>/roles/{assignment.json, records.jsonl, coverage_labels.csv}, where <folder> is
iclr2021_prechatgpt, iclr2022_prechatgpt, neurips2022_prechatgpt, or study2325.
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

OUT = discussion.OUT.parent / "roles"
CONTENT = ROOT / "results" / ("study2325_panel8f" if discussion.CORPUS == "2023-25"
                              else f"{discussion.CORPUS}_prechatgpt") / "content"
CACHE = ROOT / "cache" / "responses" / "role_panel"
ROLE_SEED = f"{SEED}:roles"   # seeds the role assignment of papers without one in assignment.json
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
        random.Random(f"{ROLE_SEED}:{pid}").shuffle(roles)
        out[pid] = dict(zip(discussion.PANEL, roles, strict=True))
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": f"{ROLE_SEED}:<paper_id>", "papers": out}, indent=1))
    return out


def manifest() -> pd.DataFrame:
    m = pd.DataFrame(json.loads(discussion.MANIFEST.read_text())["papers"]).set_index("paper_id")
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
    jobs = [(p, m) for p in ids for m in discussion.PANEL if (p, m) not in done]
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
            parsed = discussion.parse_round2(raw, venue.scores)
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
    return {(r["paper_id"], r["model_id"]): r for r in map(json.loads, (OUT / "records.jsonl").open()) if r["parse_ok"]}


def coverage(args):
    """Label each role review with points against the paper's human-raised points, with the coverage judge that
    labels the first reviews (same model and prompt). Writes coverage_labels.csv and match_records.jsonl; a review
    whose labels did not parse is sampled afresh on a rerun."""
    annotator = models()[args.annotator]
    client = OpenRouterClient(CACHE.parent / f"content_match_roles_{args.annotator}", args.max_usd)
    points = load_points()
    path, log_path = OUT / "coverage_labels.csv", OUT / "match_records.jsonl"
    done = set(zip(*[pd.read_csv(path)[c] for c in ("paper_id", "model")])) if path.exists() else set()
    failed: dict[tuple, int] = {}
    if log_path.exists():
        for line in log_path.open():
            r = json.loads(line)
            if not r["parse_ok"]:
                failed[(r["paper_id"], r["model"])] = max(failed.get((r["paper_id"], r["model"]), 0), r.get("attempt", 1))
    jobs = [(k, r) for k, r in role_records().items() if k not in done and k[0] in points and r.get("items")]
    print(f"{len(done)} labelled, {len(jobs)} to do", flush=True)
    lock = threading.Lock()

    def work(job):
        (pid, m), r = job
        summary = (r.get("extra") or {}).get("summary", "")
        summary = " ".join(map(str, summary)) if isinstance(summary, list) else str(summary or "")
        rec = content.match_review(client, annotator, points[pid], r["items"], summary=summary, run_key=0,
                                   attempt_offset=failed.get((pid, m), 0))
        with lock, log_path.open("a") as log:
            log.write(json.dumps(dict(paper_id=pid, model=m, role=r["role"], **rec)) + "\n")
        if not rec["parse_ok"]:
            return 0
        rows = [dict(paper_id=pid, model=m, role=r["role"], id=i, recognition=v[0], stance=v[1])
                for i, v in rec["labels"].items()]
        with lock:
            pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
        return 1

    with ThreadPoolExecutor(args.workers) as pool:
        ok = sum(pool.map(work, jobs))
    print(f"labelled {ok} of {len(jobs)}, ${client.spent:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=24)
    r.add_argument("--max-usd", type=float, required=True)
    r.set_defaults(run=run)
    c = sub.add_parser("coverage")
    c.add_argument("--annotator", default="gemini-3.5-flash-lite", choices=sorted(models()))
    c.add_argument("--workers", type=int, default=24)
    c.add_argument("--max-usd", type=float, required=True)
    c.set_defaults(run=coverage)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
