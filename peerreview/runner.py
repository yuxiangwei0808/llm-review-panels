"""Run models on a set of papers and save one record per call.

Per paper, model and run:

* call ``A``: the model reviews the paper (title withheld) and chooses a score;
* call ``F_holistic``: the model scores the paper from the human reviews, ratings removed, instead of its own review.

Records follow the layout in ``records.py``. A run can be interrupted and restarted: cells already saved with a
successful parse are skipped, and the client's cache means no request is paid for twice.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from . import parsing, prompts, records
from .config import MAX_OUTPUT_TOKENS, RAW_OUTPUTS, TEMPERATURE, Model, Venue, models, venues
from .papers import load_dossier, paper_text

SAFETY_MARGIN_TOKENS = 1024
MAX_ATTEMPTS = 3            # fresh samples of a cell whose response does not parse
PROGRESS_EVERY = 25         # paper-runs between progress lines
CONTEXT_NOTE = ("\n\nCONTEXT-LIMIT NOTE: the paper text was cut here to fit this model's context "
                "window. Material after this point exists in the submission but is not shown.")
TEMPLATE = {"A": "call_a", "F_holistic": "call_f_holistic"}
STAGE = {"A": "paper_review", "F_holistic": "fixed_evidence"}


def fit_to_context(text: str, model: Model, prompt_overhead_bytes: int) -> tuple[str, bool]:
    """Trim the paper only if the prompt cannot fit; one token is at least one UTF-8 byte."""
    budget = model.context_window_tokens - MAX_OUTPUT_TOKENS - SAFETY_MARGIN_TOKENS \
        - prompt_overhead_bytes - len(CONTEXT_NOTE.encode())
    encoded = text.encode()
    if len(encoded) <= budget:
        return text, False
    return encoded[:budget].decode("utf-8", errors="ignore") + CONTEXT_NOTE, True


class RunWriter:
    """Appends records and saves the prompt and paper-text artifacts of one run folder."""

    def __init__(self, folder: Path, stage: str):
        self.folder, self.path = folder, folder / records.STAGE_FILE[stage]
        (folder / "input_artifacts").mkdir(parents=True, exist_ok=True)
        (folder / "paper_inputs").mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.done = set()
        self.attempts: dict[tuple, int] = {}    # failed cells: highest attempt recorded so far
        if self.path.exists():
            for line in self.path.open():
                r = json.loads(line)
                key = (r["paper_id"], r["run_idx"], r["call_type"])
                # A review without strengths or weaknesses is sampled again: the coverage judge needs its points.
                if r["parse_ok"] and not (r["call_type"] == "A" and not r["items"]):
                    self.done.add(key)
                else:
                    self.attempts[key] = max(self.attempts.get(key, 0), r.get("attempt", 1))

    def artifact(self, subfolder: str, text: str, suffix: str) -> tuple[str, str]:
        digest = prompts.sha256(text)
        relative = f"{subfolder}/{digest}{suffix}"
        target = self.folder / relative
        if not target.exists():
            target.write_text(text)
        return relative, digest

    def append(self, record: dict):
        with self.lock, self.path.open("a") as out:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.done.add((record["paper_id"], record["run_idx"], record["call_type"]))


def _call(client, writer: RunWriter, model: Model, venue: Venue, paper_id: str, run_idx: int,
          call_type: str, prompt: str, parse, extra: dict) -> dict:
    """One cell: request, parse, record. A response that does not parse is recorded and the cell
    is sampled again (a fresh request, not the cached one) up to MAX_ATTEMPTS times; the record's
    ``attempt`` counts from 1. The returned record is the last one written."""
    artifact, prompt_sha = writer.artifact("input_artifacts", prompt, ".txt")
    header, _ = prompts.load_template(TEMPLATE[call_type])
    temperature = TEMPERATURE if model.accepts_temperature else None
    # A restarted run continues numbering after the failed attempts already on disk, so it
    # draws new samples instead of re-reading the cached failures.
    first = writer.attempts.get((paper_id, run_idx, call_type), 0)
    for attempt in range(first, first + MAX_ATTEMPTS):
        started = time.perf_counter()
        kwargs = dict(run_idx=run_idx, temperature=temperature, max_tokens=MAX_OUTPUT_TOKENS)
        if attempt:
            kwargs["attempt"] = attempt
        if model.request:
            kwargs["extra"] = model.request
        text, usage = client.complete(model.api_model, prompt, **kwargs)
        latency = time.perf_counter() - started
        parsed: parsing.Parsed = parse(text)
        record = _record(writer, model, venue, paper_id, run_idx, call_type, prompt_sha, artifact,
                         header, temperature, text, usage, parsed, latency, attempt + 1, extra)
        if parsed.ok:
            return record
        print(f"  parse failed: {model.key} {paper_id} run {run_idx} {call_type} attempt "
              f"{attempt + 1}: {parsed.error}", flush=True)
    return record


def _record(writer, model, venue, paper_id, run_idx, call_type, prompt_sha, artifact, header,
            temperature, text, usage, parsed, latency, attempt, extra) -> dict:
    identity = f"{model.key}|{venue.key}|{paper_id}|{call_type}|{run_idx}|{prompt_sha}"
    record = dict(
        call_id=hashlib.sha256(identity.encode()).hexdigest()[:24], block=STAGE[call_type],
        call_type=call_type, mode="sampled", paper_id=paper_id, model_id=model.key,
        run_idx=run_idx, model_snapshot=usage.get("execution_model", model.api_model),
        temperature=temperature, prompt_hash=prompts.sha256(prompts.load_template(
            TEMPLATE[call_type])[1]), prompt_version=header["version"], input_hash=prompt_sha,
        execution_model=usage.get("execution_model", model.api_model),
        emitted_score=parsed.score, accept_prob=parsed.accept_prob, items=parsed.items,
        reasoning_text=parsed.reasoning, raw_response=text, parse_ok=parsed.ok,
        parse_error=parsed.error,
        usage={k: usage.get(k) for k in ("input_tokens", "output_tokens", "reasoning_tokens",
                                         "cost_usd", "cached", "finish_reason",
                                         "native_finish_reason", "provider")},
        latency_s=round(latency, 3), timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        attempt=attempt, extra=dict(parsed.extra, **extra, rendered_prompt_artifact=artifact,
                                    rendered_prompt_sha256=prompt_sha,
                                    max_output_tokens=MAX_OUTPUT_TOKENS))
    writer.append(record)
    return record


def _with_points(parsed: parsing.Parsed) -> parsing.Parsed:
    """A review with no strengths or weaknesses is treated as unparsed and sampled again."""
    if parsed.ok and not parsed.items:
        return parsing.Parsed(ok=False, error="no review points", reasoning=parsed.reasoning)
    return parsed


def run_paper(client, model: Model, venue: Venue, paper: pd.Series, run_idx: int,
              calls: tuple[str, ...], review: RunWriter, fixed: RunWriter) -> None:
    pid = paper.paper_id
    if "A" in calls and (pid, run_idx, "A") not in review.done:
        overhead = len(prompts.own_review(venue, "").encode())
        text, trimmed = fit_to_context(paper_text(paper), model, overhead)
        paper_artifact, paper_sha = review.artifact("paper_inputs", text, ".md")
        _call(client, review, model, venue, pid, run_idx, "A", prompts.own_review(venue, text),
              lambda raw: _with_points(parsing.parse_own_review(raw, venue.scores)),
              dict(paper_text_artifact=paper_artifact, paper_text_sent_sha256=paper_sha,
                   paper_text_context_trimmed=trimmed))
    if "F_holistic" in calls and (pid, run_idx, "F_holistic") not in fixed.done:
        dossier = load_dossier(venue.key, pid)
        _call(client, fixed, model, venue, pid, run_idx, "F_holistic",
              prompts.shared_evidence(venue, dossier["holistic_text"]),
              lambda raw: parsing.parse_rating_lines(raw, venue.scores), dict(form="holistic"))


def run(client, venue_key: str, model_keys: list[str], papers: pd.DataFrame, tag: str,
        calls: tuple[str, ...] = ("A",), runs: int = 1, workers: int = 8, root: Path = RAW_OUTPUTS) -> None:
    """Run ``calls`` for every paper x model x run and write ``<root>/<model>/<venue>/<stage>_<tag>``."""
    venue = venues()[venue_key]
    papers = papers[papers.venue_key == venue_key]
    for key in model_keys:
        model = models()[key]
        writers = {stage: RunWriter(records.run_dir(key, venue_key, stage, root, tag), stage)
                   for stage in ("paper_review", "fixed_evidence")}
        for writer in writers.values():
            (writer.folder / "run_meta.json").write_text(json.dumps(dict(
                venue_key=venue.key, venue_name=venue.name, scale=list(venue.scores),
                accept_threshold=venue.accept_threshold, tag=tag), indent=1) + "\n")
            papers.to_json(writer.folder / "manifest.jsonl", orient="records", lines=True)
        jobs = [(paper, run_idx) for _, paper in papers.iterrows() for run_idx in range(runs)]
        errors, started = [], time.perf_counter()
        print(f"{key}: {len(jobs)} paper-runs over {len(papers)} papers, calls {list(calls)}, "
              f"{workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_paper, client, model, venue, paper, run_idx, calls,
                                   writers["paper_review"], writers["fixed_evidence"]):
                       (paper.paper_id, run_idx) for paper, run_idx in jobs}
            for done, future in enumerate(as_completed(futures), start=1):
                paper_id, run_idx = futures[future]
                try:
                    future.result()
                except Exception as error:      # noqa: BLE001 - one cell must not stop the run
                    errors.append(dict(paper_id=paper_id, run_idx=run_idx, error=repr(error),
                                       traceback=traceback.format_exc()))
                    print(f"  ERROR {key} {paper_id} run {run_idx}: {error!r}", flush=True)
                if done % PROGRESS_EVERY == 0 or done == len(jobs):
                    elapsed = time.perf_counter() - started
                    print(f"  {key}: {done}/{len(jobs)} paper-runs, {elapsed / 60:.1f} min, "
                          f"${client.spent:.2f} spent, {len(errors)} errors", flush=True)
        if errors:
            log = writers["paper_review"].folder / "errors.jsonl"
            with log.open("a") as out:
                for error in errors:
                    out.write(json.dumps(dict(error, model=key, venue=venue.key,
                                              timestamp=datetime.datetime.now(
                                                  datetime.UTC).isoformat())) + "\n")
        print(f"{key}: {len(jobs)} paper-runs done, ${client.spent:.2f} spent, {len(errors)} "
              f"errors{' (see errors.jsonl)' if errors else ''}", flush=True)
