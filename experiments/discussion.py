"""Discussion and rebuttal: LLM reviewers revise after reading each other's reviews or the author responses.

Round 1 is the existing own-evidence review (call A, run 0) of each panel model. Round 2 has two
arms, one sample per model and paper:
  second_look  the reviewer sees the paper and its own round-1 review (share nothing, control)
  discussion   the same plus the other panel members' round-1 reviews, text and score, anonymized
Both arms reuse the full call-A prompt, so the output parses with parse_own_review.

Usage:
  uv run python experiments/discussion.py run --pilot 10 --max-usd 20      # pilot papers only
  uv run python experiments/discussion.py run --max-usd 300                # all 225 papers
  uv run python experiments/discussion.py analyze
Also a stage-3 arm, rebuttal: the reviewer revises after the paper's real author responses
(the draft's call_a_revision prompt); second_look is its matched no-new-information control.
  uv run python experiments/discussion.py run --arms rebuttal --max-usd 200
  uv run python experiments/discussion.py analyze-rebuttal
Writes: results/iclr2022_prechatgpt/discussion/records.jsonl   (every round-2 call, append-only)
        results/iclr2022_prechatgpt/discussion/records_rebuttal.jsonl (stage-3 calls)
        results/iclr2022_prechatgpt/discussion/outcomes.csv     (preregistered outcomes by arm)
Responses are cached under cache/responses/discussion, so a rerun is free and resumes.
"""
from __future__ import annotations

import argparse
import datetime
import functools
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peerreview import parsing, prompts, records
from peerreview.client import CostCeilingReached, OpenRouterClient
from peerreview.config import MAX_OUTPUT_TOKENS, ROOT, SEED, TEMPERATURE, models, venues
from peerreview.papers import load_dossier, paper_text
from peerreview.runner import MAX_ATTEMPTS, fit_to_context

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forums as forums
from verdict_mask import masked, verdict_masked

# Corpus: ICLR 2022 (default) or the 2023-25 replication (PANEL_CORPUS=2023-25): the 180-paper sample
# frozen in results/study2325/discussion/sample.json, round-1 reviews from the panel8f runs, scores
# normalized to [0, 1] per venue in the analysis (ICML 2025 uses a 1-5 scale).
CORPUS = os.environ.get("PANEL_CORPUS", "iclr2022")
if CORPUS == "2023-25":
    MANIFEST = ROOT / "data" / "manifests" / "study_manifest.json"
    OUT = ROOT / "results" / "study2325" / "discussion"
    ROUND1_SOURCES = [("iclr2025", "panel8f"), ("neurips2023", "panel8f"), ("icml2025", "panel8f")]
elif CORPUS in ("iclr2021", "neurips2022"):   # S4, S5: more pre-ChatGPT venues (raw 1-10 scores, one venue each)
    MANIFEST = ROOT / "data" / "manifests" / f"{CORPUS}_prechatgpt_run.json"
    OUT = ROOT / "results" / f"{CORPUS}_prechatgpt" / "discussion"
    ROUND1_SOURCES = [(CORPUS, "prechatgpt")]
else:
    MANIFEST = ROOT / "data" / "manifests" / "iclr2022_prechatgpt.json"
    OUT = ROOT / "results" / "iclr2022_prechatgpt" / "discussion"
    ROUND1_SOURCES = [("iclr2022", "prechatgpt")]
CACHE = ROOT / "cache" / "responses" / "discussion"
# Complete-sample extensions (2026-10-02): SAMPLE_SCOPE=full runs E14 and the rebuttal arm on every sampled paper and
# writes *_full outputs; without it the analyses keep the original (preregistered) paper sets.
FULL = os.environ.get("SAMPLE_SCOPE") == "full"
PANEL = ["gpt-5.6-terra", "gemini-3.1-pro", "claude-sonnet-5", "deepseek-v4-flash-0731", "glm-5.2",
         "qwen3.8-2.4t-a95b", "grok-4.7", "kimi-k3"]
ARMS = {"second_look": "call_a_second_look", "discussion": "call_a_discussion"}   # E14 (stage 4)
STAGE3 = {"rebuttal": "call_a_revision",       # stage 3: revise after the real author responses (draft's prompt)
          "llm_rebuttal": "call_a_revision"}   # ... after an LLM-written rebuttal targeted at this review
AUTHOR_MODEL = "gpt-5.1"   # writes the llm_rebuttal responses (author_targeted_v2); not on the 8-family panel
E14B = {"discussion_text": "call_a_discussion_text",     # E14b: peers' reviews without their scores
        "discussion_masked": "call_a_discussion_text"}   # E14c: ... and with verdict sentences masked
M6 = {"discussion_independent": "call_a_discussion_independent"}   # M6: E14's peers + "keep your judgment independent"
TEMPLATES = {**ARMS, **STAGE3, **E14B, **M6}
ROUND1_RUN, N_BOOT = 0, 2000
# R2a (preregistered second run): E14_RUN=2 starts from round-1 run 1 and writes to discussion_run2/, so the
# reviews peers read and the revisions are both new samples; ICLR 2022 only.
if os.environ.get("E14_RUN") in ("2", "3"):   # R2a second run; R3 third run (round-1 run 2) if the runs disagree
    assert CORPUS == "iclr2022", "the repeat runs are preregistered for ICLR 2022 only"
    ROUND1_RUN = int(os.environ["E14_RUN"]) - 1
    OUT = ROOT / "results" / "iclr2022_prechatgpt" / f"discussion_run{os.environ['E14_RUN']}"


def record_file(arm: str) -> Path:
    """Stage-3 records go to their own file, so the E14 and rebuttal runs can append in parallel."""
    return OUT / ("records_rebuttal.jsonl" if arm in STAGE3 else "records.jsonl")


def all_records() -> list[dict]:
    out = []
    for name in ("records.jsonl", "records_rebuttal.jsonl"):
        if (OUT / name).exists():
            out += [json.loads(line) for line in (OUT / name).open()]
    return out


def round1(papers: set[str]) -> dict[tuple[str, str], dict]:
    """(paper, model) -> the parsed round-1 own review of run 0 (the last parsed attempt)."""
    out = {}
    for m in PANEL:
        for venue_key, tag in ROUND1_SOURCES:
            try:
                recs = records.iter_records(m, venue_key, "paper_review", "A", tag=tag)
                for r in recs:
                    if r["paper_id"] in papers and r["run_idx"] == ROUND1_RUN and r["parse_ok"] and r["items"]:
                        out[(r["paper_id"], m)] = r
            except FileNotFoundError:
                continue
    return out


def _extract_json(raw: str) -> tuple[str, dict | None]:
    """(reasoning, the review object) from a response, removing premature closing braces (DeepSeek
    V4 Flash sometimes closes the object right after the summary)."""
    reasoning, rest = parsing._reasoning(raw)
    rest = rest.replace("```json", "").replace("```", "")
    start, end = rest.find("{"), rest.rfind("}")
    if start < 0 or end <= start:
        return reasoning, None
    blob = parsing.ESCAPE.sub(parsing._keep_valid_escape, rest[start:end + 1])
    for _ in range(4):
        try:
            obj = json.loads(blob)
            return reasoning, obj if isinstance(obj, dict) else None
        except json.JSONDecodeError as error:
            cut = blob.rfind("}", 0, error.pos) if error.msg.startswith("Extra data") else -1
            if cut < 0:
                return reasoning, None
            blob = blob[:cut] + blob[cut + 1:]
    return reasoning, None


def _normalize(obj: dict) -> tuple[dict, bool]:
    """Map the schema variants seen in round 2 onto the call-A schema. Returns (review, has_points).
    Variants: fields nested under "review"; rating as a bare number or {"score": n}; overall_score or
    confidence at the top level; prior-knowledge fields at the top level; strengths and weaknesses
    merged into one list (then the points cannot be split and only the score is kept)."""
    flat = dict(obj)
    for key in ("review", "prior_knowledge"):
        if isinstance(flat.get(key), dict):
            flat = {**flat.pop(key), **flat}
    rating = flat.get("rating")
    score = None
    if isinstance(rating, dict):
        score = rating.get("overall_score", rating.get("score"))
    elif isinstance(rating, int | float | str):
        score = rating
    score = flat.get("overall_score", score) if score is None else score
    confidence = rating.get("confidence") if isinstance(rating, dict) else flat.get("confidence")
    strengths = flat.get("strengths") if isinstance(flat.get("strengths"), list) else []
    weaknesses = flat.get("weaknesses") if isinstance(flat.get("weaknesses"), list) else []
    to_text = lambda x: x if isinstance(x, str) else " ".join(str(v) for v in x.values()) if isinstance(x, dict) else str(x)
    review = dict(prior_knowledge={"seen_before": flat.get("seen_before", ""), "explanation": flat.get("explanation", "")},
                  summary=flat.get("summary", ""), strengths=[to_text(x) for x in strengths],
                  weaknesses=[to_text(x) for x in weaknesses],
                  rating={"overall_score": str(score) if score is not None else "", "confidence": confidence})
    return review, bool(strengths or weaknesses)


def parse_round2(raw: str, allowed) -> parsing.Parsed:
    """parse_own_review; if that fails, the schema-normalizing fallback. A fallback parse with a
    valid score but no separable points is kept as score-only (extra.score_only = True): it counts
    for the score outcomes and not for weakness coverage."""
    parsed = parsing.parse_own_review(raw, allowed)
    if parsed.ok and parsed.items:
        return parsed
    reasoning, obj = _extract_json(raw)
    if obj is not None:
        review, has_points = _normalize(obj)
        fixed = f"<reasoning>\n{reasoning}\n</reasoning>\n\n{json.dumps(review)}"
        again = parsing.parse_own_review(fixed, allowed)
        if again.ok:
            again.extra["schema_normalized"] = True
            if not (has_points and again.items):
                again.extra["score_only"] = True
            return again
    # last resort: the first allowed integer under any key naming a score or rating
    for m in re.finditer(r'"[^"]*(?:score|rating)[^"]*"\s*:\s*"?(\d+)', raw, re.IGNORECASE):
        if int(m.group(1)) in allowed and "confidence" not in m.group(0).lower():
            return parsing.Parsed(ok=True, score=int(m.group(1)), reasoning=reasoning,
                                  extra={"score_only": True, "score_by_key_pattern": True})
    if parsed.ok and not parsed.items:
        return parsing.Parsed(ok=False, error="no review points", reasoning=parsed.reasoning)
    return parsed


def review_text(record: dict) -> str:
    """A round-1 review as the reviewer wrote it: justification, summary, strengths, weaknesses."""
    extra = record.get("extra") or {}
    lines = []
    if record.get("reasoning_text"):
        lines += ["Justification:", record["reasoning_text"].strip(), ""]
    if extra.get("summary"):
        lines += ["Summary:", str(extra["summary"]).strip(), ""]
    for polarity, heading in (("strength", "Strengths:"), ("weakness", "Weaknesses:")):
        items = [i for i in record["items"] if i["polarity"] == polarity]
        if items:
            lines.append(heading)
            lines += [f"- [{i['dimension']}] {i['text']}" if i.get("dimension") else f"- {i['text']}" for i in items]
            lines.append("")
    return "\n".join(lines).strip()


def peer_block(paper_id: str, model: str, reviews: dict, with_scores: bool = True, mask_verdicts: bool = False) -> str:
    peers = [m for m in PANEL if m != model and (paper_id, m) in reviews]
    random.Random(f"{SEED}:e14:{paper_id}:{model}").shuffle(peers)   # same order in E14 and E14b
    head = lambda k, m: (f"### Reviewer {k} (score: {reviews[(paper_id, m)]['emitted_score']})" if with_scores
                         else f"### Reviewer {k}")
    body = lambda m: (verdict_masked(masked(review_text(reviews[(paper_id, m)]))) if mask_verdicts
                      else review_text(reviews[(paper_id, m)]))
    return "\n\n".join(f"{head(k, m)}\n\n{body(m)}" for k, m in enumerate(peers, start=1))


def author_rebuttal(client, venue, paper: pd.Series, own: dict) -> tuple[str, dict]:
    """The draft's S1 condition: an LLM acting as the authors drafts a rebuttal to this reviewer's
    review, using only the paper and the real author response as sources (author_targeted_v2)."""
    author = models()[AUTHOR_MODEL]
    prompt = prompts.render("author_targeted_v2", paper_text=paper_text(paper), source_review=review_text(own),
                            initial_score=str(own["emitted_score"]), real_response=real_author_response(paper.paper_id))
    kwargs = dict(run_idx=0, temperature=TEMPERATURE if author.accepts_temperature else None, max_tokens=4096)
    if author.request:
        kwargs["extra"] = author.request
    text, usage = client.complete(author.api_model, prompt, **kwargs)
    return text.strip(), dict(author_model=AUTHOR_MODEL, author_prompt_sha256=prompts.sha256(prompt),
                              author_cost_usd=usage.get("cost_usd"), author_finish_reason=usage.get("finish_reason"))


REBUTTAL_PANEL = ROOT / "data" / "manifests" / "rebuttal_panel.json"   # 2023-25 author responses (previous draft)


@functools.lru_cache
def _rebuttal_panel() -> dict:
    return {p["paper_id"]: p for p in json.loads(REBUTTAL_PANEL.read_text())["papers"]}


def real_author_response(paper_id: str) -> str:
    """The real author responses before the decision, labelled by what they reply to. ICLR 2021/2022 and NeurIPS 2022
    from the OpenReview forum files (iclr2022_forums); 2023-25 (2026-10-02) from the previous draft's frozen notes in
    data/manifests/rebuttal_panel.json, with the same headings."""
    if CORPUS != "2023-25":
        return forums.author_response(paper_id)
    paper = _rebuttal_panel().get(paper_id)
    if not paper or not (paper.get("initial_author_response") or {}).get("notes"):
        return _fetched_author_response(paper_id) if FULL else ""
    d = load_dossier(paper["venue_key"], paper_id)
    labels = d["reviewer_labels"]; labels = dict(json.loads(labels) if isinstance(labels, str) else labels)
    src = d["source_reviews"]; src = json.loads(src) if isinstance(src, str) else src
    number = {s["note_id"]: ord(labels[s["reviewer_id"]]) - ord("A") + 1 for s in src if s["reviewer_id"] in labels}
    parts = []
    for n in sorted(paper["initial_author_response"]["notes"], key=lambda n: n.get("cdate_ms") or 0):
        target = n.get("replyto")
        head = (f"### Author response to Reviewer {number[target]}" if target in number else
                "### General author response" if target == paper_id else "### Author follow-up comment")
        title = (n.get("title") or "").strip()
        parts.append(f"{head}\n\n" + (f"**{title}**\n\n" if title else "") + (n.get("raw_text") or "").strip())
    return "\n\n".join(parts)


@functools.lru_cache
def _fetched_forums() -> dict:
    out = {}
    for venue in ("iclr2025", "neurips2023", "icml2025"):
        path = ROOT / "data" / "raw" / "openreview_corpus" / venue / "forums_study.jsonl"
        if path.exists():
            for line in path.open():
                r = json.loads(line); out[r["submission"]["id"]] = (venue, r)
    return out


def _fetched_author_response(paper_id: str) -> str:
    """2023-25 papers outside the frozen panel (complete-sample extension): the authors' comments before the decision
    from the fetched API v2 forum, labelled like the other corpora."""
    hit = _fetched_forums().get(paper_id)
    if not hit:
        return ""
    venue, forum = hit
    val = lambda c, k: (c.get(k) or {}).get("value") if isinstance(c.get(k), dict) else c.get(k)
    # API v2 notes carry the original invitation plus an "Edit" one; read them all. Responses as in the frozen panel:
    # NeurIPS 2023 Rebuttal / Author_Rebuttal notes, ICML 2025 Rebuttal notes, ICLR 2025 authors' comments.
    kinds = lambda n: {str(i).split("/")[-1] for i in (n.get("invitations") or [n.get("invitation", "")])}
    accept = {"iclr2025": {"Official_Comment", "Comment"}, "neurips2023": {"Rebuttal", "Author_Rebuttal"},
              "icml2025": {"Rebuttal"}}[venue]
    replies = forum["replies"]
    decision = [n for n in replies if "Decision" in kinds(n)]
    cutoff = int(decision[0]["cdate"]) if decision else None
    d = load_dossier(venue, paper_id)
    labels = d["reviewer_labels"]; labels = dict(json.loads(labels) if isinstance(labels, str) else labels)
    src = d["source_reviews"]; src = json.loads(src) if isinstance(src, str) else src
    number = {s["note_id"]: ord(labels[s["reviewer_id"]]) - ord("A") + 1 for s in src if s["reviewer_id"] in labels}
    parts = []
    for n in sorted(replies, key=lambda n: int(n.get("cdate") or 0)):
        if not any(str(x).endswith("/Authors") for x in n.get("signatures", [])) or not (kinds(n) & accept):
            continue
        if cutoff is not None and int(n.get("cdate") or 0) >= cutoff:
            continue
        target = n.get("replyto")
        head = (f"### Author response to Reviewer {number[target]}" if target in number else
                "### General author response" if target == paper_id else "### Author follow-up comment")
        title = (val(n["content"], "title") or "").strip()
        body = (val(n["content"], "comment") or val(n["content"], "rebuttal") or "").strip()
        parts.append(f"{head}\n\n" + (f"**{title}**\n\n" if title else "") + body)
    return "\n\n".join(parts)


def build_prompt(arm: str, venue, paper: pd.Series, model, reviews: dict,
                 author_response: str | None = None) -> tuple[str, bool]:
    own = reviews[(paper.paper_id, model.key)]
    fields = dict(source_review=review_text(own), initial_score=str(own["emitted_score"]))
    if arm in ("discussion", "discussion_independent"):
        fields["peer_reviews"] = peer_block(paper.paper_id, model.key, reviews)
    if arm == "discussion_text":
        fields["peer_reviews"] = peer_block(paper.paper_id, model.key, reviews, with_scores=False)
    if arm == "discussion_masked":
        fields["peer_reviews"] = peer_block(paper.paper_id, model.key, reviews, with_scores=False, mask_verdicts=True)
    if arm == "rebuttal":
        fields["author_response"] = real_author_response(paper.paper_id)
    if arm == "llm_rebuttal":
        fields["author_response"] = author_response or ""

    def render(text):
        if arm in STAGE3:   # call_a_revision: the frozen call-A prompt split around the paper text
            return prompts.render("call_a_revision", baseline_prompt_prefix=prompts.own_review(venue, ""),
                                  paper_text=text, baseline_prompt_suffix="", **fields)
        return prompts.render(TEMPLATES[arm], own_review_prompt=prompts.own_review(venue, text), **fields)

    overhead = len(render("").encode())
    text, trimmed = fit_to_context(paper_text(paper), model, overhead)
    return render(text), trimmed


def done_cells() -> set[tuple[str, str, str]]:
    return {(r["paper_id"], r["model_id"], r["arm"]) for r in all_records() if r["parse_ok"]}


def failed_attempts() -> dict[tuple[str, str, str], int]:
    counts: dict = {}
    for r in all_records():
        if not r["parse_ok"]:
            key = (r["paper_id"], r["model_id"], r["arm"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def run(args):
    manifest = pd.DataFrame(json.loads(MANIFEST.read_text())["papers"])
    ids = sorted(manifest.paper_id)
    if CORPUS == "2023-25" and not FULL:
        ids = sorted(json.loads((OUT / "sample.json").read_text())["papers"])
    if {"discussion_text", "discussion_masked"} & set(args.arms):   # E14b, E14c: frozen sample (or all, per amendment)
        ids = sorted(json.loads((OUT / (complete_sample_file() if FULL else
                                        e14b_sample_file(getattr(args, "sample", "preregistered")))).read_text())["papers"])
    if "discussion_independent" in args.arms:   # M6 (preregistered): all E14 papers with all 8 reviewers in both arms
        ids = sorted(json.loads((OUT / (complete_sample_file() if FULL else e14b_sample_file("full"))).read_text())["papers"])
    if "rebuttal" in args.arms and CORPUS in ("iclr2021", "neurips2022") and not FULL:   # preregistered 283cff2: E14 sample
        ids = sorted(json.loads((OUT / e14b_sample_file("full")).read_text())["papers"])
    if args.pilot:
        ids = sorted(random.Random(f"{SEED}:e14:pilot").sample(ids, args.pilot))
    papers = manifest.set_index("paper_id").loc[ids].reset_index()
    reviews = round1(set(ids))
    model_of = models()
    client = OpenRouterClient(CACHE, args.max_usd)
    OUT.mkdir(parents=True, exist_ok=True)
    done, fails = done_cells(), failed_attempts()
    # deviation 2026-10-01 (as M7's deviation 2): a cell whose earlier attempt was cut off at the output limit is
    # retried with twice the limit; recorded per record as max_output_tokens
    truncated = {(r["paper_id"], r["model_id"], r["arm"]) for r in all_records()
                 if not r["parse_ok"] and (r.get("usage") or {}).get("finish_reason") == "length"}
    if "rebuttal" in args.arms and CORPUS == "2023-25":   # 2026-10-02: the 180 papers with frozen author responses
        ids = sorted(_rebuttal_panel()) if not FULL else sorted(manifest.paper_id)   # full: every paper with a response
        papers = manifest.set_index("paper_id").loc[ids].reset_index()
        reviews = round1(set(ids))
    has_response = {pid for pid in ids if real_author_response(pid)}
    jobs = [(p, m, arm) for _, p in papers.iterrows() for m in PANEL for arm in args.arms
            if (p.paper_id, m, arm) not in done and (p.paper_id, m) in reviews
            and (arm not in STAGE3 or p.paper_id in has_response)]
    missing = sum((p, m) not in reviews for p in ids for m in PANEL)
    print(f"{len(ids)} papers, {len(jobs)} round-2 cells to run ({len(done)} done; {missing} model-paper pairs "
          f"without a round-1 review)", flush=True)
    lock = threading.Lock()

    def work(job):
        paper, key, arm = job
        model = model_of[key]
        venue = venues()[paper.venue_key]
        author_text, author_meta = (author_rebuttal(client, venue, paper, reviews[(paper.paper_id, key)])
                                    if arm == "llm_rebuttal" else (None, {}))
        prompt, trimmed = build_prompt(arm, venue, paper, model, reviews, author_text)
        temperature = TEMPERATURE if model.accepts_temperature else None
        first = fails.get((paper.paper_id, key, arm), 0)
        header, _ = prompts.load_template(TEMPLATES[arm])
        cut = (paper.paper_id, key, arm) in truncated
        for attempt in range(first, first + MAX_ATTEMPTS):
            limit = 2 * MAX_OUTPUT_TOKENS if cut else MAX_OUTPUT_TOKENS
            kwargs = dict(run_idx=ROUND1_RUN, temperature=temperature, max_tokens=limit)
            if attempt:
                kwargs["attempt"] = attempt
            if model.request:
                kwargs["extra"] = model.request
            started = time.perf_counter()
            text, usage = client.complete(model.api_model, prompt, **kwargs)
            parsed = parse_round2(text, venue.scores)
            own = reviews[(paper.paper_id, key)]
            record = dict(
                paper_id=paper.paper_id, model_id=key, arm=arm, prompt_template=TEMPLATES[arm],
                prompt_version=header["version"], prompt_sha256=prompts.sha256(prompt),
                round1_call_id=own["call_id"], round1_score=own["emitted_score"],
                peers=[m for m in PANEL if m != key] if arm == "discussion" else [],
                emitted_score=parsed.score, items=parsed.items, reasoning_text=parsed.reasoning,
                raw_response=text, parse_ok=parsed.ok, parse_error=parsed.error, extra=parsed.extra,
                paper_text_context_trimmed=trimmed, temperature=temperature, attempt=attempt + 1, max_output_tokens=limit,
                author_response_text=author_text, **author_meta,
                usage={k: usage.get(k) for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cost_usd",
                                                 "cached", "finish_reason", "provider")},
                execution_model=usage.get("execution_model"), latency_s=round(time.perf_counter() - started, 3),
                timestamp=datetime.datetime.now(datetime.UTC).isoformat())
            with lock, record_file(arm).open("a") as f:
                f.write(json.dumps(record) + "\n")
            if parsed.ok:
                return record
            cut = cut or usage.get("finish_reason") == "length"
            print(f"  parse failed: {key} {paper.paper_id} {arm} attempt {attempt + 1}: {parsed.error}", flush=True)
        return record

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
                for f in futures:
                    f.cancel()
                break
            except (RuntimeError, OSError, ValueError, KeyError) as error:   # one failed cell never stops the run
                n_fail += 1
                print(f"  error: {type(error).__name__}: {str(error)[:200]}", flush=True)
            if k % 25 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} cells, {n_ok} parsed, {n_fail} failed, ${client.spent:.2f}", flush=True)
    print(f"done: {n_ok} parsed, {n_fail} failed, spent ${client.spent:.2f}", flush=True)


# ---------------------------------------------------------------- analysis (preregistered outcomes)

def auroc(score: np.ndarray, truth: np.ndarray) -> float:
    pos, neg = score[truth == 1], score[truth == 0]
    return float((pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean())


def outcomes(S: np.ndarray, human: np.ndarray, truth: np.ndarray, peer_ref: np.ndarray) -> dict:
    """S: papers x models scores. peer_ref: papers x models mean of the other models' round-1 scores."""
    E = S - human[:, None]
    C = np.corrcoef(E.T)
    iu = np.triu_indices(S.shape[1], 1)
    decided = ~np.isnan(truth)
    return dict(error_correlation=float(np.nanmean(C[iu])),
                within_paper_sd=float(S.std(axis=1, ddof=1).mean()),
                committee_abs_error=float(np.abs(S.mean(axis=1) - human).mean()),
                decision_auroc=auroc(S.mean(axis=1)[decided], truth[decided].astype(int)),
                distance_to_peers=float(np.abs(S - peer_ref).mean()))


def human_score(paper: dict) -> float:
    """The human reference on the scale of model_score: the raw mean on ICLR 2022, [0, 1] across venues."""
    return float(paper["normalized_score"]) if CORPUS == "2023-25" else float(paper["human_mean"])


def model_score(paper: dict, score: float) -> float:
    if CORPUS != "2023-25":
        return float(score)
    scale = venues()[paper["venue_key"]].scores
    return (float(score) - min(scale)) / (max(scale) - min(scale))


def analyze(args):
    manifest = {p["paper_id"]: p for p in json.loads(MANIFEST.read_text())["papers"]}
    recs = [r for r in all_records() if r["parse_ok"] and r["arm"] in ARMS]
    latest = {(r["paper_id"], r["model_id"], r["arm"]): r for r in recs}
    papers = sorted({p for p, _, _ in latest})
    if not FULL and CORPUS == "2023-25":   # the preregistered sample (complete-sample runs add papers)
        papers = [p for p in papers if p in set(json.loads((OUT / "sample.json").read_text())["papers"])]
    elif not FULL and CORPUS in ("iclr2021", "neurips2022"):
        papers = [p for p in papers if p in set(json.loads((OUT / e14b_sample_file("full")).read_text())["papers"])]
    reviews = round1(set(papers))
    # complete cases: every model has round 1 and both arms
    keep = [p for p in papers if all((p, m) in reviews and all((p, m, a) in latest for a in ARMS) for m in PANEL)]
    print(f"{len(keep)} papers with all {len(PANEL)} models in round 1 and both arms (of {len(papers)} with any)")
    if len(keep) < 5:
        print("too few complete papers for the analysis")
        return
    human = np.array([human_score(manifest[p]) for p in keep])
    truth = np.array([{"accept": 1.0, "reject": 0.0}.get(manifest[p]["outcome"], np.nan) for p in keep])
    R1 = np.array([[model_score(manifest[p], reviews[(p, m)]["emitted_score"]) for m in PANEL] for p in keep], float)
    arms = {"round1": R1}
    for a in ARMS:
        arms[a] = np.array([[model_score(manifest[p], latest[(p, m, a)]["emitted_score"]) for m in PANEL]
                            for p in keep], float)
    peer_ref = (R1.sum(axis=1, keepdims=True) - R1) / (len(PANEL) - 1)
    rng = np.random.default_rng(SEED)
    draws = [rng.integers(0, len(keep), len(keep)) for _ in range(N_BOOT)]
    rows = []
    for name, S in arms.items():
        point = outcomes(S, human, truth, peer_ref)
        boots = {k: [] for k in point}
        for idx in draws:
            if np.nansum(truth[idx] == 1) == 0 or np.nansum(truth[idx] == 0) == 0:
                continue
            for k, v in outcomes(S[idx], human[idx], truth[idx], peer_ref[idx]).items():
                boots[k].append(v)
        for k, v in point.items():
            rows.append(dict(arm=name, outcome=k, estimate=v, ci_low=np.percentile(boots[k], 2.5),
                             ci_high=np.percentile(boots[k], 97.5), n_papers=len(keep)))
    # paired differences: discussion minus second look
    for k in outcomes(R1, human, truth, peer_ref):
        diffs = []
        for idx in draws:
            if np.nansum(truth[idx] == 1) == 0 or np.nansum(truth[idx] == 0) == 0:
                continue
            d = outcomes(arms["discussion"][idx], human[idx], truth[idx], peer_ref[idx])[k] \
                - outcomes(arms["second_look"][idx], human[idx], truth[idx], peer_ref[idx])[k]
            diffs.append(d)
        est = outcomes(arms["discussion"], human, truth, peer_ref)[k] - outcomes(arms["second_look"], human, truth, peer_ref)[k]
        rows.append(dict(arm="discussion - second_look", outcome=k, estimate=est, ci_low=np.percentile(diffs, 2.5),
                         ci_high=np.percentile(diffs, 97.5), n_papers=len(keep)))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / ("outcomes_full.csv" if FULL else "outcomes.csv"), index=False)
    wide = table.pivot(index="outcome", columns="arm", values="estimate")
    print(wide[["round1", "second_look", "discussion", "discussion - second_look"]].round(3).to_string())
    diff = table[table.arm == "discussion - second_look"].set_index("outcome")
    print("\npaired 95% intervals, discussion minus second look:")
    print(diff[["estimate", "ci_low", "ci_high"]].round(3).to_string())
    changed = {a: float((arms[a] != R1).mean()) for a in ARMS}
    print(f"\nshare of scores changed from round 1: {changed}")


def reparse(args):
    """Re-read the saved responses of cells that never parsed with parse_round2 (no API calls) and
    append a record for each one that now parses, marked reparsed_from_failed_record."""
    venue = venues()[CORPUS if CORPUS in ("iclr2021", "neurips2022") else "iclr2022"]
    recs = all_records()
    ok = {(r["paper_id"], r["model_id"], r["arm"]) for r in recs if r["parse_ok"]}
    added = 0
    for r in recs:
        key = (r["paper_id"], r["model_id"], r["arm"])
        if r["parse_ok"] or key in ok:
            continue
        parsed = parse_round2(r["raw_response"], venue.scores)
        if not parsed.ok:
            continue
        new = dict(r, emitted_score=parsed.score, items=parsed.items, reasoning_text=parsed.reasoning,
                   parse_ok=True, parse_error=None, extra=parsed.extra, reparsed_from_failed_record=True,
                   timestamp=datetime.datetime.now(datetime.UTC).isoformat())
        with record_file(r["arm"]).open("a") as f:
            f.write(json.dumps(new) + "\n")
        ok.add(key)
        added += 1
    print(f"reparsed {added} previously failed cells from their saved responses")


def load_points() -> dict[str, list[dict]]:
    points: dict[str, list[dict]] = {}
    source = (ROOT / "data" / "processed" / "content" / "reference_items.jsonl" if CORPUS == "2023-25"
              else ROOT / "results" / f"{CORPUS if CORPUS in ('iclr2021', 'neurips2022') else 'iclr2022'}_prechatgpt" / "content"
              / "reference_items.jsonl")
    for line in source.open():
        item = json.loads(line)
        points.setdefault(item["paper_id"], []).append(item)
    return points


def coverage(args):
    """Label every itemized round-2 review against the paper's human-raised points with the round-1
    matcher (same model and prompt as coverage_labels.csv), for the preregistered outcome P5."""
    from peerreview import content
    annotator = models()[args.annotator]
    client = OpenRouterClient(ROOT / "cache" / "responses" / f"content_match_e14_{args.annotator}", args.max_usd)
    points = load_points()
    labels_path, log_path = OUT / "coverage_labels.csv", OUT / "match_records.jsonl"
    done = set()
    if labels_path.exists():
        prior = pd.read_csv(labels_path)
        done = set(zip(prior.paper_id, prior.model, prior.arm))
    latest = {(r["paper_id"], r["model_id"], r["arm"]): r for r in all_records()
              if r["parse_ok"] and r["items"] and r["arm"] in args.arms and r["paper_id"] in points}
    if "round1" in args.arms:   # the round-1 reviews themselves (needed on 2023-25, where no labels exist)
        ids = (json.loads((OUT / "sample.json").read_text())["papers"] if CORPUS == "2023-25" and not args.all_papers
               else list(points))
        for (pid, m), r in round1(set(ids) & set(points)).items():
            latest[(pid, m, "round1")] = r
    jobs = [(k, r) for k, r in latest.items() if k not in done]
    print(f"{len(done)} round-2 reviews labeled, {len(jobs)} to do", flush=True)
    lock = threading.Lock()

    def work(job):
        (pid, model, arm), r = job
        summary = (r.get("extra") or {}).get("summary", "")
        summary = " ".join(map(str, summary)) if isinstance(summary, list) else str(summary or "")
        return (pid, model, arm), r, content.match_review(client, annotator, points[pid], r["items"],
                                                        summary=summary, run_key=0)

    header = not labels_path.exists()
    failures = 0
    with ThreadPoolExecutor(args.workers) as pool, labels_path.open("a") as out, log_path.open("a") as log:
        if header:
            out.write("paper_id,model,arm,id,recognition,stance\n")
        for k, future in enumerate(as_completed([pool.submit(work, j) for j in jobs]), start=1):
            try:
                (pid, model, arm), _, rec = future.result()
            except (RuntimeError, OSError, ValueError, KeyError, AttributeError, TypeError) as error:
                failures += 1
                print(f"  error: {str(error)[:200]}", flush=True)
                continue
            with lock:
                log.write(json.dumps(dict(paper_id=pid, model=model, arm=arm, timestamp=datetime.datetime.now(
                    datetime.UTC).isoformat(), **rec)) + "\n")
                if rec["parse_ok"]:
                    for point_id, (recognition, stance) in rec["labels"].items():
                        out.write(f"{pid},{model},{arm},{point_id},{recognition},{stance}\n")
                else:
                    failures += 1
                out.flush()
                log.flush()
            if k % 200 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}, ${client.spent:.2f}, {failures} failures", flush=True)


def analyze_coverage(args):
    """P5: share of human-raised weaknesses mentioned by at least one of the eight reviews, per arm,
    on papers where all eight models have an itemized review in round 1 and both round-2 arms."""
    points = load_points()
    weak = {pid: {p["id"] for p in pts if p["polarity"] == "weakness"} for pid, pts in points.items()}
    r1 = pd.read_csv(ROOT / "results" / f"{CORPUS if CORPUS in ('iclr2021', 'neurips2022') else 'iclr2022'}_prechatgpt" / "content"
                     / "coverage_labels.csv")
    r1 = r1[r1.model.isin(PANEL) & (r1.run == ROUND1_RUN)].assign(arm="round1")
    r2 = pd.read_csv(OUT / "coverage_labels.csv")
    lab = pd.concat([r1[["paper_id", "model", "arm", "id", "recognition"]], r2[["paper_id", "model", "arm", "id", "recognition"]]])
    have = lab.groupby(["paper_id", "arm"]).model.nunique().unstack()
    arms = ["round1", "second_look", "discussion"]
    keep = sorted(p for p in have.index if all(have.loc[p].get(a, 0) == len(PANEL) for a in arms) and weak.get(p))
    hit = lab[lab.recognition.isin(["specific", "partial"])]
    per = {a: [] for a in arms}
    per_review = {a: [] for a in arms}
    for p in keep:
        for a in arms:
            h = hit[(hit.paper_id == p) & (hit.arm == a) & hit.id.isin(weak[p])]
            per[a].append(h.id.nunique() / len(weak[p]))
            per_review[a].append(h.groupby("model").id.nunique().reindex(PANEL, fill_value=0).mean() / len(weak[p]))
    per = {a: np.array(v) for a, v in per.items()}
    per_review = {a: np.array(v) for a, v in per_review.items()}
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(keep), len(keep)) for _ in range(N_BOOT)]
    rows = []
    for name, d in (("panel coverage (any of 8)", per), ("single-review coverage (mean)", per_review)):
        for a in arms:
            rows.append(dict(outcome=name, arm=a, estimate=d[a].mean(), n_papers=len(keep)))
        diff = d["discussion"] - d["second_look"]
        b = [diff[i].mean() for i in idx]
        rows.append(dict(outcome=name, arm="discussion - second_look", estimate=diff.mean(),
                         ci_low=np.percentile(b, 2.5), ci_high=np.percentile(b, 97.5), n_papers=len(keep)))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "outcomes_coverage.csv", index=False)
    print(f"{len(keep)} papers with all {len(PANEL)} itemized reviews in round 1 and both arms")
    print(table.round(3).to_string(index=False))


def complete_sample_file() -> str:
    """Complete-sample remedies (2026-10-04): every paper with all eight reviewers in both E14 arms, after the complete
    discussion runs; frozen once to e14b_sample_complete.json."""
    path = OUT / "e14b_sample_complete.json"
    if not path.exists():
        recs = {(r["paper_id"], r["model_id"], r["arm"]) for r in all_records() if r["parse_ok"]}
        r1 = round1({p for p, _, _ in recs})
        full = sorted({p for p, _, _ in recs if all((p, m) in r1 and (p, m, a) in recs for m in PANEL for a in ARMS)})
        path.write_text(json.dumps({"created": "2026-10-04", "design": "complete-sample remedies: all papers with all 8 "
                                    "reviewers in round 1 and both E14 arms", "papers": full}, indent=1))
    return path.name


def e14b_sample_file(sample: str) -> str:
    """The preregistered 100-paper sample, or (amendment 2026-09-30) all E14 papers with all eight reviewers
    in both E14 arms, written once to e14b_sample_full.json."""
    if sample == "preregistered":
        return "e14b_sample.json"
    path = OUT / "e14b_sample_full.json"
    if not path.exists():
        recs = {(r["paper_id"], r["model_id"], r["arm"]) for r in all_records() if r["parse_ok"]}
        full = sorted({p for p, _, _ in recs if all((p, m, a) in recs for m in PANEL for a in ARMS)})
        path.write_text(json.dumps({"created": "2026-09-30", "design": "amendment: all E14 papers with all 8 reviewers "
                                    "in both E14 arms", "papers": full}, indent=1))
    return path.name


def analyze_e14b(args):
    """E14b (preregistered, decision rule D1-D4): peers' reviews without scores vs. E14's second look and
    discussion on the same 100 papers; share of E14's convergence retained without scores."""
    global PANEL
    full_panel = PANEL
    if getattr(args, "exclude", None):   # sensitivity (2026-10-02): drop one reviewer, separate output file
        PANEL = [m for m in PANEL if m != args.exclude]
    manifest = {p["paper_id"]: p for p in json.loads(MANIFEST.read_text())["papers"]}
    ids = sorted(json.loads((OUT / (complete_sample_file() if FULL else
                                    e14b_sample_file("full" if args.sample == "m6" else args.sample))).read_text())["papers"])
    arms = ["second_look", "discussion", "discussion_text", "discussion_masked"]   # E14c added
    if os.environ.get("E14_RUN") in ("2", "3"):   # the repeat runs have no text-only arm
        arms = ["second_look", "discussion", "discussion_masked"]
    if args.sample == "m6":
        arms = ["second_look", "discussion", "discussion_independent"]
    latest = {(r["paper_id"], r["model_id"], r["arm"]): r for r in all_records() if r["parse_ok"] and r["arm"] in arms}
    reviews = round1(set(ids))
    keep = [p for p in ids if all((p, m) in reviews and all((p, m, a) in latest for a in arms) for m in PANEL)]
    h = np.array([human_score(manifest[p]) for p in keep])
    truth = np.array([{"accept": 1.0, "reject": 0.0}.get(manifest[p]["outcome"], np.nan) for p in keep])
    # model_score: raw on the pre-ChatGPT corpora (unchanged), [0, 1] per venue on 2023-25 (ICML 2025 is 1-5)
    R1 = np.array([[model_score(manifest[p], reviews[(p, m)]["emitted_score"]) for m in PANEL] for p in keep], float)
    S = {"round1": R1, **{a: np.array([[model_score(manifest[p], latest[(p, m, a)]["emitted_score"]) for m in PANEL]
                                       for p in keep], float) for a in arms}}
    peer_ref = (R1.sum(axis=1, keepdims=True) - R1) / (len(PANEL) - 1)
    iu = np.triu_indices(len(PANEL), 1)

    def stats(M, i=slice(None)):
        X, hh = M[i], h[i]
        o = outcomes(X, hh, truth[i], peer_ref[i])
        R = np.column_stack([X[:, j] - np.polyval(np.polyfit(hh, X[:, j], 1), hh) for j in range(X.shape[1])])
        o["error_correlation_calibration_free"] = float(np.nanmean(np.corrcoef(R.T)[iu]))
        return o

    rng = np.random.default_rng(SEED)
    draws = [d for d in (rng.integers(0, len(keep), len(keep)) for _ in range(N_BOOT))
             if np.nansum(truth[d] == 1) and np.nansum(truth[d] == 0)]
    point = {a: stats(M) for a, M in S.items()}
    boots = [{a: stats(M, d) for a, M in S.items() if a != "round1"} for d in draws]
    rows = []
    for k in point["second_look"]:
        row = dict(outcome=k, **{a: point[a][k] for a in S})
        pairs = ((("discussion_independent", "second_look"), ("discussion", "second_look"),
                  ("discussion_independent", "discussion")) if args.sample == "m6" else
                 tuple((x, y) for x, y in (("discussion_text", "second_look"), ("discussion_masked", "second_look"),
                                          ("discussion", "second_look"), ("discussion_masked", "discussion_text"))
                       if x in arms and y in arms))
        for a, b in pairs:
            diffs = [x[a][k] - x[b][k] for x in boots]
            row[f"{a} - {b}"] = f"{point[a][k] - point[b][k]:+.3f} [{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}]"
        full = point["discussion"][k] - point["second_look"][k]
        for a, tag in ((("discussion_independent", "independent"),) if args.sample == "m6" else
                       tuple((x, y) for x, y in (("discussion_text", "text"), ("discussion_masked", "masked"))
                             if x in arms)):
            row[f"share_{tag}"] = (point[a][k] - point["second_look"][k]) / full if abs(full) > 1e-9 else np.nan
            shares = [(x[a][k] - x["second_look"][k]) / (x["discussion"][k] - x["second_look"][k])
                      for x in boots if abs(x["discussion"][k] - x["second_look"][k]) > 1e-9]
            row[f"share_{tag}_ci"] = f"[{np.percentile(shares, 2.5):.2f}, {np.percentile(shares, 97.5):.2f}]"
        rows.append(row)
    table = pd.DataFrame(rows)
    name = {"preregistered": "outcomes_e14b.csv", "full": "outcomes_e14b_full.csv", "m6": "outcomes_m6.csv"}[args.sample]
    if FULL:   # complete-sample remedies: separate outputs
        name = name.replace("_full.csv", ".csv").replace(".csv", "_complete.csv")
    if getattr(args, "exclude", None):
        name = name.replace(".csv", f"_without_{args.exclude}.csv")
        PANEL = full_panel
        table.to_csv(OUT / name, index=False)
        print(f"{len(keep)} papers with all {len(PANEL) - 1} remaining reviewers; wrote {name}")
        print(table.round(3).to_string(index=False))
        return
    table.to_csv(OUT / name, index=False)
    print(f"{len(keep)} papers with all {len(PANEL)} reviewers in round 1 and all four arms")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))
    changed = {a: float((S[a] != R1).mean()) for a in arms}
    print("share of scores changed from round 1:", {a: round(v, 3) for a, v in changed.items()})
    # coverage (panel and single-review) on the same papers, where every review lists its points
    cov = pd.read_csv(OUT / "coverage_labels.csv")
    weak = {pid: {p["id"] for p in pts if p["polarity"] == "weakness"} for pid, pts in load_points().items()}
    hit = cov[cov.recognition.isin(["specific", "partial"])]
    have = cov.groupby(["paper_id", "arm"]).model.nunique().unstack()
    both = [p for p in keep if all(have.get(a, pd.Series(dtype=float)).get(p, 0) == len(PANEL) for a in arms)]
    for a in arms:
        panel = [hit[(hit.paper_id == p) & (hit.arm == a) & hit.id.isin(weak[p])].id.nunique() / len(weak[p]) for p in both]
        single = [hit[(hit.paper_id == p) & (hit.arm == a) & hit.id.isin(weak[p])].groupby("model").id.nunique()
                  .reindex(PANEL, fill_value=0).mean() / len(weak[p]) for p in both]
        print(f"coverage {a:16s} panel {100 * np.mean(panel):.1f}%  single-review {100 * np.mean(single):.1f}%  ({len(both)} papers)")


def analyze_rebuttal(args):
    """Stage 3: revise after the real author responses vs. the second-look control, on the papers
    with an author response. Inflation = mean score change from round 1."""
    manifest = {p["paper_id"]: p for p in json.loads(MANIFEST.read_text())["papers"]}
    recs = [r for r in all_records() if r["parse_ok"]]
    latest = {(r["paper_id"], r["model_id"], r["arm"]): r for r in recs}
    arm, ref = args.arm, args.against
    papers = sorted({p for p, _, a in latest if a == arm})
    if not FULL and CORPUS in ("iclr2021", "neurips2022"):   # the preregistered sets (complete-sample runs add papers)
        papers = [p for p in papers if p in set(json.loads((OUT / e14b_sample_file("full")).read_text())["papers"])]
    elif not FULL and CORPUS == "2023-25":
        papers = [p for p in papers if p in _rebuttal_panel()]
    reviews = round1(set(papers))
    keep = [p for p in papers if all((p, m) in reviews and (p, m, arm) in latest
                                     and (p, m, ref) in latest for m in PANEL)]
    print(f"{len(keep)} papers with an author response and all {len(PANEL)} models in both arms")
    if len(keep) < 5:
        return
    human = np.array([human_score(manifest[p]) for p in keep])   # raw mean, or [0, 1] per venue on 2023-25
    truth = np.array([{"accept": 1.0, "reject": 0.0}.get(manifest[p]["outcome"], np.nan) for p in keep])
    R1 = np.array([[model_score(manifest[p], reviews[(p, m)]["emitted_score"]) for m in PANEL] for p in keep], float)
    arms = {"round1": R1, **{a: np.array([[model_score(manifest[p], latest[(p, m, a)]["emitted_score"]) for m in PANEL]
                                          for p in keep], float) for a in dict.fromkeys(("second_look", ref, arm))}}
    peer_ref = (R1.sum(axis=1, keepdims=True) - R1) / (len(PANEL) - 1)
    rng = np.random.default_rng(SEED)
    draws = [rng.integers(0, len(keep), len(keep)) for _ in range(N_BOOT)]
    ok = [idx for idx in draws if np.nansum(truth[idx] == 1) and np.nansum(truth[idx] == 0)]

    def stats(S, idx=slice(None)):
        o = outcomes(S[idx], human[idx], truth[idx], peer_ref[idx])
        o["mean_score_change"] = float((S[idx] - R1[idx]).mean())
        return o

    rows = []
    for name, S in arms.items():
        for k, v in stats(S).items():
            b = [stats(S, idx)[k] for idx in ok]
            rows.append(dict(arm=name, outcome=k, estimate=v, ci_low=np.percentile(b, 2.5),
                             ci_high=np.percentile(b, 97.5), n_papers=len(keep)))
    for k in stats(R1):
        d = [stats(arms[arm], idx)[k] - stats(arms[ref], idx)[k] for idx in ok]
        rows.append(dict(arm=f"{arm} - {ref}", outcome=k,
                         estimate=stats(arms[arm])[k] - stats(arms[ref])[k],
                         ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5), n_papers=len(keep)))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / (f"outcomes_{arm}_vs_{ref}_full.csv" if FULL else f"outcomes_{arm}_vs_{ref}.csv"), index=False)
    print(table.pivot(index="outcome", columns="arm", values="estimate")
          [list(arms) + [f"{arm} - {ref}"]].round(3).to_string())
    d = table[table.arm == f"{arm} - {ref}"].set_index("outcome")[["estimate", "ci_low", "ci_high"]]
    print(f"\npaired 95% intervals, {arm} minus {ref}:\n" + d.round(3).to_string())
    by = pd.DataFrame(dict(outcome=[manifest[p]["outcome"] for p in keep],
                           change=(arms[arm] - R1).mean(axis=1), control=(arms[ref] - R1).mean(axis=1)))
    print(f"\nmean score change by outcome ({arm} / {ref}):")
    print(by.groupby("outcome")[["change", "control"]].mean().round(2).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--pilot", type=int, default=0, help="run only this many seeded pilot papers")
    r.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(TEMPLATES))
    r.add_argument("--sample", default="preregistered", choices=["preregistered", "full"], help="E14b/E14c sample")
    r.add_argument("--workers", type=int, default=24)
    r.add_argument("--max-usd", type=float, required=True)
    r.set_defaults(run=run)
    a = sub.add_parser("analyze")
    a.set_defaults(run=analyze)
    b = sub.add_parser("analyze-rebuttal")
    b.add_argument("--arm", default="rebuttal", choices=list(STAGE3))
    b.add_argument("--against", default="second_look", choices=["second_look", *STAGE3])
    b.set_defaults(run=analyze_rebuttal)
    e = sub.add_parser("analyze-e14b")
    e.add_argument("--sample", default="preregistered", choices=["preregistered", "full", "m6"])
    e.add_argument("--exclude", default=None, choices=PANEL, help="sensitivity: drop this reviewer (separate output)")
    e.set_defaults(run=analyze_e14b)
    c = sub.add_parser("reparse")
    c.set_defaults(run=reparse)
    v = sub.add_parser("coverage")
    v.add_argument("--annotator", default="gemini-3.5-flash-lite")
    v.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(TEMPLATES) + ["round1"])
    v.add_argument("--workers", type=int, default=24)
    v.add_argument("--max-usd", type=float, required=True)
    v.add_argument("--all-papers", action="store_true", help="round1 on every paper with extracted weaknesses")
    v.set_defaults(run=coverage)
    w = sub.add_parser("analyze-coverage")
    w.set_defaults(run=analyze_coverage)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
