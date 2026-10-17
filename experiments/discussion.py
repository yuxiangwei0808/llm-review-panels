"""Revisions of the first reviews: plain revision, discussion and its restrictions, and revision after author responses.

Each reviewer of the eight-family panel starts from its first review (call A, run 0) and writes a replacement review,
one sample per model, paper, and condition (field ``arm``):

  second_look             the paper and its own first review again (plain revision, the control)
  discussion              plus the other seven first reviews with their scores, anonymized, in a seeded order
  discussion_text         plus the other reviews without their scores (hidden scores)
  discussion_masked       hidden scores, and sentences stating a verdict removed from the other reviews (hidden verdicts)
  discussion_independent  discussion plus an instruction to check the other reviews against the paper, not follow them
  rebuttal                plus the authors' real responses before the decision
  llm_rebuttal            plus a reply to its own review, written by GPT-5.1 acting as the authors (ICLR 2022)

second_look and discussion run on every paper with first reviews; the restricted discussions afterwards, on the papers
where all eight reviewers have a first review and a revision in both; the rebuttal conditions on papers with an author
response. The venue is chosen with PANEL_CORPUS: iclr2021, iclr2022 (the default), neurips2022, or 2023-25. Author
responses come from the OpenReview forum files data/raw/openreview_corpus/<venue>/forums.jsonl (API v1 for 2021-22, v2
for 2023-25; not distributed). The coverage step labels revisions against the points in
results/<venue>_prechatgpt/content/ for 2021-22 and, for 2023-25, data/processed/content/ (the 332 papers whose
weaknesses were extracted first).

Usage:
  PANEL_CORPUS=iclr2021 uv run python experiments/discussion.py run --arms second_look discussion --max-usd 300
  uv run python experiments/discussion.py run --arms rebuttal llm_rebuttal --max-usd 200
  uv run python experiments/discussion.py run --repeat 2 --max-usd 300     # ICLR 2022: a second independent run
  uv run python experiments/discussion.py reparse                          # re-read failed responses, no model calls
  uv run python experiments/discussion.py coverage --max-usd 30            # coverage judge on the revisions
Writes: results/<folder>/discussion/records.jsonl (rebuttal conditions: records_rebuttal.jsonl), coverage_labels.csv,
where <folder> is iclr2021_prechatgpt, iclr2022_prechatgpt, neurips2022_prechatgpt, or study2325 (repeat runs:
discussion_run2/, discussion_run3/).
Responses are cached under cache/responses/discussion, so a rerun resumes for free.
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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peerreview import parsing, prompts, records
from peerreview.client import CostCeilingReached, OpenRouterClient
from peerreview.config import MANIFESTS, MAX_OUTPUT_TOKENS, PANEL, ROOT, SEED, TEMPERATURE, models, venues
from peerreview.papers import load_dossier, paper_text
from peerreview.runner import MAX_ATTEMPTS, fit_to_context

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forums
from verdict_mask import masked, verdict_masked

CORPUS = os.environ.get("PANEL_CORPUS", "iclr2022")
if CORPUS == "2023-25":
    MANIFEST = ROOT / "data" / "manifests" / "study_manifest.json"
    OUT = ROOT / "results" / "study2325" / "discussion"
    CONTENT = ROOT / "data" / "processed" / "content"     # the 332 papers whose weaknesses were extracted first
    FIRST_REVIEW_SOURCES = [("iclr2025", "panel8f"), ("neurips2023", "panel8f"), ("icml2025", "panel8f")]
elif CORPUS in ("iclr2021", "iclr2022", "neurips2022"):
    MANIFEST = ROOT / "data" / "manifests" / MANIFESTS[CORPUS]
    OUT = ROOT / "results" / f"{CORPUS}_prechatgpt" / "discussion"
    CONTENT = ROOT / "results" / f"{CORPUS}_prechatgpt" / "content"
    FIRST_REVIEW_SOURCES = [(CORPUS, "prechatgpt")]
else:
    raise SystemExit("PANEL_CORPUS must be iclr2021, iclr2022, neurips2022, or 2023-25")
CACHE = ROOT / "cache" / "responses" / "discussion"
ARMS = {"second_look": "call_a_second_look", "discussion": "call_a_discussion"}
RESTRICTED = {"discussion_text": "call_a_discussion_text", "discussion_masked": "call_a_discussion_text",
              "discussion_independent": "call_a_discussion_independent"}
REBUTTAL = {"rebuttal": "call_a_revision", "llm_rebuttal": "call_a_revision"}
TEMPLATES = {**ARMS, **RESTRICTED, **REBUTTAL}
PEER_ARMS = {"discussion", *RESTRICTED}   # the conditions that show the other reviews
AUTHOR_MODEL = "gpt-5.1"     # writes the llm_rebuttal replies
PEER_ORDER_SEED = f"{SEED}:peer-order"   # seeds the order in which a reviewer sees the other reviews
FIRST_REVIEW_RUN = 0


def set_repeat(k: int) -> None:
    """Point the module at repeat run k: first reviews of run k-1, outputs in discussion_run<k>/."""
    global FIRST_REVIEW_RUN, OUT
    if k > 1:
        assert CORPUS == "iclr2022", "repeat runs are on ICLR 2022"
        FIRST_REVIEW_RUN, OUT = k - 1, OUT.parent / f"discussion_run{k}"


def record_file(arm: str) -> Path:
    """The rebuttal conditions write their own file, so the two kinds of run can append in parallel."""
    return OUT / ("records_rebuttal.jsonl" if arm in REBUTTAL else "records.jsonl")


def all_records() -> list[dict]:
    out = []
    for name in ("records.jsonl", "records_rebuttal.jsonl"):
        if (OUT / name).exists():
            out += [json.loads(line) for line in (OUT / name).open()]
    return out


def first_reviews(papers: set[str]) -> dict[tuple[str, str], dict]:
    """(paper, model) -> the parsed first review (call A) of run FIRST_REVIEW_RUN, the last parsed attempt."""
    out = {}
    for m in PANEL:
        for venue_key, tag in FIRST_REVIEW_SOURCES:
            try:
                for r in records.iter_records(m, venue_key, "paper_review", "A", tag=tag):
                    if r["paper_id"] in papers and r["run_idx"] == FIRST_REVIEW_RUN and r["parse_ok"] and r["items"]:
                        out[(r["paper_id"], m)] = r
            except FileNotFoundError:
                continue
    return out


def complete_papers() -> list[str]:
    """Papers where all eight reviewers have a first review and a parsed revision in second_look and discussion."""
    done = {(r["paper_id"], r["model_id"], r["arm"]) for r in all_records() if r["parse_ok"]}
    r1 = first_reviews({p for p, _, _ in done})
    return sorted({p for p, _, _ in done if all((p, m) in r1 and (p, m, a) in done for m in PANEL for a in ARMS)})


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
    """Map the schema variants seen in revisions onto the call-A schema. Returns (review, has_points).
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


def peer_order(paper_id: str, model: str, reviews: dict) -> list[str]:
    """The other reviewers, in the seeded order a reviewer sees them (Reviewer 1..7)."""
    peers = [m for m in PANEL if m != model and (paper_id, m) in reviews]
    random.Random(f"{PEER_ORDER_SEED}:{paper_id}:{model}").shuffle(peers)
    return peers


def peer_block(paper_id: str, model: str, reviews: dict, with_scores: bool = True, mask_verdicts: bool = False) -> str:
    """The other reviewers' first reviews, as Reviewer 1..7 in a seeded order per paper and reviewer."""
    peers = peer_order(paper_id, model, reviews)
    head = lambda k, m: (f"### Reviewer {k} (score: {reviews[(paper_id, m)]['emitted_score']})" if with_scores
                         else f"### Reviewer {k}")
    body = lambda m: (verdict_masked(masked(review_text(reviews[(paper_id, m)]))) if mask_verdicts
                      else review_text(reviews[(paper_id, m)]))
    return "\n\n".join(f"{head(k, m)}\n\n{body(m)}" for k, m in enumerate(peers, start=1))


def author_rebuttal(client, venue, paper: pd.Series, own: dict) -> tuple[str, dict]:
    """Reply to this reviewer's own review: an LLM acting as the authors writes it, using only the paper and the
    authors' real responses as sources (prompt author_reply)."""
    author = models()[AUTHOR_MODEL]
    prompt = prompts.render("author_reply", paper_text=paper_text(paper), source_review=review_text(own),
                            initial_score=str(own["emitted_score"]), real_response=real_author_response(paper.paper_id))
    kwargs = dict(run_idx=0, temperature=TEMPERATURE if author.accepts_temperature else None, max_tokens=4096)
    if author.request:
        kwargs["extra"] = author.request
    text, usage = client.complete(author.api_model, prompt, **kwargs)
    return text.strip(), dict(author_model=AUTHOR_MODEL, author_prompt_sha256=prompts.sha256(prompt),
                              author_cost_usd=usage.get("cost_usd"), author_finish_reason=usage.get("finish_reason"))


FIRST_ROUND_NOTES = ROOT / "data" / "manifests" / "author_response_notes.json"


@functools.lru_cache
def _forums_v2() -> dict:
    out, paths = {}, [ROOT / "data" / "raw" / "openreview_corpus" / v / "forums.jsonl"
                      for v in ("iclr2025", "neurips2023", "icml2025")]
    if not any(p.exists() for p in paths):
        raise SystemExit(f"no OpenReview forum files found: {paths}")
    for path in paths:
        if path.exists():
            for line in path.open():
                r = json.loads(line)
                out[r["submission"]["id"]] = (path.parent.name, r)
    return out


@functools.lru_cache
def _first_round_notes() -> dict[str, list[str]]:
    return json.loads(FIRST_ROUND_NOTES.read_text())["papers"] if FIRST_ROUND_NOTES.exists() else {}


def _author_response_v2(paper_id: str) -> str:
    """The authors' responses in an OpenReview API v2 forum (2023-25 venues), labelled like the API v1 responses of
    experiments/forums.py: the notes listed in author_response_notes.json where the paper has an entry (the authors'
    first round of responses), otherwise every author note before the decision."""
    hit = _forums_v2().get(paper_id)
    if not hit:
        return ""
    venue, forum = hit
    val = lambda c, k: (c.get(k) or {}).get("value") if isinstance(c.get(k), dict) else c.get(k)
    # API v2 notes carry the original invitation plus an "Edit" one; read them all. Author responses are NeurIPS 2023
    # Rebuttal / Author_Rebuttal notes, ICML 2025 Rebuttal notes, and the authors' ICLR 2025 comments.
    kinds = lambda n: {str(i).split("/")[-1] for i in (n.get("invitations") or [n.get("invitation", "")])}
    accept = {"iclr2025": {"Official_Comment", "Comment"}, "neurips2023": {"Rebuttal", "Author_Rebuttal"},
              "icml2025": {"Rebuttal"}}[venue]
    replies = forum["replies"]
    decision = [n for n in replies if "Decision" in kinds(n)]
    cutoff = int(decision[0]["cdate"]) if decision else None
    listed = set(_first_round_notes().get(paper_id, []))
    d = load_dossier(venue, paper_id)
    labels = d["reviewer_labels"]; labels = dict(json.loads(labels) if isinstance(labels, str) else labels)
    src = d["source_reviews"]; src = json.loads(src) if isinstance(src, str) else src
    number = {s["note_id"]: ord(labels[s["reviewer_id"]]) - ord("A") + 1 for s in src if s["reviewer_id"] in labels}
    parts = []
    for n in sorted(replies, key=lambda n: int(n.get("cdate") or 0)):
        if listed:
            if n["id"] not in listed:
                continue
        elif not any(str(x).endswith("/Authors") for x in n.get("signatures", [])) or not (kinds(n) & accept) \
                or (cutoff is not None and int(n.get("cdate") or 0) >= cutoff):
            continue
        target = n.get("replyto")
        head = (f"### Author response to Reviewer {number[target]}" if target in number else
                "### General author response" if target == paper_id else "### Author follow-up comment")
        title = (val(n["content"], "title") or "").strip()
        body = (val(n["content"], "comment") or val(n["content"], "rebuttal") or "").strip()
        parts.append(f"{head}\n\n" + (f"**{title}**\n\n" if title else "") + body)
    return "\n\n".join(parts)


def real_author_response(paper_id: str) -> str:
    """The authors' responses before the decision, labelled by the review they reply to; empty if none."""
    return _author_response_v2(paper_id) if CORPUS == "2023-25" else forums.author_response(paper_id)


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
        if arm in REBUTTAL:   # call_a_revision: the first-review prompt, split around the paper text
            return prompts.render("call_a_revision", baseline_prompt_prefix=prompts.own_review(venue, ""),
                                  paper_text=text, baseline_prompt_suffix="", **fields)
        return prompts.render(TEMPLATES[arm], own_review_prompt=prompts.own_review(venue, text), **fields)

    overhead = len(render("").encode())
    text, trimmed = fit_to_context(paper_text(paper), model, overhead)
    return render(text), trimmed


def papers_for(arm: str, manifest: pd.DataFrame) -> list[str]:
    if arm in RESTRICTED:
        return complete_papers()
    if arm in REBUTTAL:
        return sorted(p for p in manifest.paper_id if real_author_response(p))
    return sorted(manifest.paper_id)


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
    set_repeat(args.repeat)
    manifest = pd.DataFrame(json.loads(MANIFEST.read_text())["papers"])
    by_arm = {arm: papers_for(arm, manifest) for arm in args.arms}
    ids = sorted(set().union(*by_arm.values()))
    rows = {p.paper_id: p for _, p in manifest.set_index("paper_id").loc[ids].reset_index().iterrows()}
    reviews = first_reviews(set(ids))
    model_of = models()
    client = OpenRouterClient(CACHE, args.max_usd)
    OUT.mkdir(parents=True, exist_ok=True)
    done, fails = done_cells(), failed_attempts()
    # a cell whose earlier attempt was cut off at the output limit is retried with twice the limit
    truncated = {(r["paper_id"], r["model_id"], r["arm"]) for r in all_records()
                 if not r["parse_ok"] and (r.get("usage") or {}).get("finish_reason") == "length"}
    jobs = [(rows[p], m, arm) for arm, arm_ids in by_arm.items() for p in arm_ids for m in PANEL
            if (p, m, arm) not in done and (p, m) in reviews]
    print(f"{len(ids)} papers, {len(jobs)} cells to run ({len(done)} done)", flush=True)
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
            kwargs = dict(run_idx=FIRST_REVIEW_RUN, temperature=temperature, max_tokens=limit)
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
                peers=peer_order(paper.paper_id, key, reviews) if arm in PEER_ARMS else [],
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


def reparse(args):
    """Re-read the saved responses of cells that never parsed (no model calls) and append a record for each one that
    now parses, marked reparsed_from_failed_record."""
    set_repeat(args.repeat)
    manifest = pd.DataFrame(json.loads(MANIFEST.read_text())["papers"])
    venue_of = dict(zip(manifest.paper_id, manifest.venue_key))
    recs = all_records()
    ok = {(r["paper_id"], r["model_id"], r["arm"]) for r in recs if r["parse_ok"]}
    added = 0
    for r in recs:
        key = (r["paper_id"], r["model_id"], r["arm"])
        if r["parse_ok"] or key in ok:
            continue
        parsed = parse_round2(r["raw_response"], venues()[venue_of[r["paper_id"]]].scores)
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
    """The paper's human-raised points (reference_items.jsonl of the corpus's content folder)."""
    points: dict[str, list[dict]] = {}
    for line in (CONTENT / "reference_items.jsonl").open():
        item = json.loads(line)
        points.setdefault(item["paper_id"], []).append(item)
    return points


def coverage(args):
    """Label every revised review with points against the paper's human-raised points (CONTENT), with the coverage
    judge's model and prompt; with --arms round1, the first reviews too. Writes coverage_labels.csv and
    match_records.jsonl. A review whose labels did not parse is sampled afresh on a rerun."""
    from peerreview import content
    set_repeat(args.repeat)
    annotator = models()[args.annotator]
    client = OpenRouterClient(CACHE.parent / f"content_match_discussion_{args.annotator}", args.max_usd)
    points = load_points()
    labels_path, log_path = OUT / "coverage_labels.csv", OUT / "match_records.jsonl"
    done = set()
    if labels_path.exists():
        prior = pd.read_csv(labels_path)
        done = set(zip(prior.paper_id, prior.model, prior.arm))
    failed: dict[tuple, int] = {}
    if log_path.exists():
        for line in log_path.open():
            r = json.loads(line)
            if not r["parse_ok"]:
                key = (r["paper_id"], r["model"], r["arm"])
                failed[key] = max(failed.get(key, 0), r.get("attempt", 1))
    latest = {(r["paper_id"], r["model_id"], r["arm"]): r for r in all_records()
              if r["parse_ok"] and r.get("items") and r["arm"] in args.arms and r["paper_id"] in points}
    if "round1" in args.arms:   # the first reviews themselves, labelled with the revisions
        for (pid, m), r in first_reviews(set(points)).items():
            latest[(pid, m, "round1")] = r
    jobs = [(k, r) for k, r in latest.items() if k not in done]
    print(f"{len(done)} reviews labelled, {len(jobs)} to do", flush=True)
    lock = threading.Lock()

    def work(job):
        (pid, model, arm), r = job
        summary = (r.get("extra") or {}).get("summary", "")
        summary = " ".join(map(str, summary)) if isinstance(summary, list) else str(summary or "")
        return (pid, model, arm), r, content.match_review(client, annotator, points[pid], r["items"], summary=summary,
                                                        run_key=0, attempt_offset=failed.get((pid, model, arm), 0))

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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(TEMPLATES))
    r.add_argument("--repeat", type=int, default=1, choices=[1, 2, 3], help="ICLR 2022: independent repeat run")
    r.add_argument("--workers", type=int, default=24)
    r.add_argument("--max-usd", type=float, required=True)
    r.set_defaults(run=run)
    c = sub.add_parser("reparse")
    c.add_argument("--repeat", type=int, default=1, choices=[1, 2, 3])
    c.set_defaults(run=reparse)
    v = sub.add_parser("coverage")
    v.add_argument("--annotator", default="gemini-3.5-flash-lite")
    v.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(TEMPLATES) + ["round1"])
    v.add_argument("--repeat", type=int, default=1, choices=[1, 2, 3])
    v.add_argument("--workers", type=int, default=24)
    v.add_argument("--max-usd", type=float, required=True)
    v.set_defaults(run=coverage)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
