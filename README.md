# LLM Review Panels

Panels of LLM reviewers taken through the stages of peer review: independent review, rebuttal, discussion, and the
area chair's decision. The code runs the panels through OpenRouter, judges which weaknesses raised by human reviewers
each LLM review also raises, and analyzes the saved records. It accompanies the paper *Second Opinions or Echoes? Shared
Errors in LLM Review Panels*.

## Installation

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Model calls go through OpenRouter: set `OPENROUTER_API_KEY`. The analyses need only the data.

## Data

The review records, judge labels, and human annotations are on Hugging Face
([stanjsx/llm-review-panels](https://huggingface.co/datasets/stanjsx/llm-review-panels)); its `data/` and `results/`
folders go in the repository root, and its `DATA_README.md` describes every file:

```bash
uvx --from huggingface_hub hf download stanjsx/llm-review-panels --repo-type dataset \
    --include "data/*" --include "results/*" --local-dir .
```

Results are organized by venue group (`iclr2021`, `iclr2022`, `neurips2022`, and `2023-25` for ICLR 2025, NeurIPS
2023, and ICML 2025), with the first reviews under `results/first_reviews/<model>/<venue>/`.

### Inputs for running models

Running models also needs inputs that are not distributed; the papers and reviews are public on OpenReview under the
ids in `data/manifests/`.

| Input | Path | Contents |
|---|---|---|
| Paper texts | `data/raw/paper_text/<venue>/<paper_id>.md` (the manifest's `paper_text_path`) | The text the models read: the submitted PDF converted with Nemotron OCR v2 (the 2023-25 venues and 144 ICLR 2022 papers) or PyMuPDF (the others), as the manifest's `paper_text_source` records |
| Human-review dossiers | `data/dossiers/<venue>/<paper_id>.json` | `holistic_text` (`## Abstract`, then one `## Reviewer A`, `## Reviewer B`, ... section per review, ratings removed), `reviewer_labels` (pairs of OpenReview reviewer id and letter), `source_reviews` (note id, reviewer id, rating) |
| OpenReview forums | `data/raw/openreview_corpus/<venue>/forums.jsonl` | One forum per line, `{"submission": {...}, "replies": [...]}`: API v1 for the 2021-22 venues (author responses, and the human reviews the ICLR 2022 area chairs read), API v2 for the 2023-25 venues (author responses) |

For 180 of the 2023-25 papers, the author responses are the authors' first round of responses, the notes listed in
`data/manifests/author_response_notes.json`; for the other papers, all author notes before the decision.

## Repository structure

```
peerreview/    library: model client, configuration, prompts, parsing, records, review runner, coverage judge, area chair
experiments/   scripts that run the panels and the judge (model calls)
analysis/      scripts that compute results from the saved records (no model calls)
configs/       models and venues
prompts/       prompt templates
```

### `peerreview`

| Module | Contents |
|---|---|
| `client.py` | Model calls through OpenRouter, with an on-disk response cache and a cost ceiling |
| `config.py` | Venue and model settings from `configs/*.yaml`, the two panels, and the venue groups |
| `prompts.py` | Loads the templates in `prompts/` and fills their fields |
| `parsing.py` | Parses model responses into scores and review items |
| `records.py` | Reads and writes per-call records, one JSON object per line |
| `papers.py` | Paper lists, the paper text a model reads, and the human-review dossiers |
| `runner.py` | First reviews for a set of papers and models |
| `content.py` | Weakness extraction from human reviews and the coverage judge |
| `area_chair.py` | The area-chair prompt and the parser of its decision |

## Running the panels

The scripts in `experiments/` call models; each script's docstring gives its options, and every run takes a spending
ceiling (`--max-usd`). Responses are cached under `cache/`, so a rerun resumes where it stopped and retries failed
cells with fresh samples. The scripts after first reviews select the venue group with the environment variable `VENUE`
(`iclr2021`, `iclr2022`, `neurips2022`, or `2023-25`; `iclr2022` when unset). The steps depend on each other in this
order:

1. `reviews.py`: first reviews.
2. `judge.py extract`, then `judge.py match`: the weaknesses in the human reviews and the labels of the first reviews;
   then `human_coverage.py`.
3. `discussion.py run` with `plain_revision` and `discussion`, then the restricted discussions (they run on the papers
   complete in both), and the rebuttal conditions; then `discussion.py coverage`.
4. `notes.py run`, then `notes.py adoption`; `roles.py run`, then `roles.py coverage`; `area_chairs.py run`.

```bash
uv run python experiments/reviews.py iclr2025 --models kimi-k3 grok-4.7 --max-usd 50
VENUE=iclr2021 uv run python experiments/judge.py match --annotator gemini-3.5-flash-lite
VENUE=iclr2021 uv run python experiments/discussion.py run --arms plain_revision discussion --max-usd 300
```

| Script | Runs | Prompts |
|---|---|---|
| `reviews.py` | First reviews; with `--calls score_from_human_reviews`, scores from the human reviews instead | `review`, `score_from_human_reviews` |
| `judge.py` | `extract`: the weaknesses in each paper's human reviews; `match`: the coverage judge's labels of the first reviews | `extract_points`, `coverage_judge` |
| `human_coverage.py` | The coverage judge applied to human committees | `coverage_judge` |
| `discussion.py` | `run`: plain revision; discussion with the other reviews, with scores hidden, with verdicts hidden, or with an instruction to judge independently; revision after author responses or after a reply to the reviewer's own review. `coverage`: the coverage judge on the revisions | `plain_revision`, `discussion*`, `revision_after_response`, `author_reply` |
| `notes.py` | `run`: revision or fresh review after notes with weaknesses the panel missed, and control notes. `adoption`: which notes the reviews raise | `notes_revision`, `notes_fresh_review`, `coverage_judge` |
| `roles.py` | `run`: first reviews with an assigned focus. `coverage`: the coverage judge on them | `role_review`, `coverage_judge` |
| `area_chairs.py` | Area-chair decisions from the human reviews or the panel's first reviews | `area_chair` |
| `forums.py`, `verdict_mask.py` | Helpers: OpenReview forum threads; masking of scores and verdict sentences | |

## Analyses

The scripts in `analysis/` read the saved records and make no model calls. Each prints its results and writes a JSON
file to `outputs/`. `common.py` holds the data loading and the measures.

```bash
uv run python analysis/scaling.py
```

| Script | Computes | Runtime |
|---|---|---|
| `scaling.py` | Correlation with a held-out human rating as LLM panels grow from one to eight models and human committees from one to three reviewers | 4 min |
| `shared_errors.py` | Miss rate of pooled reviews, missed weaknesses by kind, and the error correlation of model pairs and human pairs | 6 min |
| `review_cycle.py` | Score changes after rebuttal, discussion against a plain revision, and unanimous area-chair errors | 4 min |
| `remedies.py` | Each remedy against its matched comparison: panel error, decision AUROC, score correlation, error correlation, miss rate | 12 min |
| `score_shifts.py` | How each remedy moves the panel's scores: share of scores lowered, and the change in score and error correlation | 6 min |

In the paper, these give Figure 1 and Table 1 (`scaling.py`), Section 3.1 and Figure 4 (`shared_errors.py`), Sections
3.2–3.3 and Table 2 (`review_cycle.py`), Section 4 and Table 3 (`remedies.py`), and Section 5 and Figure 5
(`score_shifts.py`).

## Configuration

| File | Contents |
|---|---|
| `configs/models.yaml` | Each model's OpenRouter id, family, context window, and request settings |
| `configs/venues.yaml` | Each venue's rating scale, accept threshold, and review-form wording |
| `prompts/*.md` | Prompt templates: a YAML header and a body with `{placeholder}` fields |

The prompt templates retain the wording used in the experiments. The reference to "verifier-derived confidence" in
`prompts/review.md` comes from an earlier workflow; no such verifier stage is used in this paper.
