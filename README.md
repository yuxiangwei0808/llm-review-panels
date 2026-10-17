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

The review records, judge labels, and human annotations are in `dataset.zip`. Unzip it in the repository root; it
creates `data/` and `results/`, and its `DATA_README.md` describes every file.

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
configs/       models, venues, and experiment settings
prompts/       prompt templates
```

### `peerreview`

| Module | Contents |
|---|---|
| `client.py` | Model calls through OpenRouter, with an on-disk response cache and a cost ceiling |
| `config.py` | Venue and model settings from `configs/*.yaml` |
| `prompts.py` | Loads the templates in `prompts/` and fills their fields |
| `parsing.py` | Parses model responses into scores and review items |
| `records.py` | Reads and writes per-call records, one JSON object per line |
| `papers.py` | Paper lists, the paper text a model reads, and the human-review dossiers |
| `runner.py` | First reviews for a set of papers and models |
| `content.py` | Weakness extraction from human reviews and the coverage judge |
| `area_chair.py` | The area-chair prompt and the parser of its decision |
| `experiment.py` | Experiment selection for the coverage judge (`PEERREVIEW_EXPERIMENT`) |

## Running the panels

The scripts in `experiments/` call models; each script's docstring gives its options, and every run takes a spending
ceiling (`--max-usd`). Responses are cached under `cache/`, so a rerun resumes where it stopped and retries failed
cells with fresh samples. The scripts after first reviews select the venue with the environment variable `PANEL_CORPUS`
(`iclr2021`, `iclr2022`, `neurips2022`, or `2023-25`; `iclr2022` when unset); the coverage judge selects a panel and
venues with `PEERREVIEW_EXPERIMENT` (a config in `configs/experiments/`). The steps depend on each other in this order:

1. `reviews.py`: first reviews.
2. `judge.py extract`, then `judge.py match`: the weaknesses in the human reviews and the labels of the first reviews;
   then `human_coverage.py`.
3. `discussion.py run` with `second_look` and `discussion`, then the restricted discussions (they run on the papers
   complete in both), and the rebuttal conditions; then `discussion.py coverage`.
4. `notes.py run`, then `notes.py adoption`; `roles.py run`, then `roles.py coverage`; `area_chairs.py run`.

```bash
uv run python experiments/reviews.py iclr2025 --models kimi-k3 grok-4.7 --tag panel8f --max-usd 50
PANEL_CORPUS=iclr2021 uv run python experiments/discussion.py run --arms second_look discussion --max-usd 300
PEERREVIEW_EXPERIMENT=iclr2021_prechatgpt uv run python experiments/judge.py match --annotator gemini-3.5-flash-lite --tag prechatgpt
```

| Script | Runs | Prompts |
|---|---|---|
| `reviews.py` | First reviews; with `--calls F_holistic`, scores from the human reviews instead | `call_a`, `call_f_holistic` |
| `judge.py` | `extract`: the weaknesses in each paper's human reviews; `match`: the coverage judge's labels of the first reviews | `extract_points`, `match_points` |
| `human_coverage.py` | The coverage judge applied to human committees | `match_points` |
| `discussion.py` | `run`: plain revision; discussion with the other reviews, with scores hidden, with verdicts hidden, or with an instruction to judge independently; revision after author responses or after a reply to the reviewer's own review. `coverage`: the coverage judge on the revisions | `call_a_second_look`, `call_a_discussion*`, `call_a_revision`, `author_reply` |
| `notes.py` | `run`: revision or fresh review after notes with weaknesses the panel missed, and control notes. `adoption`: which notes the reviews raise | `call_a_notes`, `call_a_fresh_notes`, `match_points` |
| `roles.py` | `run`: first reviews with an assigned focus. `coverage`: the coverage judge on them | `call_a_role`, `match_points` |
| `area_chairs.py` | Area-chair decisions from the human reviews or the panel's first reviews | `ac_binary` |
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
| `configs/experiments/*.yaml` | The eight-family panel on each venue group (models, venues, paper manifest), for the coverage judge |
| `prompts/*.md` | Prompt templates: a YAML header and a body with `{placeholder}` fields |
