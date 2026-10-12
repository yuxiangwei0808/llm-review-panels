# Second Opinions or Echoes? Shared Errors in LLM Review Panels

Code for the paper (anonymized for review). Panels of eight LLMs from eight model families review 1,404 submissions to
six machine learning venues (ICLR 2021, ICLR 2022, NeurIPS 2022, ICLR 2025, NeurIPS 2023, ICML 2025), and we compare
them with the recorded human process through independent review, rebuttal, discussion, and the area chair's decision.

## Setup

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The data (review records, judge labels, annotations, corpora) are in `dataset.zip`, provided with the submission.
Unzip it in this folder; it creates `data/` and `results/`.

## Reproduce the main results

These scripts read the saved records and make no model calls. Each writes a JSON file to `outputs/` and prints the
numbers reported in the paper.

| Script | Paper | Runtime |
|---|---|---|
| `analysis/scaling.py` | Figure 1 and Table 1: gains from adding LLMs vs. adding human reviewers | 4 min |
| `analysis/shared_errors.py` | Section 3.1 and Figure 4: missed weaknesses, error correlation vs. human reviewers | 6 min |
| `analysis/review_cycle.py` | Sections 3.2–3.3 and Table 2: rebuttal, discussion, area chairs | 4 min |
| `analysis/remedies.py` | Section 4 and Table 3: the remedies | 12 min |
| `analysis/score_shifts.py` | Section 5 and Figure 5: scores shift together | 6 min |

```bash
uv run python analysis/scaling.py
```

`analysis/common.py` holds the data loading and the measures (held-out correlation, miss rate, error correlation,
panel error, decision AUROC, score correlation).

## Run the experiments

These scripts produced the records in `results/`. They call models through OpenRouter (set `OPENROUTER_API_KEY`) and
need the paper texts, which are not distributed: the papers are public on OpenReview under the ids in
`data/manifests/`. Each script's docstring gives its usage; responses are cached, so reruns resume. The runners
for revisions and area chairs select the venues with the environment variable `PANEL_CORPUS` (`iclr2021`,
`iclr2022`, `neurips2022`, or `2023-25`).

| Script | Stage |
|---|---|
| `experiments/reviews.py` | First reviews (prompt `call_a`) |
| `experiments/discussion.py` | Plain revision, discussion and its restrictions, rebuttal |
| `experiments/notes.py` | Notes with missed weaknesses, and their controls |
| `experiments/roles.py` | Reviews with an assigned aspect |
| `experiments/area_chairs.py`, `experiments/area_chairs_iclr2022.py` | Nine LLM area chairs |
| `experiments/judge.py` | Weakness extraction from human reviews and the coverage judge |
| `experiments/human_coverage.py` | Human committees' coverage (Figure 4's human curve) |

`experiments/forums.py` and `experiments/verdict_mask.py` are helpers. Models are configured in
`configs/models.yaml`, venues and their rating scales in `configs/venues.yaml`, and the prompts are in `prompts/`.

## Layout

| Path | Contents |
|---|---|
| `analysis/` | The main results from the saved records |
| `experiments/` | The scripts that ran the panel |
| `peerreview/` | Model client, prompt rendering, response parsing, review runner, coverage judge, area chair |
| `prompts/` | Prompt templates of the reported experiments |
| `configs/` | Models, venues, and the first-review runs |

## Data guide

| Path (after unzipping `dataset.zip`) | Contents |
|---|---|
| `data/manifests/` | The six corpora: OpenReview ids, titles, abstracts, decisions, and human ratings |
| `results/raw/<model>/<venue>/<run>/paper_review.jsonl` | First reviews: items, reasoning, raw response, score, token usage |
| `results/<venue>/discussion/` | Revisions after a plain revision, discussion and its restrictions (`records.jsonl`), and rebuttal (`records_rebuttal.jsonl`) |
| `results/<venue>/injection/`, `results/<venue>/roles/` | Notes and roles |
| `results/<venue>/area_chair/` | Meta-reviews and decisions of the nine LLM area chairs |
| `results/<venue>/content/` | Weaknesses extracted from the human reviews and the judge's labels |

`<venue>` is `iclr2021_prechatgpt`, `iclr2022_prechatgpt`, `neurips2022_prechatgpt`, or `study2325` (ICLR 2025,
NeurIPS 2023, ICML 2025; their notes and judge labels are in `study2325_panel8f`). The field `arm` in the records
names the condition:

| `arm` | Condition in the paper |
|---|---|
| `second_look` | Plain revision (control) |
| `discussion` | Discussion: peers' reviews with scores |
| `discussion_text`, `discussion_masked` | Hidden scores; hidden verdicts |
| `discussion_independent` | Independence instruction |
| `rebuttal`, `llm_rebuttal` | Author responses; replies written to each LLM's review (ICLR 2022) |
| `missed`, `mentioned`, `false`, `neutral` | Notes; control notes already raised, from another paper, or neutral |
| `fresh_missed`, `fresh_false` | Notes before a fresh review |
| `H0`, `P0` | Area chairs reading human reviews or the panel's reviews |

## License

Paper metadata and human ratings come from OpenReview and remain under their original terms. A license for the code
and the generated data will be added with the public release.
