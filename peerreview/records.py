"""Records of model calls: one JSON object per line under ``results/first_reviews/<model>/<venue>/``.

================================ ================================= =============================================
call (``call_type``)             file                              what the model does
================================ ================================= =============================================
``review``                       ``reviews.jsonl``                 reviews the paper and gives a score
``score_from_human_reviews``     ``scores_from_human_reviews.jsonl`` scores the paper from the human reviews,
                                                                   ratings removed
================================ ================================= =============================================

The folder also holds ``input_artifacts/`` (every rendered prompt, named by SHA-256), ``paper_inputs/`` (the paper text
as sent), ``manifest.jsonl`` (the papers), and ``run_meta.json`` (the venue and its rating scale).

Record fields: ``call_id, call_type, paper_id, model_id, run_idx, model_snapshot, temperature, prompt_hash,
prompt_version, input_hash, execution_model, emitted_score, accept_prob, items, reasoning_text, raw_response, parse_ok,
parse_error, usage, latency_s, timestamp, attempt, extra``.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .config import RAW_OUTPUTS

FILES = {"review": "reviews.jsonl", "score_from_human_reviews": "scores_from_human_reviews.jsonl"}


def run_dir(model: str, venue: str, root: Path = RAW_OUTPUTS) -> Path:
    return root / model / venue


def iter_records(model: str, venue: str, call: str = "review", root: Path = RAW_OUTPUTS) -> Iterator[dict]:
    with (run_dir(model, venue, root) / FILES[call]).open(encoding="utf-8") as lines:
        for line in lines:
            if f'"call_type": "{call}"' in line:     # skip before decoding: records are large
                yield json.loads(line)
