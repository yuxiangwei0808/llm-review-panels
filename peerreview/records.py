"""Records of model calls: one JSON object per line under ``results/raw/<model>/<venue>/<stage>_<tag>/``.

=================== ========================= ===========================================================
stage               file                      call
=================== ========================= ===========================================================
``paper_review``    ``paper_review.jsonl``    ``A``: the model reviews the paper and gives a score
``fixed_evidence``  ``fixed_evidence.jsonl``  ``F_holistic``: the model scores the human reviews, ratings removed
=================== ========================= ===========================================================

The tag names the run (``prechatgpt`` for the 2021-22 venues, ``panel8f`` for the eight-family panel on the 2023-25
venues). A run folder also holds ``input_artifacts/`` (every rendered prompt, named by SHA-256), ``paper_inputs/``
(the paper text as sent), ``manifest.jsonl`` (the papers), and ``run_meta.json`` (the venue and its rating scale).

Record fields: ``call_id, block, call_type, mode, paper_id, model_id, run_idx, model_snapshot, temperature,
prompt_hash, prompt_version, input_hash, execution_model, emitted_score, accept_prob, items, reasoning_text,
raw_response, parse_ok, parse_error, usage, latency_s, timestamp, attempt, extra``.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .config import RAW_OUTPUTS

STAGE_FILE = {"paper_review": "paper_review.jsonl", "fixed_evidence": "fixed_evidence.jsonl"}


def run_dir(model: str, venue: str, stage: str, root: Path = RAW_OUTPUTS, tag: str = "") -> Path:
    """``<root>/<model>/<venue>/<stage>_<tag>`` (``<stage>`` without a tag)."""
    return root / model / venue / (f"{stage}_{tag}" if tag else stage)


def iter_records(model: str, venue: str, stage: str, call_type: str | None = None,
                 root: Path = RAW_OUTPUTS, tag: str = "") -> Iterator[dict]:
    path = run_dir(model, venue, stage, root, tag) / STAGE_FILE[stage]
    with path.open() as lines:
        for line in lines:
            if call_type and f'"call_type": "{call_type}"' not in line:
                continue            # skip before decoding: records are large
            record = json.loads(line)
            if call_type is None or record["call_type"] == call_type:
                yield record
