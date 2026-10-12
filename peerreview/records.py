"""Raw call records: one JSON object per line under ``res/<model>/<venue>/<stage>[_<tag>]/``.

Stages and their record files:

=================  ======================  ==========================================
stage              file                    call types
=================  ======================  ==========================================
``paper_review``   ``paper_review.jsonl``  ``A`` own review, ``B`` own-review rubric
``fixed_evidence`` ``fixed_evidence.jsonl`` ``F_holistic`` shared evidence, ``F_itemized`` shared rubric
=================  ======================  ==========================================

A run directory also holds ``input_artifacts/`` (every rendered prompt, named by SHA-256),
``paper_inputs/`` (paper text as sent), ``manifest.jsonl`` (the papers), ``run_meta.json`` and,
for ``fixed_evidence``, ``dossier_items.jsonl`` (the human points as rated in that run).

Record fields: ``call_id, block, call_type, mode, paper_id, model_id, run_idx, model_snapshot,
temperature, prompt_hash, prompt_version, input_hash, dependency_hash, execution_model,
emitted_score, accept_prob, distribution, items, item_ratings, reasoning_text, raw_response,
parse_ok, parse_error, usage, latency_s, timestamp, attempt, extra``.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from .config import DATA, RAW_OUTPUTS

STAGE_FILE = {"paper_review": "paper_review.jsonl", "fixed_evidence": "fixed_evidence.jsonl",
              "llm_as_ac": "llm_as_ac.jsonl"}
PANEL_MANIFEST = DATA / "manifests" / "core_panel_20260802.json"


@lru_cache
def panel() -> dict:
    """The eight-model panel behind the paper: models, venues, and each run's directory tag."""
    return json.loads(PANEL_MANIFEST.read_text())


def run_dir(model: str, venue: str, stage: str, root: Path = RAW_OUTPUTS,
            tag: str | None = None) -> Path:
    """``res/<model>/<venue>/<stage>[_<tag>]``. With ``tag=None`` the tag of the paper's panel
    is used (repaired or re-extracted runs carry one); new experiments pass their own tag."""
    if tag is None:
        tag = panel()["result_tags"].get(venue, {}).get(model, "")
    return root / model / venue / (f"{stage}_{tag}" if tag else stage)


def iter_records(model: str, venue: str, stage: str, call_type: str | None = None,
                 root: Path = RAW_OUTPUTS, tag: str | None = None) -> Iterator[dict]:
    path = run_dir(model, venue, stage, root, tag) / STAGE_FILE[stage]
    with path.open() as lines:
        for line in lines:
            if call_type and f'"call_type": "{call_type}"' not in line:
                continue            # skip before decoding: records are large
            record = json.loads(line)
            if call_type is None or record["call_type"] == call_type:
                yield record


def read_artifact(model: str, venue: str, stage: str, relative: str,
                  root: Path = RAW_OUTPUTS, tag: str | None = None) -> str:
    return (run_dir(model, venue, stage, root, tag) / relative).read_text()
