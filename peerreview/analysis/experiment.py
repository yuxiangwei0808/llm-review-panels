"""Which data the analysis scripts run on.

The default is the paper: eight models, three venues, the frozen score tables. A new experiment
is a YAML file under ``configs/experiments/`` selected with the PEERREVIEW_EXPERIMENT environment
variable (its name or a path); its results go to ``results/<name>/``. Example::

    name: icml_six_models
    models: [gpt-5.1, gpt-5.6-luna, gemini-3.1-pro, gemini-3.5-flash, glm-5.2, deepseek-v4-flash-0731]
    venues: [icml2025]
    scores: results/my_scores.csv          # written by `python -m peerreview scores`
    ac_records: results/my_ac.parquet      # optional, written by `python -m peerreview ac-table`
    manifest: data/manifests/study_manifest.json
    settings: [own, shared, own_rubric]    # optional; default: all four scoring settings
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .paths import MANIFEST, PROCESSED, ROOT

PAPER_MODELS = ("deepseek-v4-flash-0731", "gemini-3.1-pro", "gemini-3.5-flash",
                "gemini-3.5-flash-lite", "glm-5.2", "gpt-5.1", "gpt-5.6-luna", "gpt-5.6-terra")
PAPER_VENUES = ("iclr2025", "neurips2023", "icml2025")


@dataclass(frozen=True)
class Experiment:
    name: str = "paper"
    models: tuple[str, ...] = PAPER_MODELS
    venues: tuple[str, ...] = PAPER_VENUES
    manifest: Path = MANIFEST
    # Tables with the own-evidence calls (A, B) and the shared-evidence calls (F_*). A table
    # written by `python -m peerreview scores` holds both, so one path serves for the two.
    own_scores: Path = PROCESSED / "scores" / "score_run_records.csv"
    shared_scores: Path = PROCESSED / "scores" / "fixed_score_run_records.csv"
    min_reviews: int = 4          # a held-out target plus up to three pooled human reviewers
    # Scoring settings present in the score tables, "own" first. An experiment that did not run
    # the shared rubric (F_itemized needs itemized dossiers) lists the three it has.
    settings: tuple[str, ...] = ("own", "shared", "shared_rubric", "own_rubric")
    # Area-chair decisions written by `python -m peerreview ac-table`; None means the paper's.
    ac_records: Path | None = None

    @property
    def is_paper(self) -> bool:
        return self.name == "paper"


@lru_cache
def current() -> Experiment:
    chosen = os.environ.get("PEERREVIEW_EXPERIMENT", "")
    if not chosen or chosen == "paper":
        return Experiment()
    path = Path(chosen) if chosen.endswith((".yaml", ".yml")) \
        else ROOT / "configs" / "experiments" / f"{chosen}.yaml"
    spec = yaml.safe_load(path.read_text())
    scores = ROOT / spec["scores"]
    return Experiment(name=spec["name"], models=tuple(spec["models"]), venues=tuple(spec["venues"]),
                      manifest=ROOT / spec.get("manifest", MANIFEST.relative_to(ROOT)),
                      own_scores=scores, shared_scores=scores,
                      min_reviews=spec.get("min_reviews", 4),
                      settings=tuple(spec.get("settings", Experiment.settings)),
                      ac_records=ROOT / spec["ac_records"] if spec.get("ac_records") else None)
