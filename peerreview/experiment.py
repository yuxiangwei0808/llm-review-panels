"""Experiments: a panel, its venues, and its papers, read from ``configs/experiments/<name>.yaml``.

The content scripts (``experiments/judge.py``, ``experiments/human_coverage.py``) run on the experiment named by the
environment variable ``PEERREVIEW_EXPERIMENT`` (a name under ``configs/experiments/`` or a path to a YAML file) and write
to ``results/<name>/content/``. A config holds::

    name: iclr2022_prechatgpt
    models: [gpt-5.6-terra, gemini-3.1-pro, ...]
    venues: [iclr2022]
    manifest: data/manifests/iclr2022_prechatgpt.json
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .config import CONFIGS, ROOT


@dataclass(frozen=True)
class Experiment:
    name: str
    models: tuple[str, ...]
    venues: tuple[str, ...]
    manifest: Path


@lru_cache
def current() -> Experiment:
    chosen = os.environ.get("PEERREVIEW_EXPERIMENT")
    if not chosen:
        raise SystemExit("set PEERREVIEW_EXPERIMENT to an experiment under configs/experiments/, e.g. iclr2022_prechatgpt")
    path = Path(chosen) if chosen.endswith((".yaml", ".yml")) else CONFIGS / "experiments" / f"{chosen}.yaml"
    spec = yaml.safe_load(path.read_text())
    return Experiment(name=spec["name"], models=tuple(spec["models"]), venues=tuple(spec["venues"]),
                      manifest=ROOT / spec["manifest"])
