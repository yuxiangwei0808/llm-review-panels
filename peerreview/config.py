"""Venue and model settings, loaded from ``configs/*.yaml``."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
PROMPTS = ROOT / "prompts"
DATA = ROOT / "data"
RAW_OUTPUTS = ROOT / "results" / "raw"   # first reviews: <model>/<venue>/<stage>_<tag>/ (records.py)
SEED = 20260727                     # seed for paper sampling and review order
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 1.0


@dataclass(frozen=True)
class Venue:
    key: str
    name: str
    prompt_name: str
    short_name: str
    scores: tuple[int, ...]
    accept_threshold: int
    scale_labels: dict[int, str]
    openreview: dict = field(default_factory=dict)

    @property
    def rating_scale_block(self) -> str:
        """Official review-form scale, as written into every scoring prompt."""
        lines = [f"- {s}: {self.scale_labels[s]}" for s in sorted(self.scores, reverse=True)]
        header = f"{self.name} Rating Scale (Discrete scale: {', '.join(map(str, self.scores))}):"
        return header + "\n\n" + "\n\n".join(lines)

    @property
    def allowed_scores(self) -> str:
        return "{" + ", ".join(map(str, self.scores)) + "}"

    def normalize(self, score: float) -> float:
        low, high = min(self.scores), max(self.scores)
        return (score - low) / (high - low)


@dataclass(frozen=True)
class Model:
    key: str                     # folder name under results/raw/ and model_id in every record
    family: str
    api_model: str               # OpenRouter identifier
    context_window_tokens: int
    accepts_temperature: bool = True
    # Extra OpenRouter request fields, e.g. {reasoning: {enabled: false}}; part of the cache key.
    request: dict = field(default_factory=dict)


@lru_cache
def venues() -> dict[str, Venue]:
    raw = yaml.safe_load((CONFIGS / "venues.yaml").read_text())
    return {key: Venue(key=key, name=v["name"], prompt_name=v["prompt_name"],
                       short_name=v["short_name"], scores=tuple(v["scores"]),
                       accept_threshold=v["accept_threshold"],
                       scale_labels={int(k): label for k, label in v["scale_labels"].items()},
                       openreview=v.get("openreview", {}))
            for key, v in raw.items()}


@lru_cache
def models() -> dict[str, Model]:
    raw = yaml.safe_load((CONFIGS / "models.yaml").read_text())
    return {key: Model(key=key, **m) for key, m in raw.items()}
