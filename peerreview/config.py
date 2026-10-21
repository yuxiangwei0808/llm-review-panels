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
RAW_OUTPUTS = ROOT / "results" / "first_reviews"   # <model>/<venue>/ (records.py)
SEED = 20260727                     # seed for paper sampling and review order
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 1.0
PANEL = ["gpt-5.6-terra", "gemini-3.1-pro", "claude-sonnet-5", "deepseek-v4-flash-0731", "glm-5.2",
         "qwen3.8-2.4t-a95b", "grok-4.7", "kimi-k3"]                          # the eight-family panel
FOUR_FAMILY = ["gpt-5.1", "gpt-5.6-luna", "gpt-5.6-terra", "gemini-3.1-pro", "gemini-3.5-flash",
               "gemini-3.5-flash-lite", "deepseek-v4-flash-0731", "glm-5.2"]  # the four-family panel (Families)
GROUPS = {"iclr2021": ("iclr2021",), "iclr2022": ("iclr2022",), "neurips2022": ("neurips2022",),
          "2023-25": ("iclr2025", "neurips2023", "icml2025")}   # venue groups: results/<group>/, the VENUE variable
MANIFESTS = {v: f"{g}.json" for g, vs in GROUPS.items() for v in vs}   # under data/manifests/


@dataclass(frozen=True)
class Venue:
    key: str
    name: str
    prompt_name: str
    short_name: str
    scores: tuple[int, ...]
    accept_threshold: int
    scale_labels: dict[int, str]

    @property
    def rating_scale_block(self) -> str:
        """Official review-form scale, as written into every scoring prompt."""
        lines = [f"- {s}: {self.scale_labels[s]}" for s in sorted(self.scores, reverse=True)]
        header = f"{self.name} Rating Scale (Discrete scale: {', '.join(map(str, self.scores))}):"
        return header + "\n\n" + "\n\n".join(lines)

    @property
    def allowed_scores(self) -> str:
        return "{" + ", ".join(map(str, self.scores)) + "}"


@dataclass(frozen=True)
class Model:
    key: str                     # folder name under results/first_reviews/ and model_id in every record
    family: str
    api_model: str               # OpenRouter identifier
    context_window_tokens: int
    accepts_temperature: bool = True
    # Extra OpenRouter request fields, e.g. {reasoning: {enabled: false}}; part of the cache key.
    request: dict = field(default_factory=dict)


@lru_cache
def venues() -> dict[str, Venue]:
    raw = yaml.safe_load((CONFIGS / "venues.yaml").read_text(encoding="utf-8"))
    return {key: Venue(key=key, name=v["name"], prompt_name=v["prompt_name"],
                       short_name=v["short_name"], scores=tuple(v["scores"]),
                       accept_threshold=v["accept_threshold"],
                       scale_labels={int(k): label for k, label in v["scale_labels"].items()})
            for key, v in raw.items()}


@lru_cache
def models() -> dict[str, Model]:
    raw = yaml.safe_load((CONFIGS / "models.yaml").read_text(encoding="utf-8"))
    return {key: Model(key=key, **m) for key, m in raw.items()}
