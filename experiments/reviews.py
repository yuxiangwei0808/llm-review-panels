"""First reviews: each panel model reads a paper and writes a review with a score (prompt call_a).

Calls models through OpenRouter (set OPENROUTER_API_KEY) and needs the paper texts, which are not distributed: the
papers are public on OpenReview under the ids in data/manifests/, and peerreview.papers extracts their text.
Responses are cached under cache/, so a rerun resumes for free.

Usage:  uv run python experiments/reviews.py iclr2025 --models kimi-k3 grok-4.7 --tag panel8f --max-usd 50
Writes: results/raw/<model>/<venue>/paper_review_<tag>/paper_review.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from peerreview import runner  # noqa: E402
from peerreview.client import OpenRouterClient  # noqa: E402
from peerreview.config import models  # noqa: E402

MANIFESTS = {"iclr2021": "iclr2021_prechatgpt_run.json", "iclr2022": "iclr2022_prechatgpt.json",
             "neurips2022": "neurips2022_prechatgpt_run.json"}   # the 2023-25 venues: study_manifest.json


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("venue", help="iclr2021 | iclr2022 | neurips2022 | iclr2025 | neurips2023 | icml2025")
    ap.add_argument("--models", nargs="+", required=True, choices=sorted(models()))
    ap.add_argument("--tag", required=True, help="names the output folder paper_review_<tag>")
    ap.add_argument("--runs", type=int, default=1, help="independent reviews per paper and model")
    ap.add_argument("--max-usd", type=float, required=True, help="hard spending ceiling")
    a = ap.parse_args()
    papers = json.loads((ROOT / "data" / "manifests" / MANIFESTS.get(a.venue, "study_manifest.json")).read_text())["papers"]
    papers = pd.DataFrame(papers)
    if "venue_key" not in papers:
        papers["venue_key"] = a.venue
    client = OpenRouterClient(ROOT / "cache" / "responses" / a.tag, a.max_usd)
    runner.run(client, a.venue, a.models, papers[papers.venue_key == a.venue], tag=a.tag, calls=("A",), runs=a.runs)


if __name__ == "__main__":
    main()
