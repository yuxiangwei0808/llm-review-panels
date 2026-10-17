"""First reviews: each model reads a paper and writes a review with a score (prompt review).

With --calls score_from_human_reviews, each model instead scores the paper from the human reviews with ratings removed
(prompt score_from_human_reviews), which needs the human-review dossiers under data/dossiers/.

Calls models through OpenRouter (set OPENROUTER_API_KEY) and needs the paper texts, which are not distributed: each
manifest entry's paper_text_path (data/raw/paper_text/<venue>/<paper_id>.md) holds the text the models read: the
submitted PDF converted with Nemotron OCR v2 (the 2023-25 venues and 144 ICLR 2022 papers) or PyMuPDF text extraction
(the other papers), as the manifest field paper_text_source records. The papers are public on OpenReview under the ids
in data/manifests/. Responses are cached under cache/, so a rerun resumes for free.

Usage:  uv run python experiments/reviews.py iclr2025 --models kimi-k3 grok-4.7 --max-usd 50
Writes: results/first_reviews/<model>/<venue>/reviews.jsonl (scores_from_human_reviews.jsonl)
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
from peerreview.config import MANIFESTS, models, venues  # noqa: E402



def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("venue", choices=sorted(venues()))
    ap.add_argument("--models", nargs="+", required=True, choices=sorted(models()))
    ap.add_argument("--calls", nargs="+", default=["review"], choices=["review", "score_from_human_reviews"])
    ap.add_argument("--runs", type=int, default=1, help="independent reviews per paper and model")
    ap.add_argument("--max-usd", type=float, required=True, help="hard spending ceiling")
    a = ap.parse_args()
    papers = pd.DataFrame(json.loads((ROOT / "data" / "manifests" / MANIFESTS[a.venue]).read_text())["papers"])
    client = OpenRouterClient(ROOT / "cache" / "responses" / "first_reviews", a.max_usd)
    runner.run(client, a.venue, a.models, papers[papers.venue_key == a.venue], calls=tuple(a.calls), runs=a.runs)


if __name__ == "__main__":
    main()
