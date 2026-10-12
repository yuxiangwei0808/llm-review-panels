"""Shared data loading and measures for the analysis scripts. Everything reads the saved records; no model calls.

Scores are rescaled to [0, 1] per venue with the venue's rating scale (configs/venues.yaml), human ratings alike.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from peerreview import content, records  # noqa: E402
from peerreview.config import venues  # noqa: E402

RESULTS, DATA, OUTPUTS = ROOT / "results", ROOT / "data", ROOT / "outputs"
PANEL = ["gpt-5.6-terra", "gemini-3.1-pro", "claude-sonnet-5", "deepseek-v4-flash-0731", "glm-5.2",
         "qwen3.8-2.4t-a95b", "grok-4.7", "kimi-k3"]                          # the eight-family panel
FOUR_FAMILY = ["gpt-5.1", "gpt-5.6-luna", "gpt-5.6-terra", "gemini-3.1-pro", "gemini-3.5-flash",
               "gemini-3.5-flash-lite", "deepseek-v4-flash-0731", "glm-5.2"]  # the Families comparison panel
# (name, venue key, results folder, venue group)
VENUES = [("ICLR 2021", "iclr2021", "iclr2021_prechatgpt", "2021-22"),
          ("ICLR 2022", "iclr2022", "iclr2022_prechatgpt", "2021-22"),
          ("NeurIPS 2022", "neurips2022", "neurips2022_prechatgpt", "2021-22"),
          ("ICLR 2025", "iclr2025", "study2325", "2023-25"),
          ("NeurIPS 2023", "neurips2023", "study2325", "2023-25"),
          ("ICML 2025", "icml2025", "study2325", "2023-25")]
PRE_CHATGPT = {"iclr2021", "iclr2022", "neurips2022"}
MANIFESTS = {"iclr2021": "iclr2021_prechatgpt_run.json", "iclr2022": "iclr2022_prechatgpt.json",
             "neurips2022": "neurips2022_prechatgpt_run.json"}                 # 2023-25 venues: study_manifest.json
SEED, DRAWS, BOOT = 20260930, 200, 500


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def folder(venue: str) -> str:
    return next(f for _, v, f, _ in VENUES if v == venue)


def norm(venue: str, x: float) -> float:
    s = venues()[venue].scores
    return (float(x) - min(s)) / (max(s) - min(s))


def manifest(venue: str) -> dict[str, dict]:
    """paper id -> paper (title, decision, human ratings, ...) for one venue."""
    papers = json.loads((DATA / "manifests" / MANIFESTS.get(venue, "study_manifest.json")).read_text())["papers"]
    out = {p["paper_id"]: p for p in papers if p.get("venue_key", venue) == venue}
    for p in out.values():
        p.setdefault("venue_key", venue)
    return out


def ratings(p: dict) -> list[float]:
    """A paper's human ratings, rescaled to [0, 1]."""
    raw = p["human_ratings"] if isinstance(p["human_ratings"], list) else json.loads(p["human_ratings"])
    return [norm(p["venue_key"], r) for r in raw]


def decision(p: dict) -> float:
    return {"accept": 1.0, "reject": 0.0}.get(p["outcome"], np.nan)


# ---------------------------------------------------------------- first reviews and the conditions built on them

def first_reviews(venue: str, run: int = 0) -> dict[tuple[str, str], float]:
    """(paper, model) -> first-review score on [0, 1]: the last parsed review with items of the given run."""
    tag = "prechatgpt" if venue in PRE_CHATGPT else "panel8f"
    out = {}
    for m in PANEL:
        for r in records.iter_records(m, venue, "paper_review", "A", tag=tag):
            if r["run_idx"] == run and r["parse_ok"] and r["items"]:
                out[(r["paper_id"], m)] = norm(venue, r["emitted_score"])
    return out


def score_table(venue: str) -> pd.DataFrame:
    """Papers x the eight models, first reviews on [0, 1], papers with all eight. The 2021-22 venues read the score
    table written when the first reviews were collected (results/<venue>/scores.csv), as Table 1 does; it lacks two
    NeurIPS 2022 papers whose reviews were completed later."""
    if venue in PRE_CHATGPT:
        s = pd.read_csv(RESULTS / folder(venue) / "scores.csv")
        s = s[(s.call_type == "A") & (s.run_idx == 0) & s.model_id.isin(PANEL)]
        return s.pivot_table(index="paper_id", columns="model_id", values="score")[PANEL].dropna()
    man = manifest(venue)
    r1 = first_reviews(venue)
    rows = [dict(paper_id=p, model_id=m, score=s) for (p, m), s in r1.items() if p in man]
    return pd.DataFrame(rows).pivot_table(index="paper_id", columns="model_id", values="score")[PANEL].dropna()


def conditions(venue: str) -> dict[tuple[str, str, str], float]:
    """(paper, model, condition) -> score on [0, 1] for every condition of the paper (field ``arm`` in the records):
    first reviews (round1, and run1 for the fresh-review comparison), the plain revision, discussion and its
    restrictions, rebuttal, notes, and roles."""
    d = folder(venue)
    man = manifest(venue)
    S = {}
    for name in ("records.jsonl", "records_rebuttal.jsonl"):
        for r in jsonl(RESULTS / d / "discussion" / name):
            if r["parse_ok"] and r["paper_id"] in man:
                S[(r["paper_id"], r["model_id"], r["arm"])] = norm(venue, r["emitted_score"])
    notes = RESULTS / ("study2325_panel8f" if d == "study2325" else d) / "injection" / "records.jsonl"
    for r in jsonl(notes):
        if r.get("parse_ok") and r["paper_id"] in man and r["arm"] in ("missed", "fresh_missed"):
            S[(r["paper_id"], r["model_id"], r["arm"])] = norm(venue, r["emitted_score"])
    for r in jsonl(RESULTS / d / "roles" / "records.jsonl"):
        if r.get("parse_ok") and r["paper_id"] in man:
            S[(r["paper_id"], r["model_id"], "roles")] = norm(venue, r["emitted_score"])
    for run, arm in ((0, "round1"), (1, "run1")):
        for (p, m), s in first_reviews(venue, run).items():
            if p in man:
                S[(p, m, arm)] = s
    return S


def complete(man: dict, S: dict, arms) -> list[str]:
    """Papers on which every panel model has a score under every one of ``arms``."""
    return [p for p in man if all((p, m, a) in S for m in PANEL for a in arms)]


def matrix(S: dict, ids: list[str], arm: str) -> np.ndarray:
    return np.array([[S[(p, m, arm)] for m in PANEL] for p in ids])


# ---------------------------------------------------------------- measures

def error_correlation(X: np.ndarray, h: np.ndarray) -> float:
    """Mean pairwise Pearson correlation of the models' errors: residuals of each model's own least-squares line on
    the mean human rating (so leniency and scale use are removed)."""
    R = np.column_stack([X[:, j] - np.polyval(np.polyfit(h, X[:, j], 1), h) for j in range(X.shape[1])])
    return float(np.nanmean(np.corrcoef(R.T)[np.triu_indices(X.shape[1], 1)]))


def measures(X: np.ndarray, h: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Panel error (1-10 points), decision AUROC, score correlation, error correlation, mean score (1-10 points)."""
    from sklearn.metrics import roc_auc_score
    m = X.mean(1)
    ok = ~np.isnan(t)
    return np.array([9 * np.abs(m - h).mean(), roc_auc_score(t[ok], m[ok]), np.corrcoef(m, h)[0, 1],
                     error_correlation(X, h), 9 * m.mean()])


MEASURES = ("panel_error", "auroc", "score_corr", "error_corr", "score_change")


def held_out(man: dict, L: pd.DataFrame) -> dict:
    """Held-out correlation. On papers with at least four human ratings, hold out one human rating and correlate it
    (Pearson, across papers) with the mean score of an LLM panel of 1-8 models (every subset) or of a human committee
    of 1-3 of the paper's other reviewers; average over 200 draws. Gains run from one to eight LLMs and from one to
    three humans; 95% intervals resample papers (500 draws)."""
    ids = [p for p in L.index if len(ratings(man[p])) >= 4]
    X = L.loc[ids].to_numpy(dtype=float)
    R = [np.array(ratings(man[p])) for p in ids]
    rng = np.random.default_rng(SEED)
    n = len(ids)
    held = np.zeros((DRAWS, n))
    hum = {k: np.zeros((DRAWS, n)) for k in (1, 2, 3)}
    for d in range(DRAWS):
        for i, r in enumerate(R):
            j = rng.integers(len(r))
            held[d, i] = r[j]
            rest = np.delete(r, j)
            for k in (1, 2, 3):
                hum[k][d, i] = rng.choice(rest, size=k, replace=False).mean()
    means = {k: np.stack([X[:, list(c)].mean(axis=1) for c in itertools.combinations(range(8), k)]) for k in range(1, 9)}
    z = lambda a: (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)

    def curves(idx):
        hz = z(held[:, idx])
        llm = {k: float((z(M[:, idx]) @ hz.T / len(idx)).mean()) for k, M in means.items()}
        humans = {k: float((z(H[:, idx]) * hz).mean(1).mean()) for k, H in hum.items()}
        return llm, humans

    llm, humans = curves(np.arange(n))
    sub = np.random.default_rng(SEED + 1)
    B = [curves(sub.integers(0, n, n)) for _ in range(BOOT)]
    ci = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    out = dict(papers=n,
               llm={k: [v, *ci([b[0][k] for b in B])] for k, v in llm.items()},
               human={k: [v, *ci([b[1][k] for b in B])] for k, v in humans.items()},
               gain_llm_1_to_8=[llm[8] - llm[1], *ci([b[0][8] - b[0][1] for b in B])],
               gain_human_1_to_3=[humans[3] - humans[1], *ci([b[1][3] - b[1][1] for b in B])],
               gain_difference=[(llm[8] - llm[1]) - (humans[3] - humans[1]),
                                *ci([(b[0][8] - b[0][1]) - (b[1][3] - b[1][1]) for b in B])])
    hk, hv = list(humans), [humans[k] for k in humans]
    out["human_equivalent"] = float(np.interp(llm[8], hv, hk)) if hv[0] <= llm[8] <= hv[-1] else None
    return out


def miss_rate(items_path: Path, labels: pd.DataFrame, papers: list[str]) -> dict:
    """Share (%) of human-raised weaknesses that no review among k pooled reviews raises, k = 1..8, averaged over
    every set of k models; 95% intervals resample papers (2,000 draws)."""
    weak: dict[str, list[str]] = {}
    for w in jsonl(items_path):
        if w["polarity"] == "weakness":
            weak.setdefault(w["paper_id"], []).append(w["id"])
    hit = labels[labels.recognition.isin(["specific", "partial"])]
    count = hit.groupby(["paper_id", "id"]).model.nunique()
    per = [[int(count.get((p, i), 0)) for i in weak[p]] for p in papers if weak.get(p)]
    point = content.omission_curve([c for cs in per for c in cs], 8)
    rng = np.random.default_rng(SEED)
    boots = np.array([content.omission_curve([c for i in rng.integers(0, len(per), len(per)) for c in per[i]], 8)
                      for _ in range(2000)])
    return dict(papers=len(per), curve=[[100 * v, 100 * np.percentile(boots[:, k], 2.5),
                                         100 * np.percentile(boots[:, k], 97.5)] for k, v in enumerate(point)])


def write(name: str, data: dict) -> None:
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / name).write_text(json.dumps(data, indent=1))
    print("->", OUTPUTS / name)
