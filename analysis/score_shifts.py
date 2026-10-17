"""Section 5 (Figure 5): remedies shift the whole panel's scores together instead of correcting the misjudged papers.

For each remedy, against the same models' first reviews, pooled over the six venues:
  share fell          share of papers whose panel mean score fell (papers pooled)
  score correlation   change in the correlation of the panel mean score with the mean human rating (venue mean)
  error correlation   change in the error correlation among the eight LLMs (venue mean)
with 95% intervals resampling papers within each venue (2,000 draws), and the plain revision for reference. A full
fix would lower exactly the papers the panel overrated (``share_overrated``), reach a score correlation of 1, and
leave no shared error.

Usage:  uv run python analysis/score_shifts.py
Writes: outputs/score_shifts.json
"""
import numpy as np
import pandas as pd

from common import PANEL, VENUES, conditions, error_correlation, manifest, ratings, write

ARMS = {"plain_revision": "plain_revision", "roles": "roles", "notes": "missed", "discussion": "discussion",
        "hidden_scores": "hidden_scores", "hidden_verdicts": "hidden_verdicts", "independence": "independence"}


def long_table() -> pd.DataFrame:
    """One row per venue, paper, model, and remedy with all eight scores: score, first-review score, mean human rating."""
    rows = []
    for _, venue, _, _ in VENUES:
        man, S = manifest(venue), conditions(venue)
        for p in man:
            if any((p, m, "first_review") not in S for m in PANEL):
                continue
            h = float(np.mean(ratings(man[p])))
            for arm, key in ARMS.items():
                if all((p, m, key) in S for m in PANEL):
                    rows += [dict(venue=venue, paper=p, model=m, arm=arm, score=S[(p, m, key)], first=S[(p, m, "first_review")],
                                  human=h) for m in PANEL]
    return pd.DataFrame(rows)


def summarize(long: pd.DataFrame) -> dict:
    rng = np.random.default_rng(20261008)
    panel = long.groupby(["venue", "paper", "arm"]).agg(score=("score", "mean"), first=("first", "mean"),
                                                        human=("human", "first")).reset_index()
    out = {}
    for arm in ARMS:
        g, gl = panel[panel.arm == arm], long[long.arm == arm]
        venues = {}
        for v, gv in g.groupby("venue"):
            per_model = gl[gl.venue == v]
            S = per_model.pivot(index="paper", columns="model", values="score").loc[gv.paper]
            F = per_model.pivot(index="paper", columns="model", values="first").loc[gv.paper, S.columns]
            venues[v] = (gv[["first", "score", "human"]].to_numpy(), F.to_numpy(), S.to_numpy())

        def stat(sample):
            fell = np.concatenate([X[:, 1] - X[:, 0] < -1e-9 for X, _, _ in sample])
            rank = np.mean([np.corrcoef(X[:, 1], X[:, 2])[0, 1] - np.corrcoef(X[:, 0], X[:, 2])[0, 1] for X, _, _ in sample])
            err = [(error_correlation(F, X[:, 2]), error_correlation(S, X[:, 2])) for X, F, S in sample]
            return fell.mean(), rank, np.mean([b - a for a, b in err])

        point = stat(venues.values())
        boots = np.array([stat([(X[i], F[i], S[i]) for X, F, S in venues.values() for i in [rng.integers(0, len(X), len(X))]])
                          for _ in range(2000)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        out[arm] = {k: [float(point[j]), float(lo[j]), float(hi[j])]
                    for j, k in enumerate(("share_fell", "score_corr_change", "error_corr_change"))} | {"papers": len(g)}
        print(f"{arm:15s} scores fell on {point[0]:.0%} of papers; score correlation {point[1]:+.3f}; error correlation "
              f"{point[2]:+.3f}", flush=True)
    base = panel[panel.arm == "plain_revision"]
    out["share_overrated"] = float((base["first"] - base.human > 1e-9).mean())
    print(f"the panel overrated {out['share_overrated']:.0%} of papers in its first reviews")
    return out


def main():
    write("score_shifts.json", summarize(long_table()))


if __name__ == "__main__":
    main()
