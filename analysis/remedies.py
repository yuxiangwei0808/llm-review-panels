"""Section 4 (Table 3): what each remedy changes, against its matched comparison on the same papers.

  Families          eight model families vs. eight models from four families (first reviews)
  Roles             reviews with an assigned aspect vs. first reviews
  Notes             revision after notes with missed weaknesses vs. plain revision
  Notes, fresh      fresh review after notes vs. a fresh review without them
  Hidden scores, Hidden verdicts, Independence    restricted discussions vs. plain revision
  Discussion, Rebuttal                            (for reference) vs. plain revision
Measures (common.measures): panel error, decision AUROC, score correlation, error correlation, mean score. Each is
the design minus its comparison; a venue group's value is the mean over its three venues, with a 95% interval that
resamples papers within each venue (1,000 draws).

Usage:  uv run python analysis/remedies.py
Writes: outputs/remedies.json
"""
import numpy as np
import pandas as pd

from common import (FOUR_FAMILY, MEASURES, PANEL, PRE_CHATGPT, RESULTS, VENUES, complete, conditions, decision, manifest,
                    matrix, measures, norm, ratings, write)
from peerreview import records

DESIGNS = [("notes", "missed", "plain_revision"), ("notes_fresh", "fresh_missed", "first_review_run1"), ("roles", "roles", "first_review"),
           ("discussion", "discussion", "plain_revision"), ("hidden_scores", "hidden_scores", "plain_revision"),
           ("hidden_verdicts", "hidden_verdicts", "plain_revision"), ("independence", "independence", "plain_revision"),
           ("rebuttal", "rebuttal", "plain_revision")]


def family_scores(models, venue):
    """First reviews of run 0 on [0, 1], papers x models (2021-22 venues)."""
    rows = []
    for m in models:
        for r in records.iter_records(m, venue):
            if r["parse_ok"] and r["run_idx"] == 0 and r["emitted_score"] is not None:
                rows.append(dict(paper_id=r["paper_id"], model_id=m, score=norm(venue, r["emitted_score"])))
    s = pd.DataFrame(rows).drop_duplicates(["paper_id", "model_id"], keep="last")
    return s.pivot_table(index="paper_id", columns="model_id", values="score")[models].dropna()


def family_scores_2325():
    """2023-25 venues: the eight-family panel as first run for the comparison (the four models it shares with the
    four-family panel from that panel's run, the other four from the eight-family run) and the four-family panel."""
    s = pd.read_csv(RESULTS / "2023-25" / "four_family_scores.csv", low_memory=False)
    s = s[(s.call_type == "review") & (s.run_idx == 0)]
    shared = [m for m in PANEL if m in FOUR_FAMILY]
    rows = [s[s.model_id.isin(shared)][["paper_id", "model_id", "score"]]]
    for m in [m for m in PANEL if m not in shared]:
        for v in ("iclr2025", "neurips2023", "icml2025"):
            for r in records.iter_records(m, v):
                if r["parse_ok"] and r["run_idx"] == 0 and r["emitted_score"] is not None:
                    rows.append(pd.DataFrame([dict(paper_id=r["paper_id"], model_id=m, score=norm(v, r["emitted_score"]))]))
    new = pd.concat(rows).drop_duplicates(["paper_id", "model_id"], keep="last")
    new = new.pivot_table(index="paper_id", columns="model_id", values="score")[PANEL].dropna()
    old = s[s.model_id.isin(FOUR_FAMILY)].pivot_table(index="paper_id", columns="model_id", values="score")[FOUR_FAMILY].dropna()
    return new, old


def pairs():
    """{design: {venue: (A, B, h, t)}}: the design's scores, its comparison's, mean human rating, decision; same papers."""
    out = {}
    for _, venue, _, _ in VENUES:
        man, S = manifest(venue), conditions(venue)
        for design, a, b in DESIGNS:
            ids = complete(man, S, (a, b))
            if len(ids) >= 20:
                out.setdefault(design, {})[venue] = (matrix(S, ids, a), matrix(S, ids, b),
                                                     np.array([np.mean(ratings(man[p])) for p in ids]),
                                                     np.array([decision(man[p]) for p in ids]))
    new25, old25 = family_scores_2325()
    for _, venue, _, _ in VENUES:
        man = manifest(venue)
        new, old = (family_scores(PANEL, venue), family_scores(FOUR_FAMILY, venue)) if venue in PRE_CHATGPT else (new25, old25)
        ids = sorted(set(new.index) & set(old.index) & set(man))
        out.setdefault("families", {})[venue] = (new.loc[ids].to_numpy(dtype=float), old.loc[ids].to_numpy(dtype=float),
                                                 np.array([np.mean(ratings(man[p])) for p in ids]),
                                                 np.array([decision(man[p]) for p in ids]))
    return out


def main():
    rng = np.random.default_rng(20261010)
    res = {}
    for design, by_venue in pairs().items():
        res[design] = {}
        for group in ("2021-22", "2023-25"):
            vs = [v for _, v, _, g in VENUES if g == group and v in by_venue]
            if len(vs) < 3:
                continue
            diff = lambda v, i: measures(by_venue[v][0][i], by_venue[v][2][i], by_venue[v][3][i]) - \
                measures(by_venue[v][1][i], by_venue[v][2][i], by_venue[v][3][i])
            n = {v: len(by_venue[v][2]) for v in vs}
            pt = np.mean([diff(v, np.arange(n[v])) for v in vs], 0)
            bs = np.array([np.mean([diff(v, rng.integers(0, n[v], n[v])) for v in vs], 0) for _ in range(1000)])
            lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
            res[design][group] = {k: dict(change=float(pt[j]), ci=[float(lo[j]), float(hi[j])]) for j, k in enumerate(MEASURES)} \
                | {"papers": n}
            print(f"{design:15s} {group}: " + "  ".join(f"{k} {pt[j]:+.3f} [{lo[j]:+.3f},{hi[j]:+.3f}]"
                                                        for j, k in enumerate(MEASURES)), flush=True)
    write("remedies.json", res)


if __name__ == "__main__":
    main()
