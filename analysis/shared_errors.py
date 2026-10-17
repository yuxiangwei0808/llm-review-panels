"""Section 3.1: the panel's members miss the same weaknesses and misjudge the same papers.

  miss rate          Figure 4: share of human-raised weaknesses that no review among 1-8 pooled first reviews raises,
                     per venue (the coverage judge's labels), and the human committees' curve for comparison
  missed by kind     share of the panel's misses that are substantive and about rigor (soundness or experiments)
  error correlation  mean pairwise error correlation of two LLMs and of two human reviewers against the same
                     leave-two-out human reference (the mean of the paper's other ratings), papers with >= 4 ratings

Usage:  uv run python analysis/shared_errors.py
Writes: outputs/shared_errors.json
"""
import json

import numpy as np
import pandas as pd

from common import (PANEL, PRE_CHATGPT, RESULTS, VENUES, complete, conditions, folder, manifest, matrix, miss_rate,
                    ratings, score_table, write)

RIGOR = {"soundness", "experiments"}


def content_dir(venue):
    return RESULTS / (folder(venue) if venue in PRE_CHATGPT else "study2325_panel8f") / "content"


def first_review_labels(venue):
    """The judge's labels of the eight first reviews (run 0) on the papers with all eight labelled."""
    labels = pd.read_csv(content_dir(venue) / "coverage_labels.csv")
    labels = labels[labels.model.isin(PANEL) & (labels.run == 0)]
    if venue not in PRE_CHATGPT:
        labels = labels[labels.paper_id.isin(manifest(venue))]
    full = sorted(p for p, g in labels.groupby("paper_id") if g.model.nunique() == 8)
    return labels, full


def human_curve(path):
    """Human committees' miss rate against the same weaknesses, k = 1..4 of the paper's reviewers (precomputed with the
    same judge on the human reviews, experiments/human_coverage.py)."""
    h = pd.read_csv(path).sort_values("reviews")
    return [[int(r.reviews), float(r.missed_pct), float(r.ci_low), float(r.ci_high)] for r in h.itertuples()]


def missed_by_kind(venue, labels, full):
    hit = labels[labels.recognition.isin(["specific", "partial"])].groupby(["paper_id", "id"]).model.nunique()
    rows = []
    for line in open(content_dir(venue) / "reference_items.jsonl"):
        x = json.loads(line)
        if x["paper_id"] in set(full) and x["polarity"] == "weakness" and x.get("kind") in ("substantive", "presentation"):
            rows.append(dict(paper=x["paper_id"], substantive=x["kind"] == "substantive",
                             rigor=x["kind"] == "substantive" and x.get("category") in RIGOR,
                             missed=hit.get((x["paper_id"], x["id"]), 0) == 0))
    W = pd.DataFrame(rows)
    g = W.assign(ms=W.missed & W.substantive, mp=W.missed & ~W.substantive, p=~W.substantive,
                 mr=W.missed & W.rigor).groupby("paper")
    counts = g[["substantive", "ms", "p", "mp", "mr"]].sum().to_numpy(dtype=float)

    def summary(c):
        missed = c[..., 1] + c[..., 3]
        return np.array([c[..., 1] / c[..., 0], c[..., 3] / c[..., 2], c[..., 1] / missed, c[..., 4] / missed]).T

    point = summary(counts.sum(0))
    rng = np.random.default_rng(20261008)
    n = len(counts)
    boot = summary(np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(2000)]) @ counts)
    ci = lambda j: [float(point[j]), float(np.percentile(boot[:, j], 2.5)), float(np.percentile(boot[:, j], 97.5))]
    return dict(papers=n, weaknesses=len(W), missed_substantive=ci(0), missed_presentation=ci(1),
                substantive_share_of_missed=ci(2), rigor_share_of_missed=ci(3))


def pair_correlations(X, R, rng, draws=100, n_boot=200):
    """Error correlation of two human reviewers and of two LLMs (mean over the 28 pairs) against the same reference: per
    paper and draw, two human reviewers form the human pair and the mean of the remaining ratings is the reference.
    Errors are residuals of a least-squares line on the reference (one line per LLM, one shared by the human pair)."""
    n = len(R)
    A, B, REF = np.empty((draws, n)), np.empty((draws, n)), np.empty((draws, n))
    for d in range(draws):
        for i, r in enumerate(R):
            a, b = rng.choice(len(r), 2, replace=False)
            A[d, i], B[d, i], REF[d, i] = r[a], r[b], np.delete(r, [a, b]).mean()
    resid = lambda x, h: x - np.polyval(np.polyfit(h, x, 1), h)

    def stats(idx):
        hum, llm = [], []
        for d in range(draws):
            ref = REF[d, idx]
            coef = np.polyfit(np.concatenate([ref, ref]), np.concatenate([A[d, idx], B[d, idx]]), 1)
            hum.append(np.corrcoef(A[d, idx] - np.polyval(coef, ref), B[d, idx] - np.polyval(coef, ref))[0, 1])
            C = np.corrcoef(np.column_stack([resid(X[idx, j], ref) for j in range(X.shape[1])]).T)
            llm.append(np.mean(C[np.triu_indices(X.shape[1], 1)]))
        return float(np.mean(hum)), float(np.mean(llm))

    hp, lp = stats(np.arange(n))
    boot = np.array([stats(rng.integers(0, n, n)) for _ in range(n_boot)])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(papers=n, human_pairs=[hp, *ci(boot[:, 0])], llm_pairs=[lp, *ci(boot[:, 1])],
                llm_minus_human=[lp - hp, *ci(boot[:, 1] - boot[:, 0])])


def baseline_inputs(venue):
    """First reviews (papers x 8) and each paper's human ratings, papers with >= 4 ratings. The 2021-22 venues use
    Table 1's papers; the 2023-25 venues the papers with first reviews and discussion records (Section 3.2)."""
    man = manifest(venue)
    if venue in PRE_CHATGPT:
        L = score_table(venue)
        ids, X = list(L.index), L.to_numpy(dtype=float)
    else:
        S = conditions(venue)
        ids = complete(man, S, ("round1", "second_look", "discussion"))
        X = matrix(S, ids, "round1")
    keep = [i for i, p in enumerate(ids) if len(ratings(man[p])) >= 4]
    return X[keep], [np.array(ratings(man[ids[i]])) for i in keep]


def main():
    out = {"miss_rate": {}, "missed_by_kind": {}, "human_committees": {}, "error_correlation": {}}
    for name, venue, _, _ in VENUES:
        labels, full = first_review_labels(venue)
        out["miss_rate"][name] = miss_rate(content_dir(venue) / "reference_items.jsonl", labels, full)
        out["missed_by_kind"][name] = missed_by_kind(venue, labels, full)
        o, k = out["miss_rate"][name], out["missed_by_kind"][name]
        print(f"{name:13s} eight reviews miss {o['curve'][7][0]:.0f}% of human-raised weaknesses ({o['papers']} papers); "
              f"{k['rigor_share_of_missed'][0]:.0%} of misses concern rigor")
    for name, venue in (("ICLR 2021", "iclr2021"), ("ICLR 2022", "iclr2022"), ("NeurIPS 2022", "neurips2022")):
        out["human_committees"][name] = human_curve(content_dir(venue) / "human_omission_curve.csv")
    out["human_committees"]["2023-25 venues"] = human_curve(RESULTS / "study2325_panel8f/content/human_omission_curve.csv")
    rng = np.random.default_rng(20261008)
    for name, venue, _, _ in VENUES:
        X, R = baseline_inputs(venue)
        out["error_correlation"][name] = e = pair_correlations(X, R, rng)
        print(f"{name:13s} error correlation: two LLMs {e['llm_pairs'][0]:.2f}, two humans {e['human_pairs'][0]:.2f} "
              f"({e['papers']} papers)", flush=True)
    write("shared_errors.json", out)


if __name__ == "__main__":
    main()
