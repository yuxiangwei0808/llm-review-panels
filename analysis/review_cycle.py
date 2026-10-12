"""Sections 3.2 and 3.3 (Table 2): later review stages do not correct the panel's shared errors.

  rebuttal      share of scores that change after a plain revision and after the real author responses; on ICLR 2022
                also replies written to each LLM's own review, as reply minus plain revision on the panel measures
  discussion    per venue, against a plain revision: share of scores changed, spread of the eight scores, rise in
                error correlation (also for the three restricted discussions), change in score correlation, growth
                of misjudgments (slope of the score change on the panel's first-review overrating, net of the plain
                revision), and weaknesses raised per review and by the whole panel
  area chairs   share of papers on which all nine LLM chairs make the wrong decision, reading the human reviews or the
                panel's reviews, and the share of those errors that accept a rejected paper
Intervals: 95%, resampling papers.

Usage:  uv run python analysis/review_cycle.py
Writes: outputs/review_cycle.json
"""
import numpy as np
import pandas as pd

from common import (DATA, MEASURES, PANEL, PRE_CHATGPT, RESULTS, VENUES, complete, conditions, decision, error_correlation,
                    folder, jsonl, manifest, matrix, measures, ratings, write)

RESTRICTED = {"discussion": "discussion", "hidden_scores": "discussion_text", "hidden_verdicts": "discussion_masked",
              "independence": "discussion_independent"}
DESIGNS = [("notes", "missed", "second_look"), ("notes_fresh", "fresh_missed", "run1"), ("roles", "roles", "round1"),
           ("hidden_scores", "discussion_text", "second_look"), ("hidden_verdicts", "discussion_masked", "second_look"),
           ("independence", "discussion_independent", "second_look"), ("discussion", "discussion", "second_look"),
           ("rebuttal", "rebuttal", "second_look")]
ci = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def human_mean(man, ids):
    return np.array([np.mean(ratings(man[p])) for p in ids])


def rebuttal(by_venue):
    out, rng = {"venues": {}, "groups": {}}, np.random.default_rng(20261011)
    for name, venue, _, group in VENUES:
        man, S = by_venue[venue]
        arms = ["round1", "second_look", "rebuttal"] + (["llm_rebuttal"] if venue == "iclr2022" else [])
        ids = complete(man, S, arms)
        X = {a: matrix(S, ids, a) for a in arms}
        out["venues"][name] = {"papers": len(ids), "group": group} | {
            a: dict(changed=float((np.abs(X[a] - X["round1"]) > 1e-9).mean()), abs_change=float(9 * np.abs(X[a] - X["round1"]).mean()))
            for a in arms[1:]}
        if venue == "iclr2022":
            h, t = human_mean(man, ids), np.array([decision(man[p]) for p in ids])
            for a in ("rebuttal", "llm_rebuttal"):
                diff = lambda i: measures(X[a][i], h[i], t[i]) - measures(X["second_look"][i], h[i], t[i])
                pt, bs = diff(np.arange(len(ids))), np.array([diff(rng.integers(0, len(ids), len(ids))) for _ in range(1000)])
                out[f"iclr2022_{a}_vs_plain"] = {k: dict(change=float(pt[j]), ci=ci(bs[:, j])) for j, k in enumerate(MEASURES)}
    for group in ("2021-22", "2023-25"):
        vs = [v for v in out["venues"].values() if v["group"] == group]
        out["groups"][group] = {a: {k: float(np.mean([v[a][k] for v in vs])) for k in ("changed", "abs_change")}
                                for a in ("second_look", "rebuttal")}
    return out


def coverage(venue, man):
    """Per review and for the whole panel: share of human-raised weaknesses raised after the plain revision and after
    discussion, on papers whose eight revisions are all labelled by the judge."""
    source = DATA / "processed/content" if venue not in PRE_CHATGPT else RESULTS / folder(venue) / "content"
    weak = {}
    for w in jsonl(source / "reference_items.jsonl"):
        if w["polarity"] == "weakness":
            weak.setdefault(w["paper_id"], set()).add(w["id"])
    lab = pd.read_csv(RESULTS / folder(venue) / "discussion" / "coverage_labels.csv")
    lab = lab[lab.model.isin(PANEL) & (lab.paper_id.isin(man) if venue not in PRE_CHATGPT else True)]
    have = lab.groupby(["paper_id", "arm"]).model.nunique().unstack()
    arms = ["second_look", "discussion"]
    keep = [p for p in have.index if all(have.loc[p].get(a, 0) == len(PANEL) for a in arms) and weak.get(p)]
    hit = lab[lab.recognition.isin(["specific", "partial"])]
    rows = []
    for p in keep:
        row = {}
        for a in arms:
            h = hit[(hit.paper_id == p) & (hit.arm == a) & hit.id.isin(weak[p])]
            row[f"panel_{a}"] = h.id.nunique() / len(weak[p])
            row[f"review_{a}"] = h.groupby("model").id.nunique().reindex(PANEL, fill_value=0).mean() / len(weak[p])
        rows.append(row)
    d = pd.DataFrame(rows)
    return dict(papers=len(d), review_coverage=[float(d.review_second_look.mean()), float(d.review_discussion.mean())],
                panel_miss=[float(1 - d.panel_second_look.mean()), float(1 - d.panel_discussion.mean())])


def misjudgment_growth(man, S, rng):
    """Slope of the change in the panel's mean score (discussion minus plain revision, 1-10 points) on how much the panel
    had overrated the paper in its first reviews; positive means misjudged papers move further the same way."""
    ids = [p for p in complete(man, S, ("round1", "second_look", "discussion"))]
    first = matrix(S, ids, "round1").mean(1)
    err = 9 * (first - human_mean(man, ids))
    change = 9 * (matrix(S, ids, "discussion").mean(1) - matrix(S, ids, "second_look").mean(1))
    slope = lambda i: float(np.polyfit(err[i], change[i], 1)[0])
    return dict(papers=len(ids), slope=slope(np.arange(len(ids))),
                ci=ci([slope(rng.integers(0, len(ids), len(ids))) for _ in range(1000)]))


def discussion(by_venue):
    out, rng_corr, rng_growth = {}, np.random.default_rng(20261006), np.random.default_rng(20261010)
    for name, venue, _, _ in VENUES:
        man, S = by_venue[venue]
        ids = sorted(complete(man, S, ("round1", "second_look", "discussion")))
        R1, SL, DI = (matrix(S, ids, a) for a in ("round1", "second_look", "discussion"))
        sd = lambda M: float(M.std(1, ddof=1).mean())
        o = dict(papers=len(ids), changed_plain_revision=float((SL != R1).mean()), changed_discussion=float((DI != R1).mean()),
                 spread_ratio=sd(DI) / sd(SL), coverage=coverage(venue, man))
        # rise in error correlation from a plain revision to full and restricted discussion
        ids_r = sorted(complete(man, S, ("round1", "second_look", *RESTRICTED.values())))
        h = human_mean(man, ids_r)
        A = {a: matrix(S, ids_r, a) for a in ("second_look", *RESTRICTED.values())}
        rise = lambda i: [error_correlation(A[a][i], h[i]) - error_correlation(A["second_look"][i], h[i]) for a in RESTRICTED.values()]
        rng = np.random.default_rng(20261010)
        pt, bs = rise(np.arange(len(ids_r))), np.array([rise(rng.integers(0, len(ids_r), len(ids_r))) for _ in range(1000)])
        o["error_corr_rise"] = {k: dict(rise=pt[j], ci=ci(bs[:, j])) for j, k in enumerate(RESTRICTED)} | {"papers": len(ids_r)}
        # change in score correlation for every design against its comparison (Tables 2 and 14 use discussion)
        o["score_corr_change"] = {}
        for design, a, b in DESIGNS:
            ids_d = complete(man, S, (a, b))
            if len(ids_d) < 20:
                continue
            Am, Bm, hd = matrix(S, ids_d, a).mean(1), matrix(S, ids_d, b).mean(1), human_mean(man, ids_d)
            f = lambda i: np.corrcoef(Am[i], hd[i])[0, 1] - np.corrcoef(Bm[i], hd[i])[0, 1]
            o["score_corr_change"][design] = dict(papers=len(ids_d), change=float(f(np.arange(len(ids_d)))),
                                                  ci=ci([f(rng_corr.integers(0, len(ids_d), len(ids_d))) for _ in range(1000)]))
        o["misjudgment_growth"] = misjudgment_growth(man, S, rng_growth)
        out[name] = o
        print(f"{name:13s} scores changed {o['changed_discussion']:.0%} ({o['changed_plain_revision']:.0%} plain), spread "
              f"x{o['spread_ratio']:.2f}, error correlation {o['error_corr_rise']['discussion']['rise']:+.2f}, score correlation "
              f"{o['score_corr_change']['discussion']['change']:+.2f}, misjudgment growth {o['misjudgment_growth']['slope']:+.2f}", flush=True)
    return out


def area_chairs():
    """Unanimous errors of the nine LLM chairs (decided papers on which every chair decided under both readings)."""
    out, data, rng = {}, {}, np.random.default_rng(20261011)
    for name, venue, d, _ in VENUES:
        recs = [r for r in jsonl(RESULTS / d / "area_chair" / "records.jsonl") if r.get("parse_ok") and r.get("truth") in ("accept", "reject")]
        latest = {(r["paper_id"], r["arm"], r["ac_model_id"]): r for r in recs}
        arms, chairs = sorted({k[1] for k in latest}), sorted({k[2] for k in latest})
        papers = sorted({k[0] for k in latest})
        keep = [p for p in papers if all((p, a, c) in latest for a in arms for c in chairs)
                and latest[(p, "H0", chairs[0])].get("venue_key", venue) == venue]
        wrong = lambda a: np.array([all(latest[(p, a, c)]["decision"] != latest[(p, a, c)]["truth"] for c in chairs) for p in keep], float)
        false_accept = np.array([latest[(p, "P0", chairs[0])]["truth"] == "reject" for p in keep])
        h, pw = wrong("H0"), wrong("P0")
        data[name] = (h, pw, false_accept)
        n = len(keep)
        bs = [pw[i].mean() - h[i].mean() for i in (rng.integers(0, n, n) for _ in range(2000))]
        out[name] = dict(papers=n, chairs=len(chairs), human_reviews=float(h.mean()), panel_reviews=float(pw.mean()),
                         difference_ci=ci(bs), panel_false_accept_share=float((pw * false_accept).sum() / pw.sum()))
    N = sum(len(h) for h, _, _ in data.values())
    boots = [sum(len(h) * (p[i].mean() - h[i].mean()) for h, p, _ in data.values() for i in [rng.integers(0, len(h), len(h))]) / N
             for _ in range(2000)]
    out["Pooled"] = dict(papers=N, human_reviews=sum(h.sum() for h, _, _ in data.values()) / N,
                         panel_reviews=sum(p.sum() for _, p, _ in data.values()) / N, difference_ci=ci(boots),
                         panel_false_accept_share=sum((p * f).sum() for _, p, f in data.values()) / sum(p.sum() for _, p, _ in data.values()))
    for k, v in out.items():
        print(f"{k:13s} all nine chairs wrong: human reviews {v['human_reviews']:.1%}, panel reviews {v['panel_reviews']:.1%} "
              f"({v['papers']} papers; {v['panel_false_accept_share']:.0%} of panel-review errors accept a rejected paper)")
    return out


def main():
    by_venue = {venue: (manifest(venue), conditions(venue)) for _, venue, _, _ in VENUES}
    write("review_cycle.json", dict(rebuttal=rebuttal(by_venue), discussion=discussion(by_venue), area_chairs=area_chairs()))


if __name__ == "__main__":
    main()
