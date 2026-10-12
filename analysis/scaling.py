"""Figure 1 and Table 1: how much a panel gains from more LLMs, compared with a human committee gaining reviewers.

For each venue, and pooled over each venue group (Figure 1), the held-out correlation of LLM panels of 1-8 models and
human committees of 1-3 reviewers (common.held_out). Table 1 reports the gain from one to three humans and from one to
eight LLMs and the LLM gain as a share of the human gain; Figure 1 marks the number of humans whose committee matches
eight LLMs ("human equivalent").

Usage:  uv run python analysis/scaling.py
Writes: outputs/scaling.json
"""
import pandas as pd

from common import VENUES, held_out, manifest, score_table, write


def main():
    out, groups = {}, {}
    for name, venue, _, group in VENUES:
        man, L = manifest(venue), score_table(venue)
        out[name] = r = held_out(man, L)
        g = groups.setdefault(group, ({}, []))
        g[0].update(man)
        g[1].append(L)
        ratio = r["gain_llm_1_to_8"][0] / r["gain_human_1_to_3"][0]
        print(f"{name:13s} n={r['papers']}  +2 humans {r['gain_human_1_to_3'][0]:+.2f}  +7 LLMs {r['gain_llm_1_to_8'][0]:+.2f}"
              f"  LLMs vs. humans {ratio:.0%}  difference CI [{r['gain_difference'][1]:+.3f}, {r['gain_difference'][2]:+.3f}]")
    # Figure 1 pools each group's papers in the order the paper used (2021-22: ICLR 2022, ICLR 2021, NeurIPS 2022;
    # 2023-25: sorted by paper id). The order only decides which held-out reviewer each random draw picks.
    order = {"2021-22": [1, 0, 2], "2023-25": [0, 1, 2]}
    for group, (man, Ls) in groups.items():
        L = pd.concat([Ls[i] for i in order[group]])
        L = L[~L.index.duplicated()]
        out[f"{group} pooled"] = r = held_out(man, L.sort_index() if group == "2023-25" else L)
        print(f"{group} pooled n={r['papers']}  1 LLM {r['llm'][1][0]:.2f}  8 LLMs {r['llm'][8][0]:.2f}  "
              f"humans {[round(r['human'][k][0], 2) for k in (1, 2, 3)]}  eight LLMs = {r['human_equivalent']:.1f} humans")
    write("scaling.json", out)


if __name__ == "__main__":
    main()
