---
id: extract_points
version: v1.2.0
purpose: Weakness extraction. Turns a paper's human reviews into the distinct points the reviewers raised, merged
  across reviewers; each point is classified (polarity, category, kind) and tied to quotes from the reviews. The
  coverage judge (match_points) checks reviews against these points.
notes: One point per distinct concern; JSON output.
---

You are annotating the human peer reviews of a machine-learning paper for a content analysis.

Below are the abstract and the reviews (summary, strengths, weaknesses per reviewer; ratings
removed). Extract every distinct POINT the reviewers raise about the paper and return them as one
JSON object.

WHAT COUNTS AS A POINT
- A strength, weakness, or neutral question about the paper: a claim about its novelty,
  soundness, experiments, significance, reproducibility, related work, or clarity, or a question
  the reviewer asks the authors.
- BE EXHAUSTIVE: every distinct concern, praise, or question a reviewer states is a point,
  including minor ones (a missing figure label, an unreported hyperparameter, a missing
  baseline). Do not drop a point because it is small or phrased as a suggestion ("the authors
  should report X" is a weakness: X is not reported).
- ONE POINT PER DISTINCT CONCERN. A paragraph that raises three different concerns yields three
  points. But do not split one concern into several points: a reviewer who lists several missing
  baselines in one breath makes one point (missing baselines: A, B, C) unless they discuss an
  item separately; a concern and the evidence given for it are one point; a concern and the
  suggestion that would fix it are one point.
- Work through the reviews one at a time and extract each reviewer's points before merging.
- MERGE only when two reviewers make the SAME specific claim about the SAME object; then keep
  one entry with several sources. Related concerns about the same section are separate points.
- Skip only: reviewer summaries of what the paper does, procedural remarks, and pure
  restatements of the paper's own claims that carry no evaluation.
- A typical review yields 4 to 9 points; a typical paper 18 to 32, of which about half are
  weaknesses.

FIELDS PER POINT
- id: "C001", "C002", ... in the order extracted.
- polarity: "strength" | "weakness" | "neutral" (a question or request that is neither praise
  nor criticism).
- category: "soundness" (the method or theory itself: design choices and their justification,
  assumptions, correctness of proofs and claims, whether conclusions follow from the evidence,
  overclaiming, validity threats) | "experiments" (what was or was not run: baselines, datasets,
  metrics, ablations, scale, statistics, hyperparameter search) | "novelty" (originality vs prior
  work) | "significance" (importance, impact, practical relevance, generality of the problem or
  result) | "reproducibility" (code, data, or details needed to reproduce, including unreported
  settings) | "related work" (missing or mischaracterized citations) | "clarity" (writing,
  organization, figures, notation, definitions) | "other". When a concern is about whether the
  evidence supports a claim, use soundness; when it is about a missing experiment, use
  experiments.
- kind: "substantive" (bears on the scientific contribution) | "presentation" (how it is written
  or shown, including missing details in the text) | "neutral_question".
- assessability: "supplied_material" (can be checked from the paper and its supplement) |
  "outside_information" (needs knowledge beyond the submission, e.g. other papers or
  unreported experiments) | "uncertain".
- availability_reason: one sentence saying why the point is or is not checkable from the paper.
- text: one sentence stating the point in neutral, specific terms (name the aspect: which
  claim, experiment, baseline, section, or omission).
- sources: list of objects {"review": "<review id>", "quote": "<verbatim excerpt>"}, one per
  reviewer statement that makes the point; quotes must be copied exactly from the review text.
  Review ids are given in the review headings below (H00, H01, ...).

OUTPUT
Return exactly one JSON object and nothing else:
{"points": [{"id": "C001", "polarity": "...", "category": "...", "kind": "...",
             "assessability": "...", "availability_reason": "...", "text": "...",
             "sources": [{"review": "H00", "quote": "..."}]}, ...]}

---

## ABSTRACT

{abstract}

## REVIEWS

{reviews}
