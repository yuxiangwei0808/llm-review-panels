---
id: match_points
version: v1.1.0
purpose: Coverage judge. Given a paper's human-raised points and one review, labels each point as raised specifically,
  partially, or not at all (absent), and the review's stance toward it.
notes: 'Output: one "ID: recognition, stance" line per point.'
---

You are annotating peer reviews of a machine-learning paper for a content analysis.

Below are (1) the distinct POINTS raised by the paper's human reviewers, each with an ID and a
polarity (strength, weakness, or neutral question), and (2) one REVIEW of the same paper written
by another reviewer. For every human point, decide whether the review raises the same point.

RECOGNITION (choose exactly one per point):
- specific: the review makes the SAME claim about the SAME object (the same experiment, baseline,
  assumption, theorem, dataset, section, or omission), even if worded differently or in less
  detail. Both the object and the concern must match.
- partial: the review's claim clearly overlaps the point but is narrower or broader: it names the
  same object and the same concern but only a subset or a generalization of it (e.g., the point
  lists three missing baselines and the review names one of them; the point says the evaluation
  uses too few datasets and the review says the evaluation is narrow). Use this sparingly.
- absent: everything else. In particular, absent when the review discusses the same object with a
  DIFFERENT concern, raises a related but distinct issue, or mentions the topic only in a
  summary of what the paper does.

STANCE (only when recognition is specific or partial; otherwise not_applicable):
- endorses: the review takes the same view (raises the same weakness as a weakness, praises the
  same strength, asks the same question).
- disputes: the review explicitly takes the opposite view (e.g., calls adequate what the human
  point calls lacking, or criticizes what the human point praises).
- mixed: the review both agrees and disagrees with the point.
- unclear: the review mentions the aspect but its position cannot be determined.

CALIBRATION: a typical model review recognizes only about two in five human points; the rest are
absent. Most human points are about a specific detail that the review simply does not raise.
Do not credit the review for a point because it is critical of the same general area (e.g., "the
experiments are limited") unless the point itself is that general. When torn between partial
and absent, choose absent. When torn between specific and partial, ask whether a reader of the
review alone would learn this exact point; if not, partial.

Examples of ABSENT despite a related concern:
- point: "does not analyze other test-time compute mechanisms such as beam search";
  review: "evaluates a limited set of verification strategies" -> absent (different mechanism).
- point: "evaluation is restricted to sentiment analysis and 1,000 ImageNet images";
  review: "evaluation focuses on fidelity metrics with no user study" -> absent.
- point: "no mathematical statement supports the implicit-regularization claims";
  review: "the theory is deterministic and continuous-time" -> absent.

RULES:
- Judge only from the review text below; do not use knowledge of the paper.
- Label every point exactly once. Do not add, drop, or rename IDs.

Output EXACTLY one line per point, in the order listed, and nothing else:
ID: recognition, stance

for example (illustrative IDs):
C001: specific, endorses
C002: absent, not_applicable

---

## HUMAN POINTS

{points}

## REVIEW

{review}
