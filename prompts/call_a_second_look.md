---
id: call_a_second_look
version: v1.0.0
block: [panel_interaction]
purpose: >
  E14 control arm ("share nothing"). The reviewer sees the paper and its own round-1 review
  again and writes a replacement review, with no information from other reviewers. Separates
  the effect of a second look (re-reading, re-sampling) from the effect of seeing peers.
design_notes: >
  Mirrors call_a_discussion word for word except for the peer block, so the two arms differ
  only in what is shared. Built on the full call_a prompt so the rubric, rating scale,
  public-reasoning rule and JSON schema are unchanged and parse_own_review reads the output.
changelog: >
  v1.0.0 (2026-09-30): first version.
---

{own_review_prompt}

---

## SECOND-ROUND ADDENDUM

You are the original reviewer. This is the second round of the review process. Reconsider
your complete initial review below. You are not required to change your review or your
score; change them only where, on reflection, your assessment of the paper differs.

Return one complete replacement review using exactly the rubric, calibration guidance,
public-reasoning rule, rating scale, and JSON schema specified above. Base the update only on
the submitted paper and your original review.

## YOUR ORIGINAL COMPLETE REVIEW

{source_review}

## YOUR INITIAL SCORE

{initial_score}
