---
id: call_a_notes
version: v1.0.0
block: [panel_interaction]
purpose: >
  E8 injection arms. The reviewer sees the paper, its own round-1 review, and three notes from
  "another reader", then writes a replacement review. The notes are missed human-raised
  weaknesses, already-mentioned ones, another paper's weaknesses, or neutral summary sentences;
  the framing is identical in every arm.
design_notes: >
  Identical to call_a_second_look except for the notes block and one sentence naming it. The
  notes are presented as possibly inaccurate so that the reviewer has to assess them.
changelog: >
  v1.0.0 (2026-09-30): first version.
---

{own_review_prompt}

---

## SECOND-ROUND ADDENDUM

You are the original reviewer. This is the second round of the review process. Another reader of
this submission left the following notes. They may or may not be accurate; assess them against
the paper yourself. Reconsider your complete initial review. You are not required to change your
review or your score; change them only where, on reflection, your assessment of the paper differs.

Return one complete replacement review using exactly the rubric, calibration guidance,
public-reasoning rule, rating scale, and JSON schema specified above. Base the update only on the
submitted paper, your original review, and your own assessment of the notes.

## YOUR ORIGINAL COMPLETE REVIEW

{source_review}

## YOUR INITIAL SCORE

{initial_score}

## NOTES FROM ANOTHER READER

{notes}
