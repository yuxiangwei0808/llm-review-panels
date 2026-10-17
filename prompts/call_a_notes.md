---
id: call_a_notes
version: v1.0.0
purpose: Notes. The reviewer sees the paper, its own first review, and three notes from "another reader", then writes
  a replacement review. The notes are weaknesses the panel missed, weaknesses some reviewer raised, or another paper's
  weaknesses; the framing is the same in every condition.
notes: Identical to call_a_second_look except for the notes and one sentence introducing them. The notes are presented
  as possibly inaccurate, so the reviewer has to assess them.
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
