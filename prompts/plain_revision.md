---
id: plain_revision
version: v1.0.0
purpose: Plain revision (control). The reviewer sees the paper and its own first review again and writes a replacement
  review, with nothing from other reviewers.
notes: Matches discussion word for word except for the block of other reviews.
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
