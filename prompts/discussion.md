---
id: discussion
version: v1.0.0
purpose: Discussion. The reviewer sees the paper, its own first review, and the other panel members' first reviews
  with their scores (anonymized as Reviewer 1..k, in a seeded order), then writes a replacement review.
notes: Identical to plain_revision except for the block of other reviews and one sentence introducing it.
---

{own_review_prompt}

---

## SECOND-ROUND ADDENDUM

You are the original reviewer. This is the second round of the review process: you can now
read the other reviewers' initial reviews of the same paper, shown below. Reconsider your
complete initial review in light of them. You are not required to change your review or your
score; change them only where, on reflection, your assessment of the paper differs.

Return one complete replacement review using exactly the rubric, calibration guidance,
public-reasoning rule, rating scale, and JSON schema specified above. Base the update only on
the submitted paper, your original review, and the other reviewers' reviews.

## YOUR ORIGINAL COMPLETE REVIEW

{source_review}

## YOUR INITIAL SCORE

{initial_score}

## THE OTHER REVIEWERS' INITIAL REVIEWS

{peer_reviews}
