---
id: call_a_discussion_text
version: v1.0.0
purpose: Discussion with hidden scores. As call_a_discussion, but the other reviews are shown without their scores.
notes: The hidden-verdicts condition uses this prompt with verdict phrases also masked in the other reviews (experiments/verdict_mask.py).
---

{own_review_prompt}

---

## SECOND-ROUND ADDENDUM

You are the original reviewer. This is the second round of the review process: you can now
read the other reviewers' initial reviews of the same paper, shown below without their scores.
Reconsider your complete initial review in light of them. You are not required to change your review
or your score; change them only where, on reflection, your assessment of the paper differs.

Return one complete replacement review using exactly the rubric, calibration guidance,
public-reasoning rule, rating scale, and JSON schema specified above. Base the update only on
the submitted paper, your original review, and the other reviewers' reviews.

## YOUR ORIGINAL COMPLETE REVIEW

{source_review}

## YOUR INITIAL SCORE

{initial_score}

## THE OTHER REVIEWERS' INITIAL REVIEWS (SCORES WITHHELD)

{peer_reviews}
