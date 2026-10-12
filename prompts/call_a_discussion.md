---
id: call_a_discussion
version: v1.0.0
block: [panel_interaction]
purpose: >
  E14 sharing arm ("share reviews"). The reviewer sees the paper, its own round-1 review, and
  the round-1 reviews of the other panel members (text and score, anonymized, shuffled), then
  writes a replacement review. The minimal reviewer-to-reviewer interaction stage of paper 1.
design_notes: >
  Identical to call_a_second_look except for the peer block and one sentence naming it. Peers
  are labelled "Reviewer 1..k" in a seeded order per paper and reviewer; model identities are
  not shown. Scores are shown because real review panels see each other's ratings.
changelog: >
  v1.0.0 (2026-09-30): first version.
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
