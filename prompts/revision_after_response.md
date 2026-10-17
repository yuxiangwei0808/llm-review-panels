---
id: revision_after_response
version: v2.0.0
purpose: Revision after an author response. The first-review prompt, the reviewer's first review, and one author
  response (the authors' real rebuttal, or a reply written to this review); the reviewer writes a replacement review.
---

{baseline_prompt_prefix}{paper_text}{baseline_prompt_suffix}

---

## ONE-ROUND REVIEWER-UPDATE ADDENDUM

You are the original reviewer. Reconsider your complete initial review after reading the
author response below. Assess the response critically: it may resolve, weaken, leave
unresolved, or reinforce an issue, and you are not required to change your score.

Return one complete replacement review using exactly the rubric, calibration guidance,
public-reasoning rule, rating scale, and JSON schema specified above. Make response effects
explicit within relevant strengths and weaknesses so every updated item is self-contained.
Base the update only on the submitted paper, your original review, and this response. Do
not use or infer the human final score, final outcome, later discussion, or another
reviewer's judgment.

## YOUR ORIGINAL COMPLETE REVIEW

{source_review}

## YOUR INITIAL SCORE

{initial_score}

## AUTHOR RESPONSE

{author_response}
