---
id: call_a_revision
version: v2.0.0
block: [rebuttal]
purpose: >
  R6 one-round reviewer update. The complete rendered P0 Call-A prompt is frozen and
  reused verbatim, then the reviewer's sanitized P0 output and one real or synthetic
  author response are appended.
design_notes: >
  The model is asked for one complete replacement review, not a delta. The response
  condition and every frozen upstream source are bound into the dependency hash.
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
