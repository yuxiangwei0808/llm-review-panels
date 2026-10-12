---
id: call_a_role
version: v1.0.0
block: [panel_composition]
purpose: >
  M7 remedy arm ("role-diverse panel"). A fresh call-A review (the frozen call_a prompt, unchanged)
  with one added section assigning the reviewer a focus role on the panel: soundness of the claims,
  experiments and evaluation, clarity and reproducibility, or novelty and related work. Each paper's
  eight reviewers get the four roles twice each, rotated across models by a seeded per-paper
  assignment. The reviewer still reviews the whole paper and gives the usual holistic score.
design_notes: >
  The role text asks for extra attention to one aspect, not for a partial review, so the score stays
  a holistic recommendation comparable with call A. Written before any M7 call.
changelog: >
  v1.0.0 (2026-09-30): first version.
---

{own_review_prompt}

---

## YOUR ROLE ON THIS PANEL

The area chair has asked each reviewer on this panel to take responsibility for one aspect of the
submission, so that together the panel covers the paper thoroughly. Your aspect is:

**{role_name}.** {role_description}

Review the whole paper and give your overall recommendation as usual, but examine your aspect with
particular care: check it against the paper yourself, and make sure every real weakness you find in
it appears in your weaknesses. Use exactly the rubric, calibration guidance, public-reasoning rule,
rating scale, and JSON schema specified above.
