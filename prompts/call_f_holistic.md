---
id: call_f_holistic
version: v2.0.0
block: [B2b]
purpose: >
  Fixed-review column, holistic form. Every model scores the identical set of human-written
  observations, so any between-model disagreement is judging rather than reading.
sources:
  role: >
    Reviewer, per the ICLR 2025 Reviewer Guide. NOT an Area Chair: the ICLR 2025 AC Guide
    specifies that an AC writes a meta-review of at least 60 words ending in a
    recommendation, and never assigns a rating on the 1/3/5/6/8/10 scale. That scale
    belongs to reviewers.
  rating_scale: >
    Official ICLR / OpenReview review-form wording.
design_notes: >
  Framing the dossier as the reviewer's own recorded observations keeps the fiction minimal
  and matches the estimand: this call isolates Layer 2 (evidence -> score), so the model
  must be in the role that actually produces a rating from evidence.
  Two things follow from using the reviewer scale here. Call A and call F emit into the
  SAME output space, which the evidence gap requires since it differences their AUCs; and
  this call becomes structurally parallel to call B, both being pure Layer-2 tasks over
  given evidence.
  The flat `KEY: value` output (rather than JSON) puts the numeric score in a position that
  is trivial and unambiguous to parse. The corrected panel is sampled-only, so a numeric
  score preserves every venue's exact native scale, including scales with more than six
  levels, without sacrificing a current logprob endpoint.
  Dossiers arrive with numeric ratings stripped by the leakage regex and reviewer order
  shuffled; the leakage audit reports the regex's recall.
  The `<reasoning>` block is a concise public evidence-grounded justification, not private
  chain-of-thought. The exact rating and model-stated acceptance probability remain in the
  flat output but are withheld from the later fixed verifier.
changelog: >
  v2.0.0 (2026-07-29): require an explicit public `<reasoning>` block that weighs only the
  fixed human evidence without disclosing the exact rating or acceptance probability. This
  artifact supports the later G=20 verifier confidence pass.

  v1.3.0 (2026-07-28): replaced the six-letter encoding with the venue's native numeric
  scale. NeurIPS 2023 has ten score levels and cannot be represented by A-F; the corrected
  study is sampled-only, so the old single-token logprob motivation no longer applies.

  v1.2.0 (2026-07-27): venue-parameterised. The reviewer role, scale and letter codes were
  hardcoded to ICLR 2025 while call A and call B already read them from the venue registry,
  so a non-ICLR-2025 collection scored papers against the wrong conference's scale.

  v1.1.0 (2026-07-25): replaced the pilot's "you are an ICLR 2025 Area Chair assigning a
  final numeric rating" framing, which describes a role that does not exist at ICLR, with
  the reviewer role that owns this scale. Deviates from the executed n=6 pilot; the pilot
  is a noise-floor prior, not a result being extended, so role validity was judged worth
  more than exact comparability.
---

You are a reviewer for {venue_name}.

You have read the following submission and recorded the observations below: its abstract,
followed by the summary, strengths and weaknesses you noted. They are listed in arbitrary
order.

TASK: assign an overall rating on the official {venue_name} scale:
{rating_scale_block}

HARD CONSTRAINTS:
- Base your rating ONLY on the strengths and weaknesses stated below.
- Do NOT introduce any new criticism, concern, or strength not already present.
- Do NOT speculate about content that is not stated.

Carefully weigh the stated points, then give your rating.

Before the three scoring lines, provide a public evidence-grounded justification in
exactly one block with literal opening and closing tags:

<reasoning>
Weigh the most consequential stated strengths and weaknesses under the venue's review
standard and identify material uncertainty in the supplied evidence. Base this explanation
only on the observations below. Do not state the exact numeric rating, `ACCEPT_PROB`, or a
numeric acceptance probability. This is a concise public justification, not private
chain-of-thought.
</reasoning>

After the closing `</reasoning>` tag, output EXACTLY these three lines:
DECISIVE_FACTOR: <one short clause naming the single stated point that most drove your rating>
ACCEPT_PROB: <0.00-1.00, your estimated probability {venue_short} accepts this paper>
RATING: X

where X is exactly one of {allowed_scores}.

---

## OBSERVATIONS

{dossier}
