---
id: ac_binary
version: v1.0.0
block: [llm_as_ac]
purpose: >
  Binary area-chair decision from a frozen paper and one anonymous evidence packet.
design_notes: >
  The prompt contains no author identity, forum identifier, source-model identity,
  meta-review, recorded outcome, or post-decision note. Only paper OCR may be shortened,
  and only when the exact model context requires it.
---

Act as an area chair for {venue_name}. Make an independent binary decision using only
the submitted paper and the review record supplied below. Do not browse, use external
sources, identify the paper or authors, or infer the recorded conference decision.

Apply the venue's review standard. Reviewer recommendations are evidence, not votes; weigh
the substantive arguments, their grounding in the paper, and any author response or public
discussion that is present.

The official reviewer recommendation scale is:
{rating_scale_block}

The venue's nominal accept-side threshold on that scale is {accept_threshold}. Your task,
however, is to issue one binary `accept` or `reject` decision after synthesizing the whole
record.

## ANONYMIZED PAPER OCR

{paper_text}

## REVIEW RECORD

{evidence_packet}

## REQUIRED OUTPUT

Give a concise public meta-review that identifies the decisive strengths, weaknesses,
response effects (when present), and material uncertainty. Then give exactly one binary
decision. Return exactly these two tagged blocks and no additional block:

<meta_review>
Concise public meta-review.
</meta_review>
<decision>accept</decision>

The decision tag must contain exactly `accept` or `reject`. Mentioning the likely decision
inside the public meta-review is allowed; it is recorded as an audit flag and is not a
format failure.
