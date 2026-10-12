---
id: call_f_itemized
version: v2.0.0
block: [B2b]
sources:
  severity_taxonomy: >
    Identical to call_b: base taxonomy from PRISM (arXiv 2605.26730), extended from their
    two levels (Critical / Minor, weights 2:1) to four. See `call_b.md` for the full
    justification. The vocabulary MUST stay identical to call_b's, because the constraint
    effect endpoint compares this cell against call B and any wording difference would be
    confounded with the effect being measured.
changelog: >
  v1.1.0 (2026-07-25): opening line aligned word-for-word with call B ("raised about one
  ICLR 2025 submission" rather than "raised by the assigned reviewers of"). The constraint
  effect endpoint compares this cell against call B, so the two prompts must differ only in
  where the items came from, never in how the rating task is described.
  v1.2.1 (2026-07-26): explicitly prohibit duplicate item IDs after one sampled cell
  repeated the same item with conflicting labels across three retries.

  v2.0.0 (2026-07-29): require the same concise, rating-hidden public justification used
  by call B for every item. This makes the fixed-evidence item ratings sufficient for the
  later G=20 verifier confidence pass while retaining self-reported confidence only for
  historical compatibility.
purpose: >
  Fixed-review column, itemized form. Every model rates the identical human-stated items
  on the shared severity vocabulary; the same fixed aggregation produces the score. Each
  rating carries a public, label-hidden justification for the later fixed verifier.
design_notes: >
  This cell needs no cross-model item alignment, because the items are given by the human
  reviews rather than generated per model. That is what makes B2b the cheap, clean home
  for H5: the standards gap here cannot be contaminated by differences in what each model
  chose to raise.
  As in call B, no overall score is requested. The rating vocabulary is identical to call
  B's so that the constraint effect (does itemizing shrink the standards gap?) is measured
  on the same instrument in both columns.
  The per-item `justification` is public evidence-grounded reasoning, not private
  chain-of-thought. It must not disclose the severity label or self-reported confidence;
  both are withheld from the later fixed verifier.
---

Below are the strengths and weaknesses raised about one {venue_name} submission. They are
listed in arbitrary order, anonymized, with all numeric ratings removed.

Rate how much each point matters for the paper's fate. Judge each item on its own terms,
against the standard you would apply as a {venue_name} reviewer.

## Rating vocabulary

For **weaknesses**:

| Label | Meaning |
|---|---|
| `minor` | Worth fixing, but does not affect whether the work stands |
| `moderate` | A real shortcoming that a reader should weigh against the contribution |
| `major` | Substantially undermines the contribution as claimed |
| `fatal` | Invalidates a central claim; the paper cannot stand as written |

For **strengths**:

| Label | Meaning |
|---|---|
| `minor` | A nice property, but not a reason to accept |
| `moderate` | A genuine contribution that counts in the paper's favour |
| `major` | A substantial contribution that would carry real weight |
| `decisive` | On its own a compelling reason to accept |

For every item, first provide a concise public justification grounded only in the stated
point and paper context. Explain what makes the point more or less consequential without
using any of the exact rating-vocabulary words `minor`, `moderate`, `major`, `fatal`, or
`decisive`, and without stating the severity label you selected or a numeric confidence
value. This justification will be shown to a separate verifier with your selected label
and self-reported confidence hidden.

Also give a confidence between 0.00 and 1.00 for each rating for historical compatibility:
how sure you are that this is the right label for this item, given the evidence stated.
This self-reported value is stored but is not the study's verifier-based confidence.

## Constraints

- Rate **every** item listed, using its exact `item_id`.
- Return **exactly one** rating object for each `item_id`; never repeat an `item_id`.
- Do **not** assign an overall score or recommendation for the paper.
- Do **not** introduce new criticisms or strengths, and do not speculate about paper
  content that is not stated in the items.
- Give every item a nonempty `justification`.
- Do **not** disclose the selected severity or numeric confidence inside `justification`.

## Output format

Return **only** a JSON object matching this structure, with no surrounding prose or code
fences:

```json
{
  "ratings": [
    {
      "item_id": "<exact id from the list>",
      "justification": "<concise public rationale without the severity label or confidence>",
      "severity": "<label>",
      "confidence": <0.00-1.00>
    }
  ]
}
```

---

## PAPER CONTEXT

{paper_context}

## ITEMS

{items_block}
