---
id: call_a
version: v3.0.0
block: [B2, calib]
purpose: >
  Deployed practice. The model reads the submission, writes an itemized review, and
  assigns a holistic overall score in the same pass. It also preserves a concise public,
  rating-hidden rationale for the later fixed-verifier confidence stage. Yields S_hol.
sources:
  body: >
    Derived from LLM-as-a-Reviewer (arXiv 2605.25415) Appendix H.1 (shared review prompt)
    and H.4 (ICLR 2025 venue instruction), with the enumerated deviations below. That
    study covers 12 models and 898 papers including ICLR 2025, so Layer-1 output remains
    closely comparable while explicitly accounting for our OCR input path.
  rating_scale: >
    Official ICLR / OpenReview review-form wording, REPLACING Appendix H.4's paraphrase.
    See deviation 2.
deviations:
  1: >
    "the academic paper below" replaces "the uploaded academic paper": we pass extracted
    text, not a PDF upload.
  2: >
    RATING SCALE. Appendix H.4 paraphrases the scale as "6: Accept (threshold for
    acceptance) / 5: Borderline". We use the official OpenReview wording ("marginally
    above/below the acceptance threshold"). This is deliberate and load-bearing: we predict
    decisions made by humans reading the official form, and our aggregation places the
    accept cut precisely between 5 and 6, so "Borderline" would blur the one landmark the
    instrument depends on. It also preserves comparability with the executed pilot.
  3: >
    The paper title is masked upstream as a contamination control (recognizing a title is a
    route to recalling the recorded outcome, which is what we predict). Appendix H does not
    mask.
  4: >
    `accept_prob` is NOT requested, because Appendix H does not request it. Consequence:
    models or providers without response logprobs have no direct accept-side probability
    on call A, so the routing-screen endpoint depends on verbalized mode for them. Restore
    by adding one field if that trade is unwanted.
  5: >
    OCR NOTICE. The input is Nemotron OCR Markdown rather than the native PDF. A short
    instruction tells reviewers not to treat conversion corruption as a flaw in the
    manuscript. The nine-venue pilot made this necessary:
    105/216 Call-A responses initially included at least one weakness about OCR/formatting
    artifacts. Those are not valid judgments of the underlying paper. The runner supplies
    the complete frozen OCR by default and adds an explicit marker only if a particular
    model's context limit requires a shorter payload.
  6: >
    ITEM COUNT. Appendix H asks for 3-5 strengths and 3-5 weaknesses. We remove that
    fixed count so the prompt does not suppress distinct substantive findings or force
    padding to a quota. This is a documented adaptation, not a verbatim reproduction of
    the source prompt; the JSON schema and dimension labels are unchanged.
design_notes: >
  The score instruction is the standard holistic one -- this call must reproduce what a
  user actually gets, so it must NOT nudge the model to derive its number from its items.
  The itemized reasoning path is measured separately in call B.
  The `<reasoning>` block is a public evidence-grounded justification, not private
  chain-of-thought. The exact score and self-reported confidence remain in the JSON but
  are withheld from the later fixed verifier.
  Two fields we did not have before and should exploit: `prior_knowledge.seen_before` is a
  per-paper contamination check, recorded as a covariate; and the `[Dimension -
  Subcriterion]` prefix gives every item a free categorical label for B1's coverage
  analysis.
changelog: >
  v3.0.0 (2026-07-29): promoted the explicit public `<reasoning>` artifact from the
  completed pilot into the primary Call A prompt. The rationale must not disclose the
  exact venue score or 1-5 self-reported confidence; those remain in the JSON rating
  object for compatibility.

  v2.4.0 (2026-07-28): removed Appendix H's 3-5 strength/weakness count. Call A remains
  based on the published prompt but is explicitly described as adapted rather than
  verbatim. Models must include the distinct substantive points needed for a thorough
  review without padding or merging unrelated issues to meet a quota.

  v2.3.0 (2026-07-28): removed the global 50,000-character cutoff. The complete
  frozen OCR is supplied by default; any model-specific context overflow is marked
  explicitly in the effective paper text and recorded in call provenance.

  v2.2.0 (2026-07-28): expanded the input notice to cover the deterministic
  50,000-character truncation, absent appendices/supplements, and corrupted artifact URLs;
  added a final instruction to remove any item based on conversion or input omission.

  v2.1.0 (2026-07-28): added the OCR-input notice after the nine-venue pilot showed that
  105/216 generated reviews treated extraction corruption as a manuscript weakness.

  v2.0.0 (2026-07-25): replaced our own drafted body with Appendix H verbatim. Schema
  changed -- items are now flat strings with a bracketed dimension prefix rather than
  {text, evidence} objects, and the score moved to `rating.overall_score`. Drops the
  per-item evidence location (B1 verification must locate evidence itself).
  v1.1.0: removed the invented item cap in favour of Appendix H's 3-5 guidance.
---

You are an expert AI/ML researcher and reviewer for top-tier academic conferences. You provide thorough, critical, and constructive reviews of research papers based on established evaluation criteria.

You are a reviewer for {venue_name}. Please review the academic paper below.

INPUT NOTE: The paper is an OCR-derived Markdown transcription of the submitted PDF. The
complete frozen OCR is supplied unless an explicit `CONTEXT-LIMIT NOTE` appears at the end
of the paper text. OCR can corrupt equations, line wraps, URLs, citations, figures, tables,
or glyphs even when they are correct in the PDF. Treat conversion corruption as
unobservable, not as evidence that the paper presents the material badly. If a
`CONTEXT-LIMIT NOTE` appears, also do not infer that material beyond that point was absent
from the submission. Do not list typography, formatting, extraction corruption,
inaccessible artifact links, or unavailable supplements as a strength or weakness.
Evaluate only substantive content that is actually assessable. Before returning the JSON,
remove any strength or weakness whose basis is conversion corruption or context omission.

Evaluate the paper across these core dimensions, considering the relevant subcriteria:

1. Novelty

- Originality of the problem

- Originality of method

- Insightfulness of contributions

- Positioning of novelty vs. prior work

- Others (if not listed above, and please clarify)

2. Technical Quality

- Soundness of theoretical claims

- Robustness of study design

- Appropriateness of evaluations

- Validity of conclusions

- Others (if not listed above, and please clarify)

3. Significance

- Importance of the problem

- Strength of empirical/theoretical impact

- Potential for future research or generalizability

- Others (if not listed above, and please clarify)

4. Clarity

- Quality of writing and structure

- Adequacy of related work

- Interpretability of figures/tables

- Logical flow of arguments

- Others (if not listed above, and please clarify)

5. Reproducibility

- Completeness of experimental/methodological details

- Availability of code/data/artifacts

- Transparency in methods and limitations

- Ease of reproduction

- Others (if not listed above, and please clarify)

6. Others

- if the paper has notable aspects not covered above, and please clarify.

{rating_scale_block}

IMPORTANT: You MUST select one value from this discrete set only: {allowed_scores}. Do NOT use any other numbers.

Before returning the review JSON, provide a public, evidence-grounded justification in
exactly one block with literal opening and closing tags:

<reasoning>
Weigh the most important strengths and weaknesses, explain how they interact under the
venue's review standard, and identify material uncertainty or missing evidence. Do not
state the numeric/letter rating, a phrase such as "score X" or "rating X", or the 1-5
confidence value in this block. Qualitative language such as "leans toward acceptance" is
allowed. This is a concise public justification, not private chain-of-thought.
</reasoning>

After the closing `</reasoning>` tag, return the review as valid JSON in this exact format:

{

"prior_knowledge": {

"seen_before": "Yes|No",

"explanation": "Brief explanation if you have seen this paper or know official reviews"

},

"summary": "A single paragraph (3-5 sentences) summarizing the paper's main contribution, approach, and key results",

"strengths": [

"[Dimension - Subcriterion] Specific strength with details",

"[Dimension - Subcriterion] Another specific strength"

],

"weaknesses": [

"[Dimension - Subcriterion] Specific weakness with constructive suggestions",

"[Dimension - Subcriterion] Another specific weakness"

],

"rating": {

"overall_score": " (based on the rating scale for this venue)",

"confidence": " (an integer 1-5)"

}

}

Important Guidelines:

- Identify the most relevant Level 1 dimension and Level 2 subcriterion for each point.

- Prefix each strength/weakness with [Dimension - Subcriterion] using the exact text above.

- You may have multiple strengths/weaknesses under the same dimension.

- You do not need to cover all dimensions or all subcriteria.

- Be specific and provide concrete examples from the paper.

- For weaknesses, offer constructive suggestions for improvement.

- Include all distinct, substantive strengths and weaknesses needed for a thorough review.
  Do not pad the review with marginal points or merge unrelated issues merely to meet a
  particular count.

- Keep the exact overall score and the numeric 1-5 confidence out of `<reasoning>`; put
  them only in the JSON `rating` object.

Calibration Guidelines:

- Use the full range of the rating scale.

- Avoid clustering around the middle.

- Your JSON confidence should reflect your expertise in the specific subfield. It is
  retained only for compatibility and is not the verifier-derived confidence.

Confidence Scale (1-5):

- 5: Absolutely certain; very familiar with the related work and technical details.

- 4: Confident, though not absolutely certain.

- 3: Fairly confident.

- 2: Willing to defend the assessment, but uncertainty is substantial.

- 1: An educated guess.

---

## PAPER

{paper_text}
