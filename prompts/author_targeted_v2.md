---
id: author_targeted_v2
version: v2.0.0
block: [rebuttal]
purpose: >
  Draft one natural targeted author response using only the complete submitted paper and
  authentic author rebuttal as sources for factual and empirical claims.
design_notes: >
  The prompt uses the exact untrimmed paper text and scoring-relevant P0 review frozen for
  the paired R6-real condition. Output is plain text and is stored verbatim. There is no
  source catalog, response-unit schema, fallback template, or local rewriting.
---

Act as the paper's authors and draft a concise rebuttal addressing the target review.

The submitted paper and authentic author rebuttal are the only sources for factual and
empirical claims. You may reuse or rephrase facts, explanations, analyses, and
experimental results explicitly present in either source.

Do not invent experiments, results, numbers, citations, implementation details, or
completed revisions. If the reviewer requests an experiment that is absent from both
sources, respond using existing evidence when possible; otherwise state that it could not
be completed during the rebuttal period. Do not pretend that it was performed.

Write a natural rebuttal. Prioritize the important concerns. Do not mention the original
human reviewers, final decision, or acceptance outcome. Return only the rebuttal text,
without JSON, source IDs, or an explanation of these instructions.

## SUBMITTED PAPER

{paper_text}

## TARGET REVIEW

{source_review}

## TARGET REVIEW'S INITIAL SCORE

{initial_score}

## AUTHENTIC AUTHOR REBUTTAL

{real_response}
