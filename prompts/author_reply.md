---
id: author_reply
version: v2.0.0
purpose: Reply to an LLM's own review (ICLR 2022). Writes the authors' response to one LLM review, using only the
  submitted paper and the authors' real rebuttal as sources for factual claims.
notes: Plain-text output, stored verbatim.
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
