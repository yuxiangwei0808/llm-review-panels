---
id: score_from_human_reviews
version: v2.0.0
purpose: Scoring from the human reviews. The model reads the paper's human reviews (ratings removed, reviewer order
  shuffled) as its own recorded observations and gives a score on the venue's scale, so models that disagree differ
  in how they judge, not in what they read.
notes: 'Output: a short public <reasoning> block, then flat KEY: value lines.'
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
