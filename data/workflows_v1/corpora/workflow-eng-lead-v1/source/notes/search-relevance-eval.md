# Search relevance: what we measure - 2026-05-20

We had no way to tell whether a chunker change helped, so every argument about
search was vibes. Now there is a fixed set of 40 queries against three sample
repositories with hand-marked correct files.

Metrics: recall@5 and mean reciprocal rank. Recall@5 is the one that matters
for us because the UI shows five results and nobody scrolls.

Current numbers, chunker v4: recall@5 0.72, MRR 0.51.

The honest caveat, which I want written down before someone quotes 0.72 in a
deck: the 40 queries are ones I wrote, and I wrote them after using the tool
for months, so they are the queries I already know it can answer. A real
evaluation needs queries from people who are not me. Ben is collecting some
from support tickets.

Do not tune the chunker against this set and then report the same set as
evidence. That is how you get a number that only describes itself.
