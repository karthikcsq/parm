# Postmortem - 2026-05-19 - search returning empty results

**Impact.** Project search returned zero results for about 6% of users for
roughly 30 hours. Silent: no error, just nothing found.

**Cause.** The 1.0.70 chunker change altered the index format without changing
the format version. Clients with an existing index read it with the new reader,
got garbage offsets, and returned nothing. Users with no prior index were fine,
which is why it looked like a small percentage and why none of us reproduced it
locally.

**Detection.** 30 hours. Two support tickets that Aditi connected. Our
monitoring cannot see this: a search returning zero results is
indistinguishable from a search with no matches.

**Fix.** Index format version stamp, and a rebuild when it does not match.
Shipped in 1.0.70.2.

**Actions.**

- Version stamp on any on-disk format. Done, and now part of the review
  checklist.
- Find a signal for silent-wrong-answer failures. Open, and I do not have a
  good idea. Rate of zero-result searches is confounded by real zero-result
  searches.

The uncomfortable part: the change was reviewed by two people and neither of us
asked what happens to an index written by the previous version.
