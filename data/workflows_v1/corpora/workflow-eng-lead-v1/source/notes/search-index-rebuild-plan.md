# Search index rebuild - plan - 2026-05-11

The project search index has been drifting since we changed the chunker in
1.0.68. Symptoms: stale results for files that were renamed, and a long tail of
chunks pointing at line numbers that no longer exist.

Plan:

1. Ship the version stamp first, so an index built by an old chunker is
   detectable rather than silently wrong. Small, independent, low risk.
2. Add a background revalidation pass that drops chunks whose file hash has
   changed.
3. Only then consider a full rebuild on upgrade, which is the expensive option
   and the one users will notice.

Mei's point in review, which I agree with: do not start with step 3 because it
is the most satisfying. A full rebuild hides the drift without explaining it,
and we will be back here after the next chunker change.

No customer has reported this. It came out of my own use.
