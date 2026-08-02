# Dependency audits - 2026-03-13

Monthly, first Monday, whoever is on call.

What the pass covers:

1. `npm audit` output, triaged rather than blindly upgraded. Most advisories
   are in dev dependencies or in code paths we do not reach.
2. Anything more than two majors behind, listed with a note on why.
3. New transitive dependencies added since last pass. This is the one that
   catches things: a patch bump to something we trust can pull in three new
   packages nobody evaluated.

What it does not cover: whether a dependency is well maintained. That is a
judgement call and it belongs in review when the dependency is added, not in a
monthly sweep.

Output is a short comment on a standing issue, not a document. If the pass
produces work, file it separately.

Marcus asked whether this should be automated. Partly: the data collection
should be, the triage should not, because the triage is the entire value.
