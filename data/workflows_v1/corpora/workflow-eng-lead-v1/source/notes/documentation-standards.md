# Documentation standards - 2026-02-25

Yuki owns the docs site. These are the rules we agreed.

- **One page, one job.** A page that explains a concept and also walks through
  a task does neither well.
- **Every code sample runs.** Samples are extracted and executed in CI. A
  sample that cannot be executed is prose and should be written as prose.
- **No "simply" or "just".** They are only ever true for the writer.
- **Link forward, not sideways.** A page links to the next thing a reader
  needs. A page that links to eight related pages is an index pretending to be
  a document.

On tone: write for someone competent who has not used this tool. Not for a
beginner and not for us.

The rule I expect to be argued with: documentation changes ship in the same
pull request as the code change. Yuki reviews, but the author writes. Handing
docs to a writer afterwards is how they end up describing the previous version.
