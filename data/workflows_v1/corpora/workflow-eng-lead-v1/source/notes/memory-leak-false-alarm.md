# The 1.0.66 "memory leak" that was not - 2026-02-24

Recording this because it cost three days and the lesson is cheap.

Reports said memory grew unboundedly in long sessions. Heap snapshots showed
retained transcript entries. Obvious leak.

It was not a leak. It was the transcript doing exactly what it was designed to
do: keep everything in memory so `/compact` and history search can read it
back. Nothing was retained by accident, and nothing was reachable that should
not have been. Growth was linear in turns because the data was linear in turns.

The actual problem is that unbounded-by-design is indistinguishable from a leak
once a session runs long enough, and we had no bound.

Which is the same underlying issue as #46, approached from the other end. The
difference is that in February I concluded "not a bug, working as designed" and
closed it. In hindsight that was the wrong call: "working as designed" and
"the design is wrong" are compatible, and I used the first to avoid the second.
