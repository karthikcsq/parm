# Compaction strategies we tried - 2026-05-18

Notes from a week of experiments on what to drop when the context fills.

Tried:

1. **Oldest-first.** Simple, and wrong often enough to be annoying. The first
   few turns usually contain the task statement, which is the last thing you
   want to lose.
2. **Summarise-and-replace.** Better retention of intent, but the summary drifts
   over successive compactions. After four rounds the summary of the summary
   describes a task nobody asked for.
3. **Pin the first N turns, then oldest-first.** Currently shipping. Crude and
   works. N is 3.
4. **Relevance-scored eviction.** Score each turn against the current task,
   drop lowest. Best quality in the small tests, but scoring requires a model
   call on every compaction and doubles latency at exactly the moment the user
   is already waiting.

Where this connects to #49: auto-compact not triggering means none of this runs
at all on large repos, so the strategy question is downstream of the trigger
bug. Fix the trigger first.
