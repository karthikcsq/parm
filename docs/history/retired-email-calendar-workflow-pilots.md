# Retired Email + Calendar workflow pilots

## Status

`workflows_email_calendar_v1` and `workflows_email_calendar_v2` were retired from the active tree on 2026-08-23. They were exploratory fixture-construction efforts, not reportable PARMBench Workflow evidence.

Their complete pre-cleanup Git state is preserved in the external bundle recorded during the repository-slimming migration. The removed implementation is intentionally not an active dataset, test target, or quick-start path.

## Why they were retired

The pilots demonstrated useful adapter mechanics:

- a local, deterministic Email + Calendar fixture environment;
- list/detail separation for late observations;
- target-bound pending-review, pending-approval, and pending-exception state;
- cue-ablated controls that did not prescribe a single ordinary action.

They did **not** establish an independent workflow family. The construction was too small and too coupled to its fixture vocabulary, and the v2 diagnostic did not provide a sufficiently actionable source-included ceiling to justify a policy comparison. Treating its target-state checks as a successful benchmark would have overstated the evidence.

## What a replacement must prove before it becomes active

A future Email + Calendar suite must be a new, independently certified workflow family—not a repair of the old rates. Every scenario must have:

1. an ordinary goal that identifies a work item without prescribing an action, policy term, or hidden identifier;
2. one late list-hidden detail that establishes applicability, while the cue itself does not state the policy conclusion;
3. dated ordinary history whose commitment adds a non-obvious constraint;
4. a source-included ceiling that reaches the safe, target-bound state in end-to-end trajectories;
5. a cue-ablated twin differing only in that applicability fact;
6. a safe reversible disposition such as a pending review/exception or a routed follow-up, not an invented approval or irreversible reschedule;
7. prompt-only fairness, timing, and independently namespaced caches;
8. fresh samples and a declared threshold before any retrieval claim.

The retained `email_calendar_fixture` adapter is reusable infrastructure. A new family should be introduced under a fresh versioned dataset name only after these construction gates are met.
