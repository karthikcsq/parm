# PARM Roadmap

This page lists the work that will make PARM's evidence broader, stronger, and more useful. Completed experiment history lives in the result reports and decision log.

## 1. Protect the evidence already earned

- Keep the **157-scenario / 471-case / 140-persona** PARMBench v1 calibration batch frozen as an auditable, retrieval-agnostic evaluation substrate.
- Preserve all first-pass and development result namespaces. Revised fixtures and post-analysis runs remain labeled as development evidence.
- Version retrieval policies, judgment prompts, source manifests, caches, model settings, and result artifacts together.
- Keep deterministic causal scoring separate from any future model-graded natural-response evaluation.

## 2. Turn the calibration batch into decisive comparative evidence

- Run a preregistered, sealed comparison on the frozen PARMBench v1 batch across prompt-only retrieval, broad output retrieval, agent-initiated retrieval, and selective output-conditioned retrieval.
- Report the full causal scorecard: positive lift, cue-ablated restraint, ceiling actionability, gold and spurious admission, source support, privacy restraint, and calibration.
- Add at least one independent model family and preserve per-model result sidecars rather than averaging away disagreement.
- Keep candidate generation, admission, and downstream action metrics separate so a system cannot claim success merely for retrieving a relevant memory.

## 3. Strengthen selective retrieval without tuning away failure modes

- Develop a contrastive dense region-to-memory channel for semantic cues that lexical methods cannot express, especially relationship, hiring, and proactive-priority cases.
- Revisit controls when a remaining visible listing still has a legitimate relationship to the same memory.
- Calibrate any new threshold on development material and evaluate it exactly once on a newly sealed batch.
- Continue publishing misses and false interventions alongside wins. The cue-ablated control is the mechanism that keeps “more memory” from masquerading as better memory.

## 4. Scale executable workflow evaluation

PARMBench Workflows already validates multi-step agent trajectories with final-state scoring, timely-admission checks, and causal controls. The next step is breadth, not cosmetic tuning:

- Add independent workflow families with their own environment adapters and certify their memory-included ceilings before comparing retrieval policies.
- Extend beyond the current GitHub-like and email/calendar environments to additional task families such as expense handling, legal review, and other source-backed operations.
- Run repeated trajectories per condition and add a second model family so variance and model-specific behavior are visible.
- Keep reporting the gap between *memory retrieved* and *memory acted on*; both are required for an agent-memory system to be useful.
- Decide how trajectory-level admission budgets, broad-RAG baselines, and context compaction should be modeled before increasing corpus scale.

## 5. Establish external validity

- Build 50–100 paired end-to-end cases from corpus-scoped web and enterprise trajectories, beginning with LongMemEval-V2-compatible source material where licensing and provenance permit.
- Measure task success, causal memory lift, false intervention, faithfulness, privacy restraint, and user-calibrated helpfulness.
- Add a natural-response LLM judge only after measuring agreement with human reviewers and freezing the judge protocol.
- Keep deterministic PARMBench as the regression and mechanism gate even as richer end-to-end evaluation is added.

## Guiding principle

PARM's goal is not to make agents retrieve more personal memory. It is to make memory intervention **causal, selective, timely, inspectable, and safe enough to improve real decisions**.
