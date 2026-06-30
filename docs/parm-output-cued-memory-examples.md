# PARM Output Cue-Triggered Memory Examples

**Status:** Working draft for future expansion.

This document begins with 20 approved examples developed against the local
Amara Life GBrain corpus. It is intentionally extensible: future examples
should preserve the shared contract below and identify whether their memories
already exist in Amara or require a proposed corpus addition.

Every example follows the same contract:

1. The prompt is independently answerable and gives no reason to inspect GBrain.
2. A single large tool result or agent output introduces the decisive cue.
3. The cue is absent from the prompt.
4. Recalled memory materially changes the response decision.
5. The output contains enough other entities and propositions to make whole-output RAG noisy.

Unless noted otherwise, each mock output should be 8-12K tokens.

## 1. Conference agenda -> unresolved NovaMind diligence

**Status:** Approved; existing Amara memory.

- **Prompt:** "I have Thursday afternoon free at the conference. Pick three sessions from the agenda without schedule conflicts."
- **Noisy output cue:** A lower-ranked agenda entry names Chen Wei of NovaMind speaking on edge inference for grid reliability.
- **Memory:** Amara is leaning toward leading NovaMind's Series A but wants the Texas grid pilot data before presenting it to the partnership.
- **Decision delta:** Replace one agenda-only choice with Chen's session and use its Q&A to resolve the pilot-data gate.
- **Pass contract:** The response must select or elevate Chen's session and connect it to the outstanding Texas pilot evidence. Merely describing the session fails.
- **Cue unit:** `Chen Wei + NovaMind + grid reliability session`.
- **Source:** `notes/2026-04-09-novamind-followup.md`.

## 2. AI news digest -> CoreWeave dependency

**Status:** Approved; existing Amara memory.

- **Prompt:** "Give me a morning digest of enterprise AI-infrastructure news and choose one story worth sharing with founders."
- **Noisy output cue:** A lower-ranked item reports a material change to CoreWeave reserved-GPU pricing.
- **Memory:** Jordan Park said NovaMind's updated financial projections were built from its CoreWeave contract pricing.
- **Decision delta:** Select the CoreWeave story instead of the generic headline and flag NovaMind's cost assumptions for review.
- **Pass contract:** The response must elevate the CoreWeave story, connect it to NovaMind's projections, and recommend reassessing those assumptions.
- **Cue unit:** `CoreWeave -> material GPU pricing change`.
- **Source:** `emails/em-0017.md`.

## 3. Podcast feed -> personal burnout pattern

**Status:** Approved; existing Amara memory.

- **Prompt:** "Choose one episode from today's new podcast releases for my commute."
- **Noisy output cue:** A lower-ranked episode describes skipped workouts, desk lunches, and repeatedly working weekends as quiet burnout signals.
- **Memory:** Amara missed two gym sessions, ate at her desk four days in one week, and had worked three consecutive Saturdays.
- **Decision delta:** Choose the burnout episode instead of the feed's top-ranked technology episode.
- **Pass contract:** The response must change the episode selection because the behavioral pattern is personally timely, while avoiding an invasive recital of private facts.
- **Cue unit:** `behavioral pattern in output <-> pattern across personal notes`.
- **Sources:** `notes/2026-03-22-weekly-review.md`; `notes/2026-02-02-weekly-review.md`.

## 4. Startup expo -> outstanding Marcus Reid task

**Status:** Approved; existing Amara memory.

- **Prompt:** "Pick two demos from this startup expo that look technically distinctive."
- **Noisy output cue:** One of dozens of listings is Marcus Reid's procurement startup, framed as technically ordinary but operationally mature.
- **Memory:** Amara decided she needed to understand Marcus's product before introducing it to portfolio founders.
- **Decision delta:** Select Marcus's demo over a marginally more novel option because it resolves an outstanding decision about whether to make introductions.
- **Pass contract:** The response must select or elevate Marcus's demo and state that the goal is to evaluate it before facilitating introductions.
- **Cue unit:** `Marcus Reid -> procurement startup demo`.
- **Source:** `meetings/mtg-0003.md`.

## 5. Vendor comparison -> NovaTech counterparty risk

**Status:** Approved; existing Amara memory.

- **Prompt:** "Compare the warehouse-automation vendors in this procurement report and recommend one for a twelve-month pilot."
- **Noisy output cue:** NovaTech Labs ranks first on price and technical capability and offers a large discount for annual prepayment.
- **Memory:** NovaTech's burn increased, runway fell to roughly eight months, and Amara required updated financials before any commitment.
- **Decision delta:** Reject annual prepayment; recommend a short cancellable pilot or a different vendor pending financial review.
- **Pass contract:** The response must materially change the contract recommendation because of NovaTech's runway risk. A generic vendor-risk disclaimer fails.
- **Cue unit:** `NovaTech Labs + annual prepayment`.
- **Source:** `meetings/mtg-0006.md`.

## 6. Webinar catalog -> NovaMind's missing sales hire

**Status:** Approved; existing Amara memory.

- **Prompt:** "Pick one webinar from this week's enterprise-sales catalog for the team to attend."
- **Noisy output cue:** A lower-ranked webinar features a utility-sector sales operator explaining nine-month procurement cycles.
- **Memory:** Chen Wei acknowledged NovaMind needs a VP of Sales with utility-sector experience; enterprise sales timing is Amara's remaining concern.
- **Decision delta:** Select that webinar instead of the catalog's general sales keynote and suggest evaluating the speaker as a potential NovaMind introduction.
- **Pass contract:** The response must change the webinar choice and connect the speaker's utility-sales experience to NovaMind's identified hiring gap.
- **Cue unit:** `utility-sector sales operator + long procurement cycles`.
- **Source:** `notes/2026-04-09-novamind-followup.md`.

## 7. Robotics market map -> use the available Vela introduction

**Status:** Approved; existing Amara memory.

- **Prompt:** "Choose one company from this robotics market map that is worth learning more about."
- **Noisy output cue:** Vela appears deep in a long market map with recent European-expansion activity.
- **Memory:** Sarah Chen offered to introduce Amara to Vela's CEO after Halfway completed its Q2 thesis review.
- **Decision delta:** Prioritize Vela over the map's default top candidate and ask Sarah for the warm CEO introduction she already offered.
- **Pass contract:** The response must select or elevate Vela and recommend requesting Sarah's offered introduction. Inventing a separate cold-outreach or founder-interview process fails.
- **Cue unit:** `Vela + European expansion activity`.
- **Source:** `meetings/mtg-0005.md`.

## 8. Case-study newsletter -> Vespera hiring contradiction

**Status:** Approved; existing Amara memory.

- **Prompt:** "Choose one case study from this B2B growth report for our newsletter."
- **Noisy output cue:** The report celebrates Vespera Dynamics and recommends immediately accelerating sales hiring after one strong quarter.
- **Memory:** Amara told Elena to wait for Q2 confirmation before expanding headcount and requested updated hiring scenarios.
- **Decision delta:** Do not publish the case as an unqualified growth model; choose another case or frame Vespera around disciplined waiting.
- **Pass contract:** The response must reject the immediate-hiring lesson and preserve the Q2 evidence gate. Adding a mild caveat while endorsing acceleration fails.
- **Cue unit:** `Vespera Dynamics -> accelerate sales hiring now`.
- **Source:** `meetings/mtg-0007.md`.

## 9. Legal reading roundup -> founder-hostile terms

**Status:** Approved; existing Amara memory.

- **Prompt:** "Pick one article from this week's venture-law roundup to share with the partnership."
- **Noisy output cue:** A highly ranked article praises aggressive participating-preferred structures as the safest default in uncertain markets.
- **Memory:** Amara concluded that overly aggressive liquidation preferences select for desperate teams rather than exceptional founders and favored threshold conversion.
- **Decision delta:** Choose a different article or share this one explicitly as a position to challenge, not as recommended practice.
- **Pass contract:** The response must change the share/no-share decision or the article's intended framing based on Amara's recorded investment principle.
- **Cue unit:** `aggressive participating preferred -> default recommendation`.
- **Source:** `notes/2026-02-12-threshold-terms.md`.

## 10. Workflow marketplace -> 48-hour conviction rule

**Status:** Approved; existing Amara memory.

- **Prompt:** "Review these workflow automations and pick one lightweight tool for a two-week trial."
- **Noisy output cue:** One modestly ranked automation starts a 48-hour decision timer after an introductory meeting and asks owners to record a reason for conviction.
- **Memory:** Amara and Sarah wanted a 48-hour flag system: if neither could articulate conviction after an intro, they would pass.
- **Decision delta:** Select the 48-hour automation instead of the marketplace's more popular general task manager.
- **Pass contract:** The response must choose the timer workflow and connect it to the existing conviction rule. Generic enthusiasm about speed fails.
- **Cue unit:** `48-hour post-intro conviction timer`.
- **Source:** `notes/2026-03-14-next-quarter-plan.md`.

## 11. Phone feature digest -> neglected family connection

**Status:** Approved; existing Amara memory.

- **Prompt:** "Read these mobile OS release notes and choose one feature worth enabling."
- **Noisy output cue:** Buried among dozens of features is an opt-in reminder that notices when a recurring personal call has lapsed and suggests a private check-in.
- **Memory:** Amara noted that she had not called her mother in two weeks.
- **Decision delta:** Enable the relationship reminder instead of the release's headline productivity feature.
- **Pass contract:** The response must change the selected feature because it supports a currently neglected relationship, while describing the reason tactfully.
- **Cue unit:** `lapsed recurring personal call reminder`.
- **Source:** `notes/2026-02-26-weekly-review.md`.

## 12. Lunch search -> interrupt the desk-lunch pattern

**Status:** Approved; existing Amara memory.

- **Prompt:** "Pick one lunch option from these nearby restaurant results."
- **Noisy output cue:** The tool's top-ranked option emphasizes fast delivery directly to the office, while a slightly lower-ranked nearby restaurant has immediate dine-in availability.
- **Memory:** Amara had eaten lunch at her desk four days in one week and explicitly marked the pattern as unsustainable.
- **Decision delta:** Choose the dine-in option instead of the delivery winner so lunch becomes an actual break from the desk.
- **Pass contract:** The response must change the restaurant or fulfillment choice in favor of eating away from the desk. Ordering the top delivery option with a generic wellness reminder fails.
- **Cue unit:** `office delivery versus immediate dine-in`.
- **Source:** `notes/2026-03-22-weekly-review.md`.

## 13. Documentary catalog -> original climate motivation

**Status:** Approved; existing Amara memory.

- **Prompt:** "Choose one documentary from this new-release catalog for tonight."
- **Noisy output cue:** One lower-ranked film follows an infrastructure engineer who moved into climate finance after becoming frustrated that capital was not reaching deployable solutions.
- **Memory:** Amara described essentially that path as the reason she entered climate investing.
- **Decision delta:** Choose that documentary instead of the catalog's popularity leader.
- **Pass contract:** The response must change the selection because the film mirrors a personally important motivation, not merely because it concerns climate.
- **Cue unit:** `infrastructure engineer -> climate capital motivation`.
- **Source:** `notes/2026-02-04-morning-reflection.md`.

## 14. Weekend activity feed -> recovery over another work event

**Status:** Approved; existing Amara memory.

- **Prompt:** "Pick one thing from this weekend-events feed."
- **Noisy output cue:** The feed mixes investor breakfasts and founder salons with a phone-free outdoor workshop explicitly designed for people who have worked several consecutive weekends.
- **Memory:** Amara said three Saturdays at her laptop was unsustainable and that she needed to take a real weekend off.
- **Decision delta:** Choose the restorative event instead of the highest-ranked professional event.
- **Pass contract:** The response must choose a non-work option because of the repeated-weekend pattern. Selecting another networking event with a wellness caveat fails.
- **Cue unit:** `phone-free activity for repeated weekend workers`.
- **Source:** `notes/2026-02-02-weekly-review.md`.

## 15. Essay digest -> pre-term-sheet relationships

**Status:** Approved; existing Amara memory.

- **Prompt:** "Choose one essay from this long management-and-technology digest for our internal reading list."
- **Noisy output cue:** A lower-ranked essay argues that the most valuable founder relationships are built during quiet periods before a transaction becomes urgent.
- **Memory:** Amara identified spending more time with founders pre-term-sheet as a personal priority because she had become too reactive.
- **Decision delta:** Select that essay instead of the digest's top AI-management article and translate it into protected pre-deal founder time.
- **Pass contract:** The response must change the essay selection and connect it to proactive pre-term-sheet relationship building.
- **Cue unit:** `relationships built before transactions become urgent`.
- **Source:** `notes/2026-02-18-next-quarter-plan.md`.

## 16. Market-chart selector -> hidden capital-selection effect

**Status:** Approved; existing Amara memory.

- **Prompt:** "Pick one chart from this AI-infrastructure market pack for the Monday presentation."
- **Noisy output cue:** A lower-ranked chart shows compute costs rising fastest for early-stage teams and celebrates the resulting consolidation around well-capitalized startups.
- **Memory:** Amara worried that prohibitive compute costs were causing the fund to select for teams with deep-pocketed backers, undermining its purpose.
- **Decision delta:** Choose that chart instead of the headline market-growth chart and frame consolidation as a portfolio-selection risk rather than a success.
- **Pass contract:** The response must change both the chart choice and its interpretation based on Amara's concern.
- **Cue unit:** `rising early-stage compute cost -> well-capitalized-team consolidation`.
- **Source:** `notes/2026-02-26-weekly-review.md`.

## 17. Human-factors event catalog -> orange-mode operating model

**Status:** Approved; existing Amara memory.

- **Prompt:** "Pick one session from this broad human-factors and product-safety event."
- **Noisy output cue:** A lower-ranked talk covers interpretable handoff logs and operator cognitive load when automation confidence drops.
- **Memory:** Amara's orange-mode notes emphasize exactly those decision logs and the underestimated cognitive load during human handoffs.
- **Decision delta:** Choose that session instead of the event's popular autonomous-systems keynote.
- **Pass contract:** The response must select the handoff session because it advances the existing orange-mode framework; generic AI-safety relevance fails.
- **Cue unit:** `interpretable handoff logs + operator cognitive load`.
- **Sources:** `notes/2026-02-24-orange-mode.md`; `notes/2026-03-20-orange-mode.md`.

## 18. Health-tech event roundup -> available warm path

**Status:** Approved; existing Amara memory.

- **Prompt:** "Choose one event from this Austin technology roundup."
- **Noisy output cue:** A health-tech conference listing names Hannah Liu as a speaker and offers limited founder-office-hour slots.
- **Memory:** Hannah offered to connect Amara with the conference organizers and with Vero Health's CEO when Amara was ready.
- **Decision delta:** Select the health-tech conference over the roundup's generic top event and use the warm path rather than cold registration.
- **Pass contract:** The response must elevate the conference and recommend using Hannah's offered introduction without falsely claiming it is already scheduled.
- **Cue unit:** `Hannah Liu + Austin health-tech conference`.
- **Source:** `meetings/mtg-0001.md`.

## 19. Trail report -> smoke-triggered health constraint

**Status:** Approved example; requires a proposed personal-memory addition.

- **Prompt:** "Choose one hike from this weekend trail report."
- **Noisy output cue:** The report's top trail has reopened but warns of lingering prescribed-burn smoke; dozens of other trail, weather, wildlife, and road details create noise.
- **Proposed memory addition:** Amara's asthma is reliably aggravated by wildfire or prescribed-burn smoke.
- **Decision delta:** Reject the otherwise top-ranked trail and choose a smoke-free alternative.
- **Pass contract:** The response must change the trail selection because of the smoke-asthma connection. A generic air-quality caveat without switching trails fails.
- **Cue unit:** `lingering prescribed-burn smoke`.

## 20. Film-festival program -> migraine trigger

**Status:** Approved example; requires a proposed personal-memory addition.

- **Prompt:** "Pick one screening from this weekend film-festival program."
- **Noisy output cue:** A visually acclaimed film's buried accessibility note warns of extended high-frequency strobe sequences.
- **Proposed memory addition:** Amara has a history of migraines triggered by strobing light.
- **Decision delta:** Reject the critic-favorite screening and choose another film.
- **Pass contract:** The response must change the screening choice because of the strobe-migraine connection. Repeating the warning while retaining the selection fails.
- **Cue unit:** `extended high-frequency strobe warning`.

## Coverage summary

| Cue shape | Examples |
|---|---|
| Named entity plus event | 1, 2, 4, 5, 7, 8, 18 |
| Entity plus relationship or dependency | 2, 4, 6, 7 |
| Semantic behavioral pattern | 3, 11, 12, 14 |
| Personal value or identity resonance | 13, 15 |
| Decision-policy contradiction | 8, 9, 10, 16, 17 |
| Safety or health constraint | 19, 20 |
| Existing Amara memory | 1-18 |
| Proposed addition | 19-20 |

## Shared evaluation rule

The primary score is not whether the system retrieved a relevant memory. It is
whether the admitted memory caused the post-output response to make a better
decision. Each example therefore needs:

- a baseline decision supported by the tool output alone;
- a different gold decision supported by the output plus memory;
- required evidence or action language;
- forbidden unchanged-decision language; and
- enough noisy output to make whole-output retrieval produce irrelevant or
  misleading memory candidates.
