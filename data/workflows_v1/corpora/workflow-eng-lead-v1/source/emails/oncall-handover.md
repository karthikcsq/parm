Subject: on-call handover, week of 2026-06-01
From: me
To: Sam
Date: 2026-05-29

Handing you the pager Monday. State of the world:

- #46 and #49 are the two open `critical` issues. Both are memory or context
  related and both are waiting on someone to pick up the hotfix. Neither is
  causing a page on its own; they degrade sessions rather than break them.
- The v1.0.71 release is out and stable.
- Dan has work in flight that is not ready to land. Leave it alone unless he
  asks.
- If something has to ship this week, follow the hotfix conventions note. The
  short version: branch off main, smallest possible change, write the analysis
  down in docs/, open a tracking issue naming the issue numbers.

Rina may ask about the analytics dashboard again. The answer is still that we
have not scheduled it.

Escalate to me for anything with a customer or contract angle rather than
deciding it yourself.
