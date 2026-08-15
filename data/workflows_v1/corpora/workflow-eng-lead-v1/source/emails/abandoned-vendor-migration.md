Subject: Re: moving off the managed queue - ABANDONED
From: me
To: Tomas
Date: 2026-03-23

Adding a note at the top because this thread gets found: we did not do this.
Do not use it as a plan.

The original proposal was to move the job queue off the managed service onto
self-hosted Redis to cut cost.

Where it died: Tomas costed the operational side properly and the saving was
about $400 a month against roughly a day a month of someone's attention plus a
new thing that can page us at night. At our size that is a bad trade.

Worth keeping because the analysis is right and the question will come back
when the bill grows. The number that would change the answer is somewhere
around $3,000 a month, or the point where we have someone whose job is
infrastructure rather than Tomas doing it alongside everything else.

Thread closed. Nothing was started; no code exists.
