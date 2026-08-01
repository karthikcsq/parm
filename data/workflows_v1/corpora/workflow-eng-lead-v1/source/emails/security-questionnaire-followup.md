Subject: Re: security questionnaire, section 7
From: me
To: Priya
Date: 2026-04-30

Section 7 asks what data leaves the machine. Today the honest answer is short:
crash reports, opt-in, with stack traces scrubbed of paths.

I filled in the rest of the questionnaire and left section 7 to you since the
wording matters more than the facts there.

One thing worth flagging for whenever we do instrument: their question is
phrased per data category, not per vendor, so adding a collector does not by
itself change the answer, but adding a new category does. Session identifiers
and workspace identifiers would be a new category.
