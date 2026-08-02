Subject: Re: how long do we keep request logs?
From: me
To: Lena
Date: 2026-04-16

Thirty days for raw request logs, then they age out of the hot store and are
deleted. Aggregates derived from them are kept indefinitely because they carry
no identifiers.

Crash reports are ninety days, which is Sentry's default and we never changed
it. I have wondered whether ninety is defensible when thirty is enough for
anything we actually do with them. Not urgent, but if you are writing the
retention page, flag it as a question rather than documenting ninety as though
we chose it deliberately.

Flag evaluation records are not retained at all in a form we can query. Statsig
keeps their own; what they keep is in their DPA, not ours.

No customer has asked us to shorten any of these.
