Subject: Re: picking an events vendor
From: me
To: Dan
Date: 2026-03-19

Dan compared Statsig, Amplitude, and rolling our own thin collector.

Statsig came out ahead on the things we care about: the client bundle is small,
the SDK does not phone home on import, and their retention controls are per
event rather than per project. Amplitude was better on the query side but we do
not have anyone who would use it.

Decision was to standardise on Statsig when we do instrument. Dan has a sandbox
project set up and a client key in the shared vault under `statsig/dev`.

This was a vendor choice, not a schedule. It says nothing about when anything
ships.
