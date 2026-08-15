# Alert noise review - 2026-05-22

Sam went through six months of pages. 94 total, 11 were real.

The offenders:

- **Build box disk usage**: 38 pages, 0 real. Fires at 80%, and the box sits at
  79% because the artefact cache is bounded at that size on purpose. Fix: raise
  to 92% and alert on rate of change rather than level.
- **API p99 latency**: 22 pages, 2 real. Threshold set when we had a tenth of
  the traffic. Recalibrated.
- **Certificate expiry**: 9 pages, 9 real, but all for the same cert over
  successive days because there is no acknowledgement. Fix: acknowledge
  silences for 24 hours.

Left alone: error rate and auth failure rate. Both quiet, both real when they
fire.

The number I want remembered is 11 out of 94. An on-call who has been woken by
83 false alarms responds differently to the 84th page, and that is the actual
risk here, not the sleep.
