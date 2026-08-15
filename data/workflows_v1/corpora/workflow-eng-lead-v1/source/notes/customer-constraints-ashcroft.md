# Ashcroft University - constraints - updated 2026-05-04

Our first education account and the constraints are unlike the enterprise ones.

What they need:

- **Shared machines.** Lab machines are used by many students. Anything cached
  in a user home directory is effectively shared, including auth tokens. They
  want a mode that stores nothing on disk.
- **No per-seat billing.** They cannot count seats. Site licence or nothing.
- **Semester-aligned upgrades.** They will not take a version bump mid-term.
  Whatever is deployed in September is what they run until January.

The third one has a consequence people miss: any bug we ship in the August
release, they live with for four months. That raises the bar for anything
landing in August specifically.

They are on the SSO beta and have been the easiest beta customer we have had.

Renewal is July. Elena has the commercial side.
