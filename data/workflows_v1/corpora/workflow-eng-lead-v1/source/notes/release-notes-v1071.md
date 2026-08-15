# Release notes - v1.0.71 - 2026-05-26

Small release. Mostly the fallout from 1.0.70.

**Fixed**

- Search returned no results for users whose index was written by a pre-1.0.70
  chunker (#37). Index format is now versioned and rebuilt on mismatch. This
  shipped as 1.0.70.2 and is included here for completeness.
- Plugin discovery no longer walks the plugin root on every start.

**Changed**

- Certificate expiry alerting moved into the standard automation. No user
  impact; noting it because the March outage is referenced in support macros.

**Known issues**

- #46 and #49 remain open and unassigned. Both are memory or context related.
  Neither has a workaround beyond restarting the session.

Stable. No hotfixes needed against this one, which is the first time since
1.0.67.
