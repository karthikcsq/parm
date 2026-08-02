# Release notes - v1.0.69 - 2026-04-28

**Added**

- Seat changes are now prorated immediately on addition. Removals credit on the
  next cycle. See the proration decision log.
- Audit log export for enterprise tenants.

**Fixed**

- Plugin manifest cache was invalidated on every start, adding ~190ms.

**Changed**

- Issue labels migrated to the new scheme. Old `P0`-`P3` labels remain on
  historical issues and do not carry their old meaning.

**Known issues**

- Long sessions grow heap without bound (#46).
- A regression in this release caused duplicate audit log entries for tenants
  with more than one active session. Fixed in 1.0.69.1 on 2026-04-30. This is
  the regression that prompted the regression-suite gap audit.

Nothing here is telemetry-related; the instrumentation work was in flight but
did not land in this release.
