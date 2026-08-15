# Release notes - v1.0.70 - 2026-05-12

**Fixed**

- Token refresh race that caused intermittent 401s during parallel tool calls
  (#33). Concurrent refreshes now share a single request.
- Config precedence: project-level settings were ignored on first run after
  install.

**Changed**

- Project search chunker rewritten for better handling of large files. Existing
  indexes are rebuilt automatically on first search.
- Startup time reduced from ~840ms to ~210ms.

**Known issues**

- Long sessions still grow heap without bound (#46).
- Auto-compact does not trigger reliably on very large repositories (#49).

Post-release note added 2026-05-20: the chunker change in this release shipped
without an index format version stamp and caused silent empty search results
for users with an existing index. Fixed in 1.0.70.2. See the 2026-05-19
postmortem.
