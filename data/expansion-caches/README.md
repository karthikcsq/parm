# Expansion Caches

This directory stores tracked frozen query-expansion caches for enhanced
retrieval. Canonical enhanced runs should use `--expansion-policy frozen` so a
benchmark replay never silently generates new expansion queries.

`amara-life-v1/` contains one cache entry for each distinct prompt in the
current benchmark set, with three intent-preserving alternatives per prompt.
