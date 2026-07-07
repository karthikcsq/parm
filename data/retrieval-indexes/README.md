# Retrieval Indexes

This directory stores tracked frozen retrieval artifacts for PARM benchmark
replay. Canonical runs load these files directly and do not call `gbrain
search` or require a live GBrain database.

`amara-life-v1/` was exported from the Amara Life GBrain substrate using
GBrain `0.42.53.0` and `openai:text-embedding-3-small` at 512 dimensions.
Refresh it only when intentionally changing the corpus, chunker, embedding
model, or exporter.
