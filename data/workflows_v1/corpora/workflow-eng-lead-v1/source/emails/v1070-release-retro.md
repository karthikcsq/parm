Subject: v1.0.70 retro notes
From: me
To: eng
Date: 2026-05-06

Short one. v1.0.70 went out clean on Tuesday, no rollback, no follow-up patch.

Two things that went right and I want repeated: Mei smoke-tested the tag by
hand before we announced, and the release note said what changed in plain
words instead of listing commit subjects.

One thing to fix: we tagged before the CHANGELOG heading was bumped, so the
tag and the changelog disagree by one version. Same mistake as v1.0.66. Bump
them in the same commit.

Nothing else. Good week.
