# Release checklist

Patch releases go out Tuesday and Thursday mornings. Anything landing after
Thursday noon waits for the next window.

1. Confirm the milestone is empty of `critical` labels.
2. Bump the version in `package.json` and the CHANGELOG heading together. They
   drift constantly when done in separate commits.
3. Tag as `v1.0.x`. Hotfix branches use the version they are fixing toward, so
   a fix aimed at 1.0.72 lives on `hotfix/<topic>-v1.0.72`.
4. Wait for the smoke job on the tag, not on the branch.
5. Post the release note in #eng-releases with the issue numbers it closes.

I do not want release notes generated from commit subjects. Write two sentences
that say what a user will notice.
