# Branching model - SUPERSEDED 2026-03-20

Do not follow this. Kept for context on old pull requests.

This described the git-flow arrangement we used until March: a long-lived
`develop` branch, release branches cut from it, hotfixes branched from `main`
and merged back to both.

> Feature work branches from `develop`. Release branches are cut on the first
> Monday of the month. `main` only ever receives merges from a release branch
> or a hotfix branch.

Why it went: with one release a week and a team of five, `develop` was a
merge-conflict holding pen that nobody looked at until release day. The double
merge on hotfixes was forgotten roughly half the time, so `develop` regularly
lacked fixes that were live in production.

Replaced by trunk-based development off `main` with short-lived branches, see
the hotfix conventions note for what a hotfix looks like now.

The only reason to read this is to understand a pull request opened before
2026-03-20.
