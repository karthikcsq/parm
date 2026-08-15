# Hotfix retro - 2026-03-11 - the empty config crash

We shipped a hotfix for a crash on empty config files. Start to finish: 90
minutes. Retro because it went well and I want to know why.

Timeline:

- 09:12 first report in the community channel
- 09:20 Mei reproduced it with a zero-byte `.clirc`
- 09:35 one-line fix on `hotfix/empty-config-v1.0.68` off main
- 09:52 review from Dan, merged
- 10:41 1.0.68.1 out

What made it fast: the reproduction was trivial and someone found it in eight
minutes. Everything after that was mechanical.

What I would not generalise: this was a one-line fix to a crash with a
deterministic repro. Most of what we call a hotfix is not that. The memory
regressions in #46 and #49 are the opposite shape, and if anyone points at this
90-minute retro as the standard for those, that is a misuse of it.

No process changes.
