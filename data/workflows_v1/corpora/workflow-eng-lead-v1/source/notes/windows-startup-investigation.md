# Windows startup investigation

Nine seconds cold, two on macOS. Traced most of it.

- 4.1s: enumerating the workspace with `fs.readdir` recursively. Windows
  Defender scans each open. Excluding the project directory in Defender takes
  this to 0.9s, which is not a fix we can ship.
- 2.3s: resolving the shell environment by spawning a login shell. We do this
  to pick up the user's PATH. On Windows it spawns PowerShell, which is slow to
  start.
- 1.4s: the rest, evenly spread.

Fix order I would take: cache the resolved environment with a short TTL, then
replace the recursive readdir with a bounded breadth-first walk. Neither is
urgent enough for a hotfix. This is a v1.1 item.
