# What must never reach a log line - 2026-02-27

Standing rules. These apply to local logs, crash reports, and anything we send
anywhere. Independent of any review or freeze; these are always true.

Never logged, in any form:

- prompt text or model output text
- file contents, file paths outside the project root, or repository names
- environment variable values, tokens, keys, or anything from a `.env`
- the user's name, email, or machine hostname

Allowed:

- counts, durations, exit codes, error class names
- the name of a tool that was invoked, not its arguments
- a stable anonymous install ID that we generate and the user can reset

The test is whether the line would still be safe if it were pasted into a
public issue by a user trying to get help. That happens constantly.

Marcus reviews the scrubber's deny list every quarter. Last pass 2026-02-24,
no changes.
