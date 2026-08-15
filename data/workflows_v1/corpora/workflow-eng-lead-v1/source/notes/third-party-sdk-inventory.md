# Third-party SDK and processor inventory - status 2026-05-28

Assembling this for the Northwind DPA renewal. Priya owns the final document;
this is my working list of what the CLI actually talks to.

| Processor | What it receives | Where |
| --- | --- | --- |
| Statsig | feature flag checks, anonymous install ID | on start, on flag read |
| Sentry | stack traces with paths scrubbed, release version | on crash only |
| npm registry | package names on install | install time |
| our own API | auth token, request metadata | per request |

Open questions I still owe Priya:

- whether the Statsig SDK phones home on its own schedule separate from our
  flag reads (Dan is checking)
- whether Sentry's default breadcrumb collection is on and what it captures
- exact retention on our own API request logs

Status: incomplete. Do not send this to the customer until the three open
questions are answered and Priya has rewritten it in her own format.
