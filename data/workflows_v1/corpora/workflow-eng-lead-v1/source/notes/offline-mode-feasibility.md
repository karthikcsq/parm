# Offline mode: not feasible, and why - 2026-04-18

Asked to scope "works on a plane". Answer is no, and the reason is worth
recording so we stop rescoping it every quarter.

What could work offline: file navigation, search over an already-built index,
reading past transcripts, config editing.

What cannot: anything that calls the model. Which is the product.

So an offline mode is a file browser with our keybindings. The honest version
of the request is usually one of:

- "I want it to fail gracefully when the network drops mid-session" — that is
  real, that is a bug, and we should fix it properly.
- "I want to queue work and have it run when I reconnect" — plausible, but it
  is a job queue and a different product surface.
- "I want a local model" — a different company.

Recommending we fix graceful degradation, name it that, and stop calling it
offline mode. Filed as #38.
