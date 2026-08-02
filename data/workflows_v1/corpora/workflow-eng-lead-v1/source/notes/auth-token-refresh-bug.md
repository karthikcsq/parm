# Token refresh race - 2026-04-27

Two concurrent requests both see an expired token, both refresh, second refresh
invalidates the first, one request fails with a 401 that looks random.

Reproduced by running any two long tool calls in parallel across the expiry
boundary. Rare in practice because the window is narrow, which is exactly why
it took six weeks of "it logged me out once" reports to find.

Fix is a single-flight lock around refresh: first caller refreshes, others wait
on the same promise. Twelve lines in `src/auth/session.ts`.

What I want remembered is the diagnosis, not the fix. The reports said
"randomly logged out". The stack traces said 401. Neither pointed at
concurrency. What found it was Sam noticing the failures clustered at exactly
the times a token would expire, which is the sort of thing you only see if you
plot the timestamps rather than reading the tickets one at a time.

Landed in 1.0.70.
