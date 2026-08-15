# CLI startup latency - 2026-03-09

Cold start was 840ms. Target was under 300ms. Got to 210ms.

Where it went:

- 310ms: loading the full config schema validator on every start. Now lazy;
  only constructed when a config file is actually present and non-trivial.
- 190ms: the plugin discovery walk, which stat'd every directory under the
  plugin root. Now reads a cached manifest and only walks on cache miss.
- 140ms: importing the model client library at module scope for a code path
  most invocations never reach. Moved behind the first call that needs it.

The remaining 210ms is roughly half Node startup and half our own module graph.
Getting below 150ms means shipping a bundled binary, which is a different
project.

General lesson, which I keep relearning: almost all of it was work done eagerly
for a case that usually does not happen. Nobody had written a slow function.
