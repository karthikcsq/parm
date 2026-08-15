# Plugin API versioning - 2026-04-02

We are about to have third-party plugins, which means we are about to have a
compatibility promise whether or not we write one down. Writing one down.

The contract:

- The plugin API is versioned separately from the CLI. A plugin declares
  `apiVersion: 1`.
- Within a major version we add, we never remove or change meaning.
- A removal needs a new major version and one full release cycle where both are
  supported and the old one warns.

The part I expect to regret if we skip it: plugins get a capability manifest
from day one, even though today every plugin would ask for everything. Adding
capabilities later means either breaking every plugin or grandfathering them
all in, and grandfathering is forever.

Yuki is writing the plugin author docs against this note, so if the design
changes, the docs change with it and not a month later.
