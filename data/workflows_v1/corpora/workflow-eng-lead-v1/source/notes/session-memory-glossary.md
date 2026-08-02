# Glossary: the four things we call "memory" - 2026-05-25

Three separate conversations went wrong last month because two people meant
different things. Fixing the vocabulary.

- **Transcript.** The in-process array of turns for the current session. Lives
  in `src/session/transcript.ts`. This is what grows unboundedly and what #46
  is about.
- **Context.** The subset of the transcript we actually send to the model,
  bounded by the context window. What `/compact` manages and what #49 is about.
- **Heap.** Node's process memory. Where the OOM happens. Related to the
  transcript by cause but not the same thing.
- **Project memory.** The `CLI.md` file users write to give the tool standing
  instructions. Unrelated to all of the above and the most common source of
  confusion in support tickets, because users say "memory" and mean this.

When writing an issue title, say which one. "Memory regression" could be any of
the first three and the title alone will not tell the next person.
