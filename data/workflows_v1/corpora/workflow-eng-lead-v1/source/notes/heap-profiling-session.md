# Heap profiling session - 2026-05-28

Spent an afternoon with the Chrome inspector attached to a session that had
been running since morning.

Retained size at four hours: 1.9 GB. Of that, 1.6 GB is a single array of tool
result strings. The transcript renderer holds a reference to the whole array so
that scroll-to-top works without a re-fetch.

Secondary offender, much smaller but growing the same way: the file watcher
keeps a debounce closure per watched path and never releases them when a
directory is deleted. About 40 MB after four hours. Worth a separate issue.

A hard byte cap on the transcript array plus spilling older entries to a temp
file would bring the four-hour figure under 300 MB by my arithmetic. That is
the fix I want in the hotfix. Anything cleverer can wait for a real release.
