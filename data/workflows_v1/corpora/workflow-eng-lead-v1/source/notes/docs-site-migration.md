# Docs site migration - 2026-02-13

Moving the docs off the wiki onto a static site in the repository.

The reason is not that the wiki is bad. It is that documentation which does not
live beside the code does not get updated in the same pull request as the code,
and six months later it describes a product that no longer exists. Roughly a
third of the wiki was already wrong when I audited it.

Mechanics: Markdown under `docs/`, built on merge to main, published to the
docs domain. Yuki owns the build.

What we lose: non-engineers can no longer edit. Aditi's team wrote a lot of the
troubleshooting pages and now they need a pull request. I think this is a real
cost and worth paying, but I want it recorded that we chose it rather than
discovered it.

Old wiki goes read-only on migration, not deleted. Deleting it breaks every
link anyone has ever pasted into a ticket.
