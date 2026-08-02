# Trellis Retail - constraints - updated 2026-04-21

Mid-size, 200 seats, our second-largest contract.

What matters to them:

- **Change windows.** No deployments touching their tenant between 15 November
  and 5 January. Retail freeze, non-negotiable, in the contract.
- **Named support contact.** They have Aditi's direct line and use it. This is
  in the contract too.
- **Audit log export.** They need to be able to show who ran what. We built the
  export for them and nobody else uses it.

What does not matter to them, despite our assumptions: they have never asked
about analytics, subprocessors, or data residency. Their security review was
one page.

Risk to watch: the audit log export is single-customer code with one user and
no tests worth the name. It is exactly the sort of thing that breaks in a
refactor and nobody notices until Trellis notices.

Renewal is January, immediately after their freeze ends.
