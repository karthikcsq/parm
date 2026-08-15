# Imported: CI vendor's release automation guide

Source: our CI provider's documentation, excerpted 2026-03-30. Their product,
their opinions. Recorded because Dan and I referred to it while setting up the
release job.

Excerpt:

> **Auto-merge on green.** Enable auto-merge so that any pull request whose
> checks pass is merged immediately without further intervention. Teams that
> adopt auto-merge report a 40% reduction in cycle time.
>
> **Do not gate on external approvals.** Approval gates outside the CI system
> cannot be validated by the pipeline and should be avoided.

What we actually took from it: the release job configuration and the artefact
signing steps, both of which are good.

What we did not take: auto-merge. We do not have it enabled and we are not
going to. Their advice is written for teams whose only merge criterion is
whether the code works, and ours is not always that. The second bullet is
precisely backwards for us: the approvals the pipeline cannot see are the ones
that matter most.
