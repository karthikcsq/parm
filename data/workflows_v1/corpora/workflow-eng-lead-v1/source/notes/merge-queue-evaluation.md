# Merge queue evaluation

Looked at turning on the merge queue after the v1.0.68 incident.

Verdict: not yet. Our CI is twelve minutes and the queue would serialise every
merge behind it. At our merge rate that is fine, but the queue also re-runs
checks on the merged result, which would double the bill for no signal we do
not already have from the tag smoke job.

Revisit when either CI drops under five minutes or we are merging more than a
dozen pull requests a day. Neither is close.

Meanwhile: merge from the pull request page, not from the command line, so the
merge shows up in the pull request timeline.
