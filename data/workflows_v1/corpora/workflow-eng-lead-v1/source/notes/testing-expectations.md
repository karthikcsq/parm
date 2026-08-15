# What I expect a change to be tested with

A bug fix comes with a test that fails before the fix. If writing that test is
hard, say so in the pull request and I will usually accept a manual repro
written down instead. What I will not accept is silence about it.

A refactor comes with no new tests and no changed tests. If the tests had to
change, it was not a refactor.

Documentation changes need no tests. Do not add a snapshot test of a markdown
file.

Coverage numbers are not a target. I have never once made a decision based on
the coverage percentage and I would rather nobody spent time raising it.

Performance claims need a measurement in the pull request body. "Should be
faster" is not a claim, it is a hope.
