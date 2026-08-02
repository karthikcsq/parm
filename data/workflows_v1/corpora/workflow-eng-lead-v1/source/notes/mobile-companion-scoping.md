# Mobile companion - what it is and is not - 2026-03-14

Kofi is building a phone app. Scoping it before it grows.

In scope:

- read a session transcript that is already finished
- approve or reject a pending action the desktop CLI is waiting on
- push notification when a long run completes

Explicitly out of scope for v1:

- starting a session from the phone
- editing files
- anything that requires the model to run without a desktop session attached

The reason for the line: the value is being able to unblock a run while away
from the desk. The moment it can start work on its own it needs its own auth
story, its own rate limits, and its own support surface, and it stops being a
companion.

Kofi disagrees about starting sessions. Noted, revisit after v1 ships and we
see whether anyone actually uses the approval flow.
