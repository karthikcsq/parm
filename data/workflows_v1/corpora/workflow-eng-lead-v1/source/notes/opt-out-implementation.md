# How the telemetry opt-out actually works - 2026-02-19

Writing this down because I have explained it three times.

Precedence, highest first:

1. `--no-telemetry` on the command line
2. `CLI_TELEMETRY=0` in the environment
3. `telemetry.enabled: false` in the project config
4. `telemetry.enabled: false` in the user config
5. default, which is on

Opting out disables the flag SDK's network calls and the crash reporter. It
does not disable local logging to `~/.cli/logs`, which never leaves the
machine.

The one subtlety: the flag SDK is initialised before we have finished reading
the project config, so an opt-out in the project file takes effect on the
second run, not the first. Dan filed this and we agreed it was acceptable for
now. If someone is fixing it, the fix is to defer SDK init until after config
resolution, not to read the config earlier.

Opt-out state is not itself reported anywhere. That would be funny.
