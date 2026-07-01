from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from parm_bench.backends import GBrainBackend


class GBrainBackendTests(unittest.TestCase):
    def test_search_pins_resolved_repo_local_home_for_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / ".gbrain-local" / "runtime" / "gbrain"
            runtime.mkdir(parents=True)
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="[0.900] notes/example.md -- Retrieved text\n",
                stderr="",
            )
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("parm_bench.backends.subprocess.run", return_value=completed)
                as run,
            ):
                hits = GBrainBackend(repo_root=root).search("query", limit=2)

            self.assertEqual(hits[0].source_id, "note/example.md")
            self.assertEqual(
                run.call_args.kwargs["env"]["GBRAIN_HOME"],
                str((root / ".gbrain-local" / "home").resolve()),
            )
            self.assertEqual(run.call_args.kwargs["cwd"], runtime.resolve())

    def test_relative_gbrain_home_override_resolves_from_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = GBrainBackend(
                repo_root=root,
                gbrain_home="custom/home",
            )
            self.assertEqual(
                backend.gbrain_home,
                (root / "custom" / "home").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
