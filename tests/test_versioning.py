import shutil
import subprocess
import sys
from pathlib import Path

import libertem_holo.versioning


def test_get_git_rev_outside_checkout(tmp_path):
    # An installed copy (no .git two levels up) must not call git,
    # which would print "fatal: not a git repository" to stderr.
    pkg = tmp_path / "site-packages" / "libertem_holo"
    pkg.mkdir(parents=True)
    shutil.copy(Path(libertem_holo.versioning.__file__), pkg / "versioning.py")
    res = subprocess.run(
        [
            sys.executable, "-c",
            "import versioning; print(versioning.get_git_rev())",
        ],
        cwd=pkg,
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == "unknown"
    assert res.stderr == ""
