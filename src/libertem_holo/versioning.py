import os
import subprocess
from pathlib import Path


def get_git_rev():
    # NOTE: there is a copy of this code in hatch_build.py!
    try:
        new_cwd = os.path.abspath(os.path.dirname(__file__))
        # Only ask git if we run from a source checkout (src/libertem_holo/);
        # an installed copy may live inside an unrelated repository.
        if not (Path(new_cwd).parent.parent / ".git").exists():
            return "unknown"
        rev_raw = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=new_cwd,
            stderr=subprocess.DEVNULL,
        )
        return rev_raw.decode("utf8").strip()
    except Exception:
        return "unknown"
