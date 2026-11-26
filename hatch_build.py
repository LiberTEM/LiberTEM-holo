import os
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def get_git_rev():
    try:
        new_cwd = os.path.abspath(os.path.dirname(__file__))
        rev_raw = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=new_cwd,
            stderr=subprocess.DEVNULL,
        )
        return rev_raw.decode("utf8").strip()
    except Exception:
        return "unknown"


def write_baked_revision(baked_dest):
    baked_dest.write_text(r'revision = "%s"' % get_git_rev())


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = 'custom'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, version, build_data):
        if self.target_name not in ["wheel", "sdist"]:
            return  # something else
        here = os.path.abspath(os.path.dirname(__file__))
        baked_dest = Path(here, "src", "libertem_holo", "_baked_revision.py")
        if os.path.exists(os.path.join(here, ".git")):
            write_baked_revision(baked_dest)
        # When building from an sdist, there is no .git, but the sdist
        # contains the file baked from the checkout.
        if version == "editable" or not baked_dest.exists():
            return
        # _baked_revision.py is in .gitignore, so it has to be included explicitly:
        if self.target_name == "wheel":
            dest = "libertem_holo/_baked_revision.py"
        else:
            dest = "src/libertem_holo/_baked_revision.py"
        build_data["force_include"][str(baked_dest)] = dest
