from __future__ import annotations

import os
import subprocess
import sys


def _normalize_git_url(url: str) -> str:
    if url.startswith("git+"):
        return url
    if url.startswith("git@"):
        host, path = url.split(":", 1)
        return f"git+ssh://{host}/{path}"
    if url.startswith(("ssh://", "https://", "http://")):
        return f"git+{url}"
    raise ValueError(
        "NEURALMAG_GIT_URL must be an https://, ssh://, git+..., or git@... clone URL."
    )


def _apply_ref(url: str, ref: str) -> str:
    if not ref:
        return url
    if "#" in url:
        base, fragment = url.split("#", 1)
        return f"{base}@{ref}#{fragment}"
    return f"{url}@{ref}"


def main() -> int:
    raw_url = os.environ.get("NEURALMAG_GIT_URL", "").strip()
    if not raw_url:
        print("NEURALMAG_GIT_URL is not set; skipping NeuralMag installation.")
        return 0

    ref = os.environ.get("NEURALMAG_GIT_REF", "").strip()
    extras = os.environ.get("NEURALMAG_INSTALL_EXTRAS", "jax").strip()
    normalized_url = _normalize_git_url(raw_url)
    requirement = _apply_ref(f"neuralmag[{extras}] @ {normalized_url}", ref)

    print(f"Installing {requirement}")
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", requirement],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())