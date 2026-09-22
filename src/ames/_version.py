"""AMES version string: hand-declared release number + the commit it ran on.

The release number is declared by hand in pyproject.toml (`[project] version`)
and is read from there at runtime, so bumping 0.1 -> 0.2 takes effect on the
next run without reinstalling the editable package.

The commit is appended as a PEP 440 local version segment:

    1.0.0+207bac7          committed
    1.0.0+207bac7.dirty    uncommitted changes in the working tree

`+` and `.dirty` (rather than `-dirty`) keep the whole string a valid PEP 440
version - hyphens are not allowed in a local segment and would be normalised
to `.` anyway.
"""

import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _declared_version() -> str:
    """The hand-edited number from pyproject.toml."""
    try:
        import tomllib  # stdlib on 3.11+
    except ImportError:  # 3.10 - fall back to what was recorded at install time
        tomllib = None

    if tomllib is not None:
        try:
            with open(_PYPROJECT, "rb") as f:
                return tomllib.load(f)["project"]["version"]
        except (OSError, KeyError, ValueError):
            pass  # not a checkout, or malformed - fall through

    try:
        return _pkg_version("ames")
    except PackageNotFoundError:
        return "unknown"


def _commit() -> str:
    """Short commit hash, suffixed with `.dirty` if the tree has local edits."""
    try:
        proc = subprocess.run(
            ["git", "describe", "--always", "--dirty=.dirty"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass  # git missing, not a checkout, or the call hung
    return ""


@lru_cache(maxsize=1)
def get_version() -> str:
    """e.g. '1.0.0+207bac7.dirty', or just '1.0.0' outside a git checkout."""
    declared = _declared_version()
    commit = _commit()
    return f"{declared}+{commit}" if commit else declared


__all__ = ["get_version"]
