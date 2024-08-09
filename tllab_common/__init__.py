from importlib.metadata import version
from pathlib import Path

try:
    __version__ = version(Path(__file__).parent.name)
except Exception:  # noqa
    __version__ = "unknown"

try:
    with open(Path(__file__).parent.parent / ".git" / "HEAD") as g:
        head = g.read().split(":")[1].strip()
    with open(Path(__file__).parent.parent / ".git" / head) as h:
        __git_commit_hash__ = h.read().rstrip("\n")
except Exception:  # noqa
    __git_commit_hash__ = "unknown"
