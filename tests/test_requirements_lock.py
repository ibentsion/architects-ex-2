"""requirements.txt and requirements.lock must not drift apart.

The GPU node builds its venv from requirements.lock and nothing else
(cloud/setup_node.sh). A dependency added to requirements.txt but not to the
lock therefore exists on every laptop and on no node — which is exactly how
`python-multipart` shipped a bridge whose every form POST 500'd in the cloud
while passing the whole suite locally.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Strip extras and version specifiers: "langchain~=1.3" -> "langchain",
#: "uvicorn[standard]>=0.30" -> "uvicorn".
_NAME = re.compile(r"^([A-Za-z0-9._-]+)")


def _normalize(name: str) -> str:
    """PEP 503: names compare case-insensitively with -/_/. equivalent."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _NAME.match(line)
        if match:
            names.add(_normalize(match.group(1)))
    return names


def test_every_direct_requirement_is_pinned_in_the_lock():
    missing = _declared(REPO_ROOT / "requirements.txt") - _declared(
        REPO_ROOT / "requirements.lock"
    )
    assert not missing, (
        f"in requirements.txt but not requirements.lock: {sorted(missing)}. "
        "The cloud node installs only the lock, so these would be absent there."
    )
