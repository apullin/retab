"""The public repo must not carry hard-coded home-directory paths or reach-ins to sibling checkouts."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ["/Users/", "/home/", "PIPELINE_ROOT", "U55_ROOT", "VELA_ROOT", "../executorch"]


def test_tracked_files_have_no_personal_paths():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    hits = []
    for needle in FORBIDDEN:
        proc = subprocess.run(
            ["git", "-C", str(ROOT), "grep", "-nF", "--untracked", needle, "--", ".", ":!tests/test_no_personal_paths.py"],
            capture_output=True,
            text=True,
        )
        hits += proc.stdout.splitlines()
    assert not hits, "personal paths or sibling reach-ins found:\n" + "\n".join(hits[:20])
