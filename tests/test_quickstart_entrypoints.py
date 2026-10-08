"""Regression tests for the commands documented in the root quick start."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ffmpeg_helper_imports_and_handles_missing_binary() -> None:
    sys.path.insert(0, str(REPO_ROOT / "moduli" / "videomae"))
    from ffmpeg_utils import resolve_ffmpeg_executable

    resolved = resolve_ffmpeg_executable(str(REPO_ROOT / "does-not-exist"))
    assert resolved is None or Path(resolved).is_file()


def test_documented_entrypoints_expose_help() -> None:
    for relative_path in (
        "scripts/predict_firstpass_and_track_from_folder.py",
        "scripts/download_and_track_range.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / relative_path), "--help"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()
