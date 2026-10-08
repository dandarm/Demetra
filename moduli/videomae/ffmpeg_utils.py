#!/usr/bin/env python3
"""Cross-platform FFmpeg discovery shared by inference scripts."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, Optional


def _executable_names() -> Iterable[str]:
    """Yield likely executable names, including the Windows ``.exe`` suffix."""
    # Checking both names makes this testable from a non-Windows host and also
    # supports an explicit ffmpeg.exe path copied from a Windows installation.
    return ("ffmpeg.exe", "ffmpeg")


def _is_executable(path: Path) -> bool:
    return path.is_file() and (os.name == "nt" or os.access(path, os.X_OK))


def resolve_ffmpeg_executable(ffmpeg_path: Optional[str] = None) -> Optional[str]:
    """Return a usable ffmpeg executable path, or None if not found.

    Resolution order:
    1. Explicit ``ffmpeg_path`` (dir or executable path)
    2. ``ffmpeg`` available in PATH
    3. ``FFMPEG`` or ``IMAGEIO_FFMPEG_EXE`` environment variable
    4. Optional project-local binary under ``tools/ffmpeg``
    """
    if ffmpeg_path:
        cand = Path(ffmpeg_path).expanduser().resolve()
        if _is_executable(cand):
            return str(cand)
        if cand.is_dir():
            for name in _executable_names():
                exe = cand / name
                if _is_executable(exe):
                    return str(exe)

    for name in _executable_names():
        which = shutil.which(name)
        if which:
            return which

    for env_name in ("FFMPEG", "IMAGEIO_FFMPEG_EXE"):
        configured = os.environ.get(env_name)
        if configured:
            candidate = Path(configured).expanduser()
            if _is_executable(candidate):
                return str(candidate.resolve())

    project_root = Path(__file__).resolve().parents[2]
    for name in _executable_names():
        candidate = project_root / "tools" / "ffmpeg" / name
        if _is_executable(candidate):
            return str(candidate)
    return None


def ensure_ffmpeg_in_path(ffmpeg_path: Optional[str] = None) -> None:
    resolved = resolve_ffmpeg_executable(ffmpeg_path)
    if resolved is None:
        return
    ffmpeg_dir = str(Path(resolved).resolve().parent)
    if ffmpeg_dir:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        return
