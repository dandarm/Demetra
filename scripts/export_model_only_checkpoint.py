#!/usr/bin/env python3
"""Create a portable, inference-only DeMeTrA tracking checkpoint.

The training checkpoint contains optimizer and run state and can be several
gigabytes larger than the tensors required by inference.  This tool extracts
only the model state dictionary, stores it in the format accepted by
``inference_tracking.load_checkpoint``, and prints a SHA-256 for release notes.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Checkpoint di training esistente.")
    parser.add_argument("--output", required=True, help="Destinazione del checkpoint model-only.")
    parser.add_argument("--force", action="store_true", help="Sovrascrive --output se esiste.")
    return parser.parse_args()


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict) and all(
                isinstance(value, torch.Tensor) for value in candidate.values()
            ):
                return candidate
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Il checkpoint non contiene un state_dict/model/module composto da tensori.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint di input non trovato: {source}")
    if destination.exists() and not args.force:
        raise FileExistsError(f"Output gia presente: {destination}. Usa --force per sovrascriverlo.")

    checkpoint = torch.load(source, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    # A detached CPU copy breaks references to optimizer/run objects and keeps
    # the serialized artifact portable across machines and Python paths.
    model_only = {key: value.detach().cpu().contiguous() for key, value in state_dict.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "demetra-tracking-model-only-v1",
            "model": model_only,
        },
        destination,
    )
    print(f"Saved model-only checkpoint: {destination}")
    print(f"Size: {destination.stat().st_size} bytes")
    print(f"SHA256: {sha256(destination)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
