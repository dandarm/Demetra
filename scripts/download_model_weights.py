#!/usr/bin/env python3
"""Download and verify the public DeMeTrA inference checkpoints from Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import Request, urlopen


RECORD_ID = "23257278"
FIRSTPASS_NAME = "firstpass_model.ckpt"
FIRSTPASS_SHA256 = "0a841577b376a077cf9eb7856f5168f4be2043066779e241203fe49b3e0c48fa"
TRACKING_NAME = "checkpoint_new_tracking2_model_only.pth"
TRACKING_SHA256 = "49097d5cff6b19a86731d24a41a0391b8672099110a0537de606c2265b9b13c6"
PART_SHA256 = (
    "a7b7af15b7c91151b19a78a6e6eb0d840e890aa762d84f0916023a597d56687f",
    "1d9116c3b3192ad684f25ab66ce3e81da8b50992cdb3adf3a154061d2d08c9ca",
    "b0384379bba1dd56d48f40643eb198177e1e2d9b0e96468d28e0d81384b17f5a",
    "d08a31c4e36383f650541461852c3b37a98d53458cba50f62a2c1057e672cc4a",
    "5e64e04d64d25a598e6142df1d14fb837c6a87cb097f95ae7f232d61cf476a3a",
    "110661f555c4d11f589d1f7c9fee603c46495ac92477b7246f495afe6e12b319",
    "6184d53833efba3f233be21f032dd91fe7dffa5d2bf9249cea91cdade2b4d152",
    "1e2f2aef8f54d0f887c292e93d199e3a06d29aedc64d79bf476b064d5c0df42d",
    "5f43a419c16151dcbc29ab12020fbb2713f652ff843a52335954dc302758a778",
    "16216e04617a4b2745cc49992e6f1fe78676f517ffb70410ca1c9608f4f2e3da",
)
CHUNK_SIZE = 1024 * 1024


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and checksum(destination) == expected_sha256:
        print(f"Verified existing {destination.name}")
        return

    url = f"https://zenodo.org/records/{RECORD_ID}/files/{name}?download=1"
    temporary = destination.with_name(destination.name + ".download")
    digest = hashlib.sha256()
    print(f"Downloading {name}", flush=True)
    request = Request(url, headers={"User-Agent": "DeMeTrA-model-downloader/1.0"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {name}; download is incomplete or corrupt")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def assemble_tracking(directory: Path) -> None:
    destination = directory / TRACKING_NAME
    if destination.is_file() and checksum(destination) == TRACKING_SHA256:
        print(f"Verified existing {destination.name}")
        return

    parts = []
    for index, expected_sha256 in enumerate(PART_SHA256):
        name = f"{TRACKING_NAME}.{index:03d}.part"
        path = directory / name
        download(name, path, expected_sha256)
        parts.append(path)

    temporary = destination.with_name(destination.name + ".assembling")
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as output:
            for part in parts:
                with part.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
                        output.write(chunk)
                        digest.update(chunk)
        if digest.hexdigest() != TRACKING_SHA256:
            raise ValueError("SHA-256 mismatch for reconstructed tracking checkpoint")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    for part in parts:
        part.unlink()
    print(f"Ready: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "trained_models",
        help="Directory for the verified checkpoints (default: <repo>/trained_models)",
    )
    args = parser.parse_args()
    directory = args.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    download(FIRSTPASS_NAME, directory / FIRSTPASS_NAME, FIRSTPASS_SHA256)
    assemble_tracking(directory)


if __name__ == "__main__":
    main()
