#!/usr/bin/env python3
"""Download EUMETSAT HRSEVIRI ZIP products for a time range.

This script is intentionally self-contained and does not import from other
modules in the DeMeTra repository. It only depends on `eumdac` and `requests`
for the EUMETSAT Data Store download workflow.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import logging
import os
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from threading import local
from typing import Iterable

import eumdac
import requests


DEFAULT_COLLECTION_ID = "EO:EUM:DAT:MSG:MSG15-RSS"
TIMESTAMP_FROM_PRODUCT_RE = re.compile(r"-(\d{14})\.\d+Z-NA$")
THREAD_LOCAL = local()
LOG = logging.getLogger("download_eumetsat_range")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scarica prodotti HRSEVIRI/MSG da EUMETSAT Data Store in un intervallo temporale. "
            "Lo script salva solo gli ZIP e non costruisce compositi airmassRGB."
        )
    )
    parser.add_argument("--start", required=True, help="Data/ora iniziale.")
    parser.add_argument("--end", required=True, help="Data/ora finale.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Cartella dove salvare gli ZIP scaricati.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_ID,
        help=f"Collection ID EUMETSAT (default: {DEFAULT_COLLECTION_ID}).",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Numero di download concorrenti.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Numero massimo di retry per ciascun prodotto.",
    )
    parser.add_argument(
        "--read-timeout",
        type=int,
        default=180,
        help="Read timeout in secondi per ogni stream HTTP.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=30,
        help="Connect timeout in secondi per ogni richiesta HTTP.",
    )
    parser.add_argument(
        "--overwrite-invalid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Se attivo, elimina eventuali ZIP locali corrotti e li riscarica.",
    )
    parser.add_argument(
        "--manifest-name",
        default="download_manifest.csv",
        help="Nome del CSV manifest scritto nella cartella di output.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Livello di log.",
    )
    return parser.parse_args()


def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def parse_user_datetime(value: str, *, is_end: bool) -> datetime:
    raw = value.strip()
    explicit_formats = (
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y_%H:%M",
        "%d-%m-%Y %H%M",
        "%d-%m-%Y_%H%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d_%H:%M",
        "%Y-%m-%d %H%M",
        "%Y-%m-%d_%H%M",
        "%Y%m%d_%H%M",
        "%Y%m%d%H%M",
        "%d-%m-%Y",
        "%Y-%m-%d",
    )
    for fmt in explicit_formats:
        try:
            parsed = datetime.strptime(raw, fmt)
            if _looks_like_date_only(raw):
                return parsed.replace(hour=23, minute=55) if is_end else parsed.replace(hour=0, minute=0)
            return parsed
        except ValueError:
            continue
    raise ValueError(
        f"Formato data non riconosciuto: {value}. "
        "Usa ad esempio '15-03-2026 12:30' o '20260315_1230'."
    )


def _looks_like_date_only(value: str) -> bool:
    return bool(re.fullmatch(r"\d{2}-\d{2}-\d{4}", value) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))


def require_credentials() -> tuple[str, str]:
    consumer_key = os.getenv("EUMETSAT_CONSUMER_KEY")
    consumer_secret = os.getenv("EUMETSAT_CONSUMER_SECRET")
    if not consumer_key or not consumer_secret:
        raise RuntimeError(
            "Credenziali EUMETSAT mancanti. "
            "Imposta EUMETSAT_CONSUMER_KEY e EUMETSAT_CONSUMER_SECRET."
        )
    return consumer_key, consumer_secret


def is_valid_eumetsat_zip(zip_path: Path) -> bool:
    if not zip_path.exists() or zip_path.stat().st_size <= 0:
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return any(name.lower().endswith(".nat") for name in zf.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def parse_product_start(product_id: str) -> datetime | None:
    match = TIMESTAMP_FROM_PRODUCT_RE.search(product_id)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def sort_product_ids(product_ids: Iterable[str]) -> list[str]:
    return sorted(
        product_ids,
        key=lambda product_id: (parse_product_start(product_id) or datetime.min, product_id),
    )


def get_thread_datastore(consumer_key: str, consumer_secret: str) -> eumdac.DataStore:
    datastore = getattr(THREAD_LOCAL, "datastore", None)
    credentials = getattr(THREAD_LOCAL, "credentials", None)
    current_credentials = (consumer_key, consumer_secret)
    if datastore is not None and credentials == current_credentials:
        return datastore

    token = eumdac.AccessToken(current_credentials)
    datastore = eumdac.DataStore(token)
    THREAD_LOCAL.datastore = datastore
    THREAD_LOCAL.credentials = current_credentials
    return datastore


def search_products(
    consumer_key: str,
    consumer_secret: str,
    collection_id: str,
    start_dt: datetime,
    end_dt: datetime,
) -> list[str]:
    token = eumdac.AccessToken((consumer_key, consumer_secret))
    datastore = eumdac.DataStore(token)
    collection = datastore.get_collection(collection_id)

    products = list(collection.search(dtstart=start_dt, dtend=end_dt))
    product_ids = sort_product_ids(str(product) for product in products)
    LOG.info(
        "Prodotti trovati in %s per %s -> %s: %d",
        collection_id,
        start_dt,
        end_dt,
        len(product_ids),
    )
    return product_ids


def download_products(
    product_ids: Iterable[str],
    output_dir: Path,
    collection_id: str,
    consumer_key: str,
    consumer_secret: str,
    download_workers: int,
    retries: int,
    connect_timeout: int,
    read_timeout: int,
    overwrite_invalid: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_paths: list[Path] = []
    pending_ids: list[str] = []

    for product_id in product_ids:
        out_path = output_dir / f"{product_id}.zip"
        if is_valid_eumetsat_zip(out_path):
            valid_paths.append(out_path)
            LOG.debug("Cache hit %s", out_path.name)
            continue
        if out_path.exists():
            if not overwrite_invalid:
                raise RuntimeError(
                    f"Trovato ZIP locale non valido e overwrite disattivato: {out_path}"
                )
            LOG.warning("ZIP locale non valido, elimino e riscarico %s", out_path.name)
            out_path.unlink()
        pending_ids.append(product_id)

    LOG.info(
        "ZIP validi gia presenti: %d | da scaricare: %d",
        len(valid_paths),
        len(pending_ids),
    )
    if not pending_ids:
        return sorted(valid_paths)

    def download_one(product_id: str) -> Path:
        out_path = output_dir / f"{product_id}.zip"
        tmp_path = out_path.with_suffix(".zip.part")
        last_error: Exception | None = None

        for attempt in range(1, retries + 2):
            try:
                datastore = get_thread_datastore(consumer_key, consumer_secret)
                product = datastore.get_product(
                    product_id=product_id,
                    collection_id=collection_id,
                )
                url = product.datastore.urls.get(
                    "datastore",
                    "download product",
                    vars={
                        "collection_id": collection_id,
                        "product_id": product_id,
                    },
                )
                headers = eumdac.common.headers.copy()
                LOG.info(
                    "Downloading %s [attempt %d/%d]",
                    out_path.name,
                    attempt,
                    retries + 1,
                )
                with requests.get(
                    url,
                    auth=product.datastore.token.auth,
                    stream=True,
                    headers=headers,
                    timeout=(connect_timeout, read_timeout),
                ) as response:
                    response.raise_for_status()
                    expected_bytes = response.headers.get("Content-Length")
                    with tmp_path.open("wb") as file_out:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                file_out.write(chunk)

                if expected_bytes is not None and tmp_path.stat().st_size != int(expected_bytes):
                    raise RuntimeError(
                        f"Download incompleto per {out_path.name}: "
                        f"{tmp_path.stat().st_size} != {expected_bytes} byte"
                    )

                tmp_path.replace(out_path)
                if not is_valid_eumetsat_zip(out_path):
                    raise RuntimeError(f"ZIP corrotto o privo di .nat: {out_path.name}")

                LOG.info("Done %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)
                return out_path
            except Exception as exc:
                last_error = exc
                if tmp_path.exists():
                    tmp_path.unlink()
                if attempt > retries:
                    break
                sleep_seconds = min(30, 2 ** (attempt - 1))
                LOG.warning(
                    "Retry %d su %s dopo errore: %s",
                    attempt,
                    out_path.name,
                    exc,
                )
                time.sleep(sleep_seconds)

        raise RuntimeError(f"Download fallito per {product_id}: {last_error}")

    all_paths = list(valid_paths)
    workers = max(1, int(download_workers))
    LOG.info(
        "Avvio download concorrente con %d worker, %d retry max, timeout connect/read %ss/%ss",
        workers,
        retries,
        connect_timeout,
        read_timeout,
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="eumdac",
    ) as executor:
        future_map = {
            executor.submit(download_one, product_id): (index, product_id)
            for index, product_id in enumerate(pending_ids, start=1)
        }
        for future in concurrent.futures.as_completed(future_map):
            index, product_id = future_map[future]
            out_path = future.result()
            LOG.info("[%d/%d] Completed %s", index, len(pending_ids), product_id)
            all_paths.append(out_path)

    return sorted(all_paths)


def write_manifest(manifest_path: Path, zip_paths: Iterable[Path]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["product_id", "product_start_utc", "zip_path", "size_bytes"])
        for zip_path in sorted(zip_paths):
            product_id = zip_path.stem
            product_start = parse_product_start(product_id)
            writer.writerow(
                [
                    product_id,
                    product_start.isoformat() if product_start else "",
                    str(zip_path.resolve()),
                    zip_path.stat().st_size,
                ]
            )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    start_dt = parse_user_datetime(args.start, is_end=False)
    end_dt = parse_user_datetime(args.end, is_end=True)
    if end_dt < start_dt:
        raise SystemExit("--end deve essere maggiore o uguale a --start.")

    consumer_key, consumer_secret = require_credentials()
    output_dir = Path(args.output_dir).expanduser().resolve()

    LOG.info("Output dir: %s", output_dir)
    LOG.info("Collection: %s", args.collection)

    product_ids = search_products(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        collection_id=args.collection,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    if not product_ids:
        LOG.warning("Nessun prodotto trovato nell'intervallo richiesto.")
        return 0

    zip_paths = download_products(
        product_ids=product_ids,
        output_dir=output_dir,
        collection_id=args.collection,
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        download_workers=args.download_workers,
        retries=args.retries,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        overwrite_invalid=args.overwrite_invalid,
    )

    manifest_path = output_dir / args.manifest_name
    write_manifest(manifest_path, zip_paths)
    LOG.info("Download completato: %d ZIP disponibili", len(zip_paths))
    LOG.info("Manifest scritto in %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
