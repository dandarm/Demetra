from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


START = datetime(2020, 4, 19, 17, 0)
END = datetime(2020, 4, 23, 0, 0)
STEP = timedelta(minutes=5)
FPS = 8.0

TRACKS = {
    "7001694": {
        "source": "CL7",
        "start": START,
        "end": END,
        "color": (0, 255, 0),
    },
    "3008275": {
        "source": "CL3",
        "start": datetime(2020, 4, 21, 10, 0),
        "end": datetime(2020, 4, 22, 13, 0),
        "color": (255, 0, 255),
    },
}

SOURCE_ROOT = Path("/media/isacDisk2/source_dataset/2020/04")
TRACK_CSV = Path(
    "/media/isacDisk1/Demetra/moduli/videomae/"
    "medicane_data_input/all_manos_CL_pixel.csv"
)
OUTPUT_DIR = Path(
    "/media/isacDisk1/Demetra/notebooks/batch_outputs/"
    "cyclone_7001694_full_lifetime_full_mediterranean"
)
OUTPUT_VIDEO = OUTPUT_DIR / "cyclone_7001694_full_lifetime_full_mediterranean.mp4"
OUTPUT_MANIFEST = OUTPUT_DIR / "frames_manifest.csv"
TIMESTAMP_FONT = Path(
    "/media/isacDisk1/Demetra/moduli/videomae/digital-7 (italic).ttf"
)


def load_hourly_tracks() -> dict[
    str, dict[datetime, tuple[float, float, float, float, float]]
]:
    tracks: dict[
        str, dict[datetime, tuple[float, float, float, float, float]]
    ] = {cyclone_id: {} for cyclone_id in TRACKS}
    with TRACK_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cyclone_id = row["id_cyc_unico"]
            if cyclone_id not in TRACKS:
                continue
            timestamp = datetime.fromisoformat(row["time"])
            track_config = TRACKS[cyclone_id]
            if track_config["start"] <= timestamp <= track_config["end"]:
                tracks[cyclone_id][timestamp] = (
                    float(row["x_pix"]),
                    float(row["y_pix"]),
                    float(row["lat"]),
                    float(row["lon"]),
                    float(row["pressure"]),
                )
    for cyclone_id, track_config in TRACKS.items():
        required: set[datetime] = set()
        required_time = track_config["start"]
        while required_time <= track_config["end"]:
            required.add(required_time)
            required_time += timedelta(hours=1)
        missing = sorted(required.difference(tracks[cyclone_id]))
        if missing:
            raise RuntimeError(
                f"Coordinate orarie mancanti per {cyclone_id}: {missing}"
            )
    return tracks


def interpolate_track(
    timestamp: datetime,
    hourly: dict[datetime, tuple[float, float, float, float, float]],
) -> tuple[float, float, float, float, float]:
    hour = timestamp.replace(minute=0, second=0, microsecond=0)
    if timestamp == hour:
        return hourly[hour]
    next_hour = hour + timedelta(hours=1)
    fraction = timestamp.minute / 60.0
    return tuple(
        start + fraction * (end - start)
        for start, end in zip(hourly[hour], hourly[next_hour])
    )


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(TIMESTAMP_FONT), size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly_tracks = load_hourly_tracks()
    timestamps: list[datetime] = []
    timestamp = START
    while timestamp <= END:
        timestamps.append(timestamp)
        timestamp += STEP

    source_paths = [
        SOURCE_ROOT / f"airmass_rgb_{timestamp:%Y%m%d_%H%M}.png"
        for timestamp in timestamps
    ]
    missing_sources = [path for path in source_paths if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(f"Frame mancanti: {missing_sources[:10]}")

    with Image.open(source_paths[0]) as first:
        width, height = first.size
    if (width, height) != (1290, 420):
        raise RuntimeError(f"Dimensioni inattese: {(width, height)}")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Impossibile inizializzare il writer MP4")

    timestamp_font = load_font(30)
    track_points: dict[str, list[tuple[int, int]]] = {
        cyclone_id: [] for cyclone_id in TRACKS
    }
    manifest_rows: list[dict[str, object]] = []

    try:
        for index, (timestamp, source_path) in enumerate(
            zip(timestamps, source_paths), start=1
        ):
            with Image.open(source_path) as source:
                frame = source.convert("RGB")
            draw = ImageDraw.Draw(frame)
            frame_states: dict[
                str, tuple[float, float, float, float, float] | None
            ] = {}

            for cyclone_id, track_config in TRACKS.items():
                if not track_config["start"] <= timestamp <= track_config["end"]:
                    frame_states[cyclone_id] = None
                    continue

                state = interpolate_track(timestamp, hourly_tracks[cyclone_id])
                frame_states[cyclone_id] = state
                x, y, _lat, _lon, _pressure = state
                track_points[cyclone_id].append((round(x), round(y)))
                color = track_config["color"]

                if len(track_points[cyclone_id]) > 1:
                    draw.line(
                        track_points[cyclone_id],
                        fill=(0, 0, 0),
                        width=7,
                        joint="curve",
                    )
                    draw.line(
                        track_points[cyclone_id],
                        fill=color,
                        width=3,
                        joint="curve",
                    )

                radius_outer = 9
                radius_inner = 6
                draw.ellipse(
                    (
                        x - radius_outer,
                        y - radius_outer,
                        x + radius_outer,
                        y + radius_outer,
                    ),
                    fill=(0, 0, 0),
                )
                draw.ellipse(
                    (
                        x - radius_inner,
                        y - radius_inner,
                        x + radius_inner,
                        y + radius_inner,
                    ),
                    fill=color,
                )

            time_label = timestamp.strftime("%H:%M %d-%m-%Y UTC")
            time_bbox = draw.textbbox((0, 0), time_label, font=timestamp_font)
            text_width = time_bbox[2] - time_bbox[0]
            text_height = time_bbox[3] - time_bbox[1]
            draw.text(
                (width - text_width - 18, height - text_height - 18),
                time_label,
                font=timestamp_font,
                fill=(255, 80, 80),
            )

            if timestamp == datetime(2020, 4, 22, 12, 0):
                frame.save(OUTPUT_DIR / "preview_full_lifetime_20200422_1200.png")

            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
            cl7_state = frame_states["7001694"]
            cl3_state = frame_states["3008275"]
            if cl7_state is None:
                raise RuntimeError("La traccia CL7 deve coprire l'intero video")
            x, y, lat, lon, pressure = cl7_state
            manifest_rows.append(
                {
                    "frame": index,
                    "timestamp": timestamp.isoformat(sep=" "),
                    "source_path": str(source_path),
                    "x_pix": round(x, 3),
                    "y_pix": round(y, 3),
                    "lat": round(lat, 4),
                    "lon": round(lon, 4),
                    "pressure_hpa": round(pressure, 2),
                    "cl3_3008275_active": int(cl3_state is not None),
                    "cl3_3008275_x_pix": (
                        round(cl3_state[0], 3) if cl3_state is not None else ""
                    ),
                    "cl3_3008275_y_pix": (
                        round(cl3_state[1], 3) if cl3_state is not None else ""
                    ),
                    "track_method": "hourly linear interpolation",
                }
            )
            if index % 48 == 0 or index == len(timestamps):
                print(f"Rendered {index}/{len(timestamps)} frames", flush=True)
    finally:
        writer.release()

    with OUTPUT_MANIFEST.open("w", newline="") as handle:
        fieldnames = list(manifest_rows[0])
        manifest_writer = csv.DictWriter(handle, fieldnames=fieldnames)
        manifest_writer.writeheader()
        manifest_writer.writerows(manifest_rows)

    print(OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
