from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SOURCE_ROOT = Path("/media/isacDisk2/source_dataset")
BATCH_ROOT = Path("/media/isacDisk1/Demetra/notebooks/batch_outputs")
TIMESTAMP_FONT = Path(
    "/media/isacDisk1/Demetra/moduli/videomae/digital-7 (italic).ttf"
)

FRAME_COUNT = 16
STEP = timedelta(minutes=5)
TILE_SIZE = 224


@dataclass(frozen=True)
class RunConfig:
    start: datetime
    output_name: str
    hourly_centers: dict[datetime, tuple[float, float]]
    offset_x: int = 639
    offset_y: int = 196


RUNS = (
    RunConfig(
        start=datetime(2020, 9, 14, 17, 0),
        output_name="ianos_20200914_1700_videotile",
        hourly_centers={
            datetime(2020, 9, 14, 17, 0): (744.0, 361.0),
            datetime(2020, 9, 14, 18, 0): (742.0, 358.0),
            datetime(2020, 9, 14, 19, 0): (741.0, 355.0),
        },
    ),
    RunConfig(
        start=datetime(2020, 9, 15, 17, 0),
        output_name="ianos_20200915_1700_videotile",
        hourly_centers={
            datetime(2020, 9, 15, 17, 0): (695.0, 328.0),
            datetime(2020, 9, 15, 18, 0): (697.0, 326.0),
            datetime(2020, 9, 15, 19, 0): (698.0, 323.0),
        },
    ),
    RunConfig(
        start=datetime(2020, 9, 17, 4, 45),
        output_name="ianos_20200917_0445_videotile",
        hourly_centers={
            datetime(2020, 9, 17, 4, 0): (745.0, 217.0),
            datetime(2020, 9, 17, 5, 0): (747.0, 216.0),
            datetime(2020, 9, 17, 6, 0): (749.0, 215.0),
            datetime(2020, 9, 17, 7, 0): (752.0, 215.0),
        },
    ),
    RunConfig(
        start=datetime(2020, 9, 19, 21, 0),
        output_name="ianos_20200919_2100_videotile",
        hourly_centers={
            datetime(2020, 9, 19, 21, 0): (906.0, 296.0),
            datetime(2020, 9, 19, 22, 0): (912.0, 299.0),
            datetime(2020, 9, 19, 23, 0): (918.0, 302.0),
        },
        offset_x=852,
        offset_y=196,
    ),
)


def interpolated_center(
    timestamp: datetime,
    hourly_centers: dict[datetime, tuple[float, float]],
) -> tuple[float, float]:
    hour = timestamp.replace(minute=0, second=0, microsecond=0)
    next_hour = hour + timedelta(hours=1)
    x0, y0 = hourly_centers[hour]
    x1, y1 = hourly_centers[next_hour]
    fraction = timestamp.minute / 60.0
    return x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0)


def draw_timestamp(image: Image.Image, timestamp: datetime) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(str(TIMESTAMP_FONT), 14)
    except OSError:
        font = ImageFont.load_default()
    label = timestamp.strftime(" %H:%M %d-%m-%Y")
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = image.width - text_width - 6
    y = image.height - text_height - 6
    draw.text((x, y), label, font=font, fill=(255, 80, 80))


def save_high_quality_gif(frames: list[Image.Image], output_path: Path) -> None:
    """Use one animation-wide palette while preserving overlay colors exactly."""
    columns = 4
    rows = (len(frames) + columns - 1) // columns
    palette_source = Image.new("RGB", (TILE_SIZE * columns, TILE_SIZE * rows))
    for index, frame in enumerate(frames):
        x = (index % columns) * TILE_SIZE
        y = (index // columns) * TILE_SIZE
        palette_source.paste(frame, (x, y))

    palette_image = palette_source.convert(
        "P", palette=Image.Palette.ADAPTIVE, colors=253
    )
    palette = palette_image.getpalette()
    if palette is None:
        raise RuntimeError("Palette GIF non generata")
    palette = (palette + [0] * 768)[:768]
    reserved_colors = ((0, 255, 0), (0, 0, 0), (255, 80, 80))
    for palette_index, color in zip(range(253, 256), reserved_colors):
        start = palette_index * 3
        palette[start : start + 3] = color
    palette_image.putpalette(palette)

    quantized = [
        frame.quantize(palette=palette_image, dither=Image.Dither.FLOYDSTEINBERG)
        for frame in frames
    ]
    quantized[0].save(
        output_path,
        save_all=True,
        append_images=quantized[1:],
        duration=200,
        loop=0,
        optimize=False,
        disposal=2,
    )


def generate(config: RunConfig) -> Path:
    output_root = BATCH_ROOT / config.output_name
    frames_dir = output_root / (
        f"{config.start:%d-%m-%Y_%H%M}_{config.offset_x}_{config.offset_y}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    timestamped_gif_frames: list[Image.Image] = []
    tracked_gif_frames: list[Image.Image] = []
    last_tracked_static: Image.Image | None = None
    manifest_rows: list[dict[str, object]] = []

    for index in range(FRAME_COUNT):
        timestamp = config.start + index * STEP
        source = (
            SOURCE_ROOT
            / f"{timestamp:%Y}"
            / f"{timestamp:%m}"
            / f"airmass_rgb_{timestamp:%Y%m%d_%H%M}.png"
        )
        if not source.exists():
            raise FileNotFoundError(source)

        with Image.open(source) as image:
            tile = image.convert("RGB").crop(
                (
                    config.offset_x,
                    config.offset_y,
                    config.offset_x + TILE_SIZE,
                    config.offset_y + TILE_SIZE,
                )
            )

        # Keep the scientific RGB frames free from visual overlays.
        frame_path = frames_dir / f"img_{index + 1:05d}.png"
        tile.save(frame_path)

        timestamped = tile.copy()
        draw_timestamp(timestamped, timestamp)
        timestamped_gif_frames.append(timestamped)

        center_x, center_y = interpolated_center(timestamp, config.hourly_centers)
        relative_x = center_x - config.offset_x
        relative_y = center_y - config.offset_y

        tracked = tile.copy()
        draw = ImageDraw.Draw(tracked)
        radius = 4
        draw.ellipse(
            (
                relative_x - radius,
                relative_y - radius,
                relative_x + radius,
                relative_y + radius,
            ),
            fill=(0, 255, 0),
            outline=(0, 0, 0),
            width=1,
        )
        if index == FRAME_COUNT - 1:
            last_tracked_static = tracked.copy()
        draw_timestamp(tracked, timestamp)
        tracked_gif_frames.append(tracked)

        manifest_rows.append(
            {
                "frame": index + 1,
                "timestamp": timestamp.isoformat(sep=" "),
                "source_path": str(source),
                "tile_path": str(frame_path),
                "offset_x": config.offset_x,
                "offset_y": config.offset_y,
                "center_x_global": round(center_x, 3),
                "center_y_global": round(center_y, 3),
                "center_x_tile": round(relative_x, 3),
                "center_y_tile": round(relative_y, 3),
                "center_method": "CL7 hourly linear interpolation",
            }
        )

    save_high_quality_gif(
        timestamped_gif_frames, output_root / "ianos_videotile.gif"
    )
    save_high_quality_gif(
        tracked_gif_frames, output_root / "ianos_videotile_center_track.gif"
    )
    if last_tracked_static is None:
        raise RuntimeError("Ultimo frame statico non generato")
    last_tracked_static.save(
        output_root / f"ianos_last_frame_{manifest_rows[-1]['timestamp'][:10].replace('-', '')}_{manifest_rows[-1]['timestamp'][11:16].replace(':', '')}_center_track.png"
    )

    with (output_root / "frames_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    first_center_x, first_center_y = interpolated_center(
        config.start, config.hourly_centers
    )
    with (output_root / "tracking_dataset.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "x_pix", "y_pix", "has_target"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "path": str(frames_dir),
                "x_pix": first_center_x - config.offset_x,
                "y_pix": first_center_y - config.offset_y,
                "has_target": 1,
            }
        )

    return output_root


def main() -> None:
    for config in RUNS:
        print(generate(config))


if __name__ == "__main__":
    main()
