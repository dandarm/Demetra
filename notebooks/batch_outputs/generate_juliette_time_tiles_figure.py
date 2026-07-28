from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SOURCE_ROOT = Path("/media/isacDisk2/source_dataset/2023/02")
OUTPUT_ROOT = Path("/media/isacDisk1/Demetra/notebooks/batch_outputs")
OUTPUT_PNG = OUTPUT_ROOT / "juliette_time_tiles_perspective_20230228_0220.png"
OUTPUT_PDF = OUTPUT_ROOT / "juliette_time_tiles_perspective_20230228_0220.pdf"

TILE_SIZE = 224
OFFSET_X = 213
OFFSET_Y = 0
CANVAS_WIDTH = 2480
CANVAS_HEIGHT = 1664

# Hourly samples beginning at 02:20. The 04:20 sample is intentionally
# omitted and represented by the ellipsis to emphasize temporal evolution.
TIMESTAMPS = (
    datetime(2023, 2, 28, 2, 20),
    datetime(2023, 2, 28, 3, 20),
    datetime(2023, 2, 28, 5, 20),
    datetime(2023, 2, 28, 6, 20),
)

# More strongly folded perspective planes, ordered from earliest/foreground
# to latest/background. A deliberate gap between planes 2 and 3 hosts the
# ellipsis that represents the omitted intermediate frames.
QUADS = (
    # Strongest fold: vertical sides, pronounced slope of top/bottom edges.
    np.float32(((60, 65), (580, 305), (580, 1260), (60, 1020))),
    np.float32(((680, 150), (1160, 335), (1160, 1165), (680, 980))),
    # Later panels repeat the size and perspective geometry of panel 2.
    np.float32(((1390, 150), (1870, 335), (1870, 1165), (1390, 980))),
    np.float32(((1960, 150), (2440, 335), (2440, 1165), (1960, 980))),
)


def load_tile(timestamp: datetime) -> np.ndarray:
    source_path = SOURCE_ROOT / f"airmass_rgb_{timestamp:%Y%m%d_%H%M}.png"
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    with Image.open(source_path) as source:
        tile = source.convert("RGB").crop(
            (OFFSET_X, OFFSET_Y, OFFSET_X + TILE_SIZE, OFFSET_Y + TILE_SIZE)
        )

    return np.asarray(tile, dtype=np.uint8)


def warp_tile(tile: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_corners = np.float32(
        ((0, 0), (TILE_SIZE - 1, 0), (TILE_SIZE - 1, TILE_SIZE - 1), (0, TILE_SIZE - 1))
    )
    transform = cv2.getPerspectiveTransform(source_corners, quad)
    warped = cv2.warpPerspective(
        tile,
        transform,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    source_mask = np.full((TILE_SIZE, TILE_SIZE), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(
        source_mask,
        transform,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, mask


def composite_panel(canvas: np.ndarray, tile: np.ndarray, quad: np.ndarray) -> None:
    warped, mask = warp_tile(tile, quad)

    # Restrained soft shadow for depth while retaining a clean paper aesthetic.
    shadow_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=10, sigmaY=10)
    shadow_transform = np.float32([[1, 0, 9], [0, 1, 12]])
    shadow_mask = cv2.warpAffine(
        shadow_mask,
        shadow_transform,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    shadow_alpha = (shadow_mask.astype(np.float32) / 255.0 * 0.16)[..., None]
    canvas[:] = (
        canvas.astype(np.float32) * (1.0 - shadow_alpha)
        + np.full_like(canvas, 105, dtype=np.float32) * shadow_alpha
    ).astype(np.uint8)

    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    canvas[:] = (
        warped.astype(np.float32) * alpha
        + canvas.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)

    cv2.polylines(
        canvas,
        [np.rint(quad).astype(np.int32)],
        isClosed=True,
        color=(70, 70, 70),
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def draw_time_arrow(canvas: np.ndarray) -> np.ndarray:
    figure = Image.fromarray(canvas)
    draw = ImageDraw.Draw(figure)
    serif_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
    try:
        time_font = ImageFont.truetype(str(serif_path), 82)
    except OSError:
        time_font = ImageFont.load_default()

    arrow_y = 1540
    arrow_start = 390
    arrow_end = 2090
    arrow_head_length = 58
    arrow_head_half_height = 27
    line_width = 5

    draw.line(
        (arrow_start, arrow_y, arrow_end - arrow_head_length, arrow_y),
        fill=(0, 0, 0),
        width=line_width,
    )
    draw.polygon(
        (
            (arrow_end, arrow_y),
            (arrow_end - arrow_head_length, arrow_y - arrow_head_half_height),
            (arrow_end - arrow_head_length, arrow_y + arrow_head_half_height),
        ),
        fill=(0, 0, 0),
    )

    label = "time"
    bbox = draw.textbbox((0, 0), label, font=time_font)
    label_width = bbox[2] - bbox[0]
    draw.text(
        ((arrow_start + arrow_end - label_width) / 2, 1428),
        label,
        font=time_font,
        fill=(0, 0, 0),
    )

    # Omitted 04:20 hourly sample between the second and third panels.
    ellipsis_y = 700
    ellipsis_radius = 12
    for ellipsis_x in (1235, 1275, 1315):
        draw.ellipse(
            (
                ellipsis_x - ellipsis_radius,
                ellipsis_y - ellipsis_radius,
                ellipsis_x + ellipsis_radius,
                ellipsis_y + ellipsis_radius,
            ),
            fill=(35, 35, 35),
        )
    return np.asarray(figure, dtype=np.uint8)


def main() -> None:
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 255, dtype=np.uint8)
    tiles = [load_tile(timestamp) for timestamp in TIMESTAMPS]

    # Paint background panels first so earlier tiles overlap later ones as in
    # the reference composition.
    for tile, quad in reversed(list(zip(tiles, QUADS))):
        composite_panel(canvas, tile, quad)

    canvas = draw_time_arrow(canvas)
    output = Image.fromarray(canvas)
    output.save(OUTPUT_PNG, dpi=(300, 300), optimize=True)
    output.save(OUTPUT_PDF, resolution=300.0)
    print(OUTPUT_PNG)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
