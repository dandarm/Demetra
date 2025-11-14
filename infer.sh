#!/bin/bash
set -euo pipefail

# esegue l'inferenza con i parametri suggeriti nel README
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_PATH="config/default.yml"
CHECKPOINT_PATH="outputs/runs/exp1/best.ckpt"
OUT_DIR="outputs/preds"
PRESENCE_THRESHOLD="0.5"
ROI_BASE_RADIUS_PX="128"
ROI_SIGMA_MULTIPLIER="2.0"

mkdir -p "$OUT_DIR"

exec python -m src.cyclone_locator.infer \
  --config "$CONFIG_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --out_dir "$OUT_DIR" \
  --presence_threshold "$PRESENCE_THRESHOLD" \
  --roi_base_radius_px "$ROI_BASE_RADIUS_PX" \
  --roi_sigma_multiplier "$ROI_SIGMA_MULTIPLIER"
