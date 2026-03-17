# Predict FirstPass And Track From Folder

Questo documento descrive lo script `scripts/predict_firstpass_and_track_from_folder.py`, che unisce:

1. inferenza **first-pass** su frame Mediterraneo stretched;
2. creazione di videotile HR (16 frame) centrate sul centro first-pass;
3. inferenza **tracking VideoMAE** sulle sole clip positive.

## Flusso

1. Legge i frame originali da `--input_dir` (atteso timestamp nel filename `YYYYMMDD_HHMM`).
2. Crea copie stretched SxS temporanee per il first-pass.
3. Esegue first-pass e ottiene `presence_prob`, `x_g`, `y_g`.
4. Back-projecta il centro nello spazio originale (`x_orig`, `y_orig`).
5. Costruisce clip da 16 frame con stride temporale di 15 minuti.
6. Per ogni clip positiva (`presence_prob >= --firstpass_threshold`):
   - calcola offset tile centrato sul centro first-pass;
   - salva folder tile con nome `DD-MM-YYYY_HHMM_offsetX_offsetY`;
   - salva i 16 frame crop `img_00001.png ... img_00016.png`.
7. Esegue tracking VideoMAE su queste folder.
8. Produce CSV finale per timeframe: `datetime, has_cyclone, pred_lat, pred_lon`.

## Argomenti principali

- `--input_dir`: cartella frame Mediterraneo full-resolution.
- `--output_dir`: root output pipeline.
- `--firstpass_model_path`: checkpoint first-pass.
- `--tracking_model_path`: checkpoint tracking VideoMAE.
- `--firstpass_threshold`: soglia presenza first-pass per selezione clip.
- `--firstpass_image_size`: lato immagini stretched per first-pass (default 224).
- `--num_frames`: numero frame per clip/tile (default 16).
- `--tile_size`: lato tile HR (default 224).
- `--manos_file`: GT opzionale per tracking.
- `--make_video`: genera anche il video ROI-firstpass finale.
- `--only_video`: crea solo MP4 da frame già renderizzati (richiede `--make_video`).
- `--video_name`: nome base del file video.
- `--ffmpeg_path`: path opzionale per ffmpeg.

## Output

- `output_dir/_tmp_firstpass_manifest.csv`
- `output_dir/_tmp_firstpass_predictions.csv`
- `output_dir/_tmp_firstpass_clip_candidates.csv`
- `output_dir/firstpass_tiles/<DD-MM-YYYY_HHMM_offx_offy>/img_*.png`
- `output_dir/firstpass_tiles_tracking_overlay/<DD-MM-YYYY_HHMM_offx_offy>/img_*.png` (stesse tile con dot rosso tracking su tutti i frame)
- `output_dir/_tmp_tracking_inference_predictions_tiles.csv`
- `output_dir/tracking_inference_predictions.csv` (finale timeframe)
- `output_dir/<video_name>.mp4` (se `--make_video`)
- `output_dir/anim_frames_<video_name>/frame_*.png` (frame del video)

## Esempio

```bash
python3 scripts/predict_firstpass_and_track_from_folder.py \
  --input_dir ../fromgcloud/2023 \
  --output_dir ../airmassRGB/firstpass_track \
  --firstpass_model_path ../firstpass/outputs/runs/exp1/checkpoints/best.ckpt \
  --tracking_model_path ./output/checkpoint-tracking-best.pth \
  --firstpass_threshold 0.2 \
  --make_video \
  --video_name mediterraneo_firstpass_track \
```

## Note

- Lo script usa solo tile first-pass positive per il tracking HR.
- Il naming tile e il formato frame sono compatibili con `track_from_folder.py`.
- Il video `--make_video` non usa il mosaico 12 tile VideoMAE: renderizza il frame originale e disegna un solo riquadro rosso (ROI) quando la detection first-pass è positiva.
- Il video viene renderizzato su **tutti** i frame disponibili in `input_dir` (es. ogni 5 minuti); tra due predizioni consecutive mantiene marker e stato dell'ultimo timestamp predetto.
- Overlay marker nel video: rombo rosso = centro coarse first-pass, punto rosso = predizione tracking VideoMAE (globale), punto verde = GT da `manos_file` quando disponibile.
- Nel CSV finale, `pred_lat/pred_lon` usano il tracking quando disponibile; se assente (es. `has_cyclone=0`) viene mantenuta la stima coarse first-pass.
- Le copie stretched temporanee vengono mantenute in `output_dir/_tmp_firstpass_stretched`.
- Caching automatico: se esistono i file tmp (`_tmp_firstpass_predictions.csv`, `_tmp_tracking_inference_predictions_tiles.csv`) non vengono ricalcolati.
- Le tile originali in `firstpass_tiles` non vengono modificate: l'overlay tracking viene scritto in una cartella separata.
