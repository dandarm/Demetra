# Download And Track Range

Questo file riassume le procedure implementate e testate nella sessione corrente per unificare:

1. download delle immagini satellitari;
2. conversione in frame Airmass RGB;
3. inferenza first-pass + tracking;
4. generazione del CSV finale e del video.

## Script creato

Lo script principale aggiunto e':

- [download_and_track_range.py](/media/isacDisk1/Demetra/scripts/download_and_track_range.py)

Questo wrapper:

- accetta solo `--start` e `--end` come input temporali principali;
- usa di default l'environment Python:
  `/home/isac/miniconda3/envs/videomae/bin/python`
- usa di default i checkpoint:
  - `/media/isacDisk1/Demetra/trained_models/firstpass_model.ckpt`
  - `/media/isacDisk1/Demetra/trained_models/checkpoint-tracking-best_1.pth`
- salva tutto sotto:
  `/media/isacDisk1/airmassRGB/`
- supporta download concorrente EUMETSAT con:
  - `--eumetsat_download_workers`
  - `--eumetsat_download_retries`
  - `--eumetsat_read_timeout`
  - `--video_coastlines`

## Logica implementata

### 1. Download source

Lo script supporta due sorgenti:

- `public`
  usa il dataset pubblico GCS storico:
  `public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4`

- `eumetsat`
  usa `eumdac` con collection:
  `EO:EUM:DAT:MSG:MSG15-RSS`

- `auto`
  prova il dataset pubblico e, se il periodo non e' coperto, passa a `eumetsat` se sono presenti le credenziali.

### 2. Conversione in Airmass RGB

Sono stati riusati due percorsi gia' presenti nel repo:

- per il dataset pubblico:
  funzioni da [download_airmassRGB.py](/media/isacDisk1/Demetra/moduli/videomae/medicane_utils/download_airmassRGB.py)

- per i dati EUMETSAT scaricati in ZIP:
  [create_airmassRGB_from_hrseviri_local.py](/media/isacDisk1/Demetra/moduli/videomae/medicane_utils/create_airmassRGB_from_hrseviri_local.py)

Nel ramo `eumetsat`, lo script scarica ZIP `MSG15-RSS` in `raw_eumetsat/` e poi genera PNG:

- `airmass_rgb_YYYYMMDD_HHMM.png`

in:

- `frames/`

### 3. Inferenza

Per l'inferenza viene richiamato:

- [predict_firstpass_and_track_from_folder.py](/media/isacDisk1/Demetra/scripts/predict_firstpass_and_track_from_folder.py)

che esegue:

- first-pass sul frame completo;
- costruzione delle tile positive;
- tracking VideoMAE;
- CSV finale per timeframe;
- video finale.

## File modificati nella sessione

- [download_and_track_range.py](/media/isacDisk1/Demetra/scripts/download_and_track_range.py)
- [predict_firstpass_and_track_from_folder.py](/media/isacDisk1/Demetra/scripts/predict_firstpass_and_track_from_folder.py)

## Test effettuati

### Test 1: dataset pubblico storico

Run corta di validazione sul dataset pubblico:

- output:
  [range_20240531_2200__20240531_2315](/media/isacDisk1/airmassRGB/range_20240531_2200__20240531_2315)

Output presenti:

- [tracking_inference_predictions.csv](/media/isacDisk1/airmassRGB/range_20240531_2200__20240531_2315/tracking_inference_predictions.csv)
- [range_20240531_2200__20240531_2315.mp4](/media/isacDisk1/airmassRGB/range_20240531_2200__20240531_2315/range_20240531_2200__20240531_2315.mp4)



### Test 2: validazione del ramo EUMETSAT su gennaio 2026

Run corta di prova sul ramo `eumetsat`:

- periodo:
  `2026-01-22 23:45 -> 2026-01-23 00:00`

output:

- [range_20260122_2345__20260123_0000](/media/isacDisk1/airmassRGB/range_20260122_2345__20260123_0000)

Output presenti:

- ZIP scaricati in:
  [raw_eumetsat](/media/isacDisk1/airmassRGB/range_20260122_2345__20260123_0000/raw_eumetsat)
- PNG generati in:
  [frames](/media/isacDisk1/airmassRGB/range_20260122_2345__20260123_0000/frames)

In questa prova il ramo `eumetsat` e' stato verificato con successo:

- ricerca prodotti `MSG15-RSS`;
- download ZIP;
- conversione ZIP -> Airmass RGB.

### Test 3: run completa sull'ultima finestra disponibile del dataset pubblico

Run completa di riferimento sul dataset pubblico:

- output:
  [range_20240529_0000__20240531_2355](/media/isacDisk1/airmassRGB/range_20240529_0000__20240531_2355)

Output principali presenti:

- [tracking_inference_predictions.csv](/media/isacDisk1/airmassRGB/range_20240529_0000__20240531_2355/tracking_inference_predictions.csv)
- [range_20240529_0000__20240531_2355.mp4](/media/isacDisk1/airmassRGB/range_20240529_0000__20240531_2355/range_20240529_0000__20240531_2355.mp4)

### Test 4: avvio della run richiesta su marzo 2026

Run avviata sul periodo richiesto:

- `15-03-2026 -> 17-03-2026`

output:

- [range_20260315_0000__20260317_2355](/media/isacDisk1/airmassRGB/range_20260315_0000__20260317_2355)

Stato attuale della cartella:

- ZIP gia' scaricati in:
  [raw_eumetsat](/media/isacDisk1/airmassRGB/range_20260315_0000__20260317_2355/raw_eumetsat)
- log della sessione in:
  [run.log](/media/isacDisk1/airmassRGB/range_20260315_0000__20260317_2355/run.log)

La run lunga e' stata interrotta manualmente dopo aver verificato che:

- la search EUMETSAT trova prodotti reali nel periodo marzo 2026;
- il download procede correttamente ma con latenza irregolare;
- i file gia' scaricati restano in cache e verranno riutilizzati al rilancio.

## Esempio di chiamata da terminale

### Caso generale

```bash
/home/isac/miniconda3/envs/videomae/bin/python \
  /media/isacDisk1/Demetra/scripts/download_and_track_range.py \
  --start 15-03-2026 \
  --end 17-03-2026 \
  --download_source eumetsat \
  --eumetsat_download_workers 4 \
  --video_coastlines
```

### Solo download e conversione, senza inferenza

```bash
/home/isac/miniconda3/envs/videomae/bin/python \
  /media/isacDisk1/Demetra/scripts/download_and_track_range.py \
  --start 15-03-2026 \
  --end 17-03-2026 \
  --download_source eumetsat \
  --skip_inference
```

## Credenziali EUMETSAT

Per usare `--download_source eumetsat`, devono essere presenti in ambiente:

- `EUMETSAT_CONSUMER_KEY`
- `EUMETSAT_CONSUMER_SECRET`

Lo script non contiene le credenziali in chiaro.

## Come rilanciare la stessa run

Per riprendere la run gia' iniziata su marzo 2026, basta rilanciare lo stesso comando:

```bash
/home/isac/miniconda3/envs/videomae/bin/python \
  /media/isacDisk1/Demetra/scripts/download_and_track_range.py \
  --start 15-03-2026 \
  --end 17-03-2026 \
  --download_source eumetsat
```

Lo script riusa automaticamente:

- ZIP gia' scaricati in:
  [raw_eumetsat](/media/isacDisk1/airmassRGB/range_20260315_0000__20260317_2355/raw_eumetsat)
- PNG gia' generati in:
  [frames](/media/isacDisk1/airmassRGB/range_20260315_0000__20260317_2355/frames)

Quindi il rilancio non riparte da zero.

## Parametri utili per EUMETSAT

Per i download via Data Store EUMETSAT sono ora disponibili questi parametri:

- `--eumetsat_download_workers`
  numero di download concorrenti; il default impostato e' `4`
- `--eumetsat_download_retries`
  numero massimo di retry per prodotto; il default e' `3`
- `--eumetsat_read_timeout`
  timeout di lettura per stream HTTP; il default e' `180` secondi

Esempio:

```bash
/home/isac/miniconda3/envs/videomae/bin/python \
  /media/isacDisk1/Demetra/scripts/download_and_track_range.py \
  --start 15-03-2026 \
  --end 17-03-2026 \
  --download_source eumetsat \
  --eumetsat_download_workers 4 \
  --eumetsat_download_retries 3 \
  --eumetsat_read_timeout 180
```

## Dove sono gia' salvati i dati

Cartelle principali gia' presenti:

- [range_20240531_2200__20240531_2315](/media/isacDisk1/airmassRGB/range_20240531_2200__20240531_2315)
- [range_20240529_0000__20240531_2355](/media/isacDisk1/airmassRGB/range_20240529_0000__20240531_2355)
- [range_20260122_2345__20260123_0000](/media/isacDisk1/airmassRGB/range_20260122_2345__20260123_0000)
- [range_20260315_0000__20260317_2355](/media/isacDisk1/airmassRGB/range_20260315_0000__20260317_2355)

## Note pratiche

- il dataset pubblico GCS disponibile nel repo arriva solo fino a `2024-05-31 23:15 UTC`;
- per il 2026 va usato il ramo `eumetsat`;
- il downloader EUMETSAT puo' avere latenza molto variabile su alcuni prodotti;
- il path di `ffmpeg` viene risolto dal file:
  [ffmpeg_utils.py](/media/isacDisk1/Demetra/moduli/videomae/ffmpeg_utils.py)
