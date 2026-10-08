# Download e tracking su un intervallo temporale

`download_and_track_range.py` scarica frame Airmass RGB, esegue il first-pass
e il tracking VideoMAE e salva CSV e MP4 in una sottocartella di `output/`.
Non dipende da percorsi della macchina di sviluppo: tutti i percorsi possono
essere passati sulla riga di comando.

## Prerequisiti

- Ambiente installato secondo il [README](../README.md), incluso FFmpeg.
- `trained_models/firstpass_model.ckpt` e
  `trained_models/checkpoint_new_tracking2.pth`.
- Per intervalli non disponibili nel bucket storico pubblico, credenziali
  EUMETSAT in `EUMETSAT_CONSUMER_KEY` e `EUMETSAT_CONSUMER_SECRET`.
- Almeno 20 GiB liberi (modificabile con `--min-free-gib`).

Il comando esegue questi controlli prima di iniziare download voluminosi.

## Esempio

```bash
python scripts/download_and_track_range.py \
  --start 15-03-2026 --end 17-03-2026 \
  --download_source eumetsat \
  --output_root output \
  --firstpass-batch-size 1 --firstpass-num-workers 0 \
  --tracking-batch-size 1 --tracking-num-workers 0
```

Su Windows usare in PowerShell:

```powershell
$env:EUMETSAT_CONSUMER_KEY = '<your_consumer_key>'
$env:EUMETSAT_CONSUMER_SECRET = '<your_consumer_secret>'
python .\scripts\download_and_track_range.py --start 15-03-2026 --end 17-03-2026
```

Il default usa `sys.executable`, quindi l'inferenza figlia usa lo stesso
ambiente virtuale del wrapper. Per un FFmpeg fuori dal `PATH`, aggiungere
`--ffmpeg-path C:\percorso\ffmpeg\bin` su Windows oppure
`--ffmpeg-path /percorso/ffmpeg/bin` su Linux.

## Sorgenti e ripresa

- `--download_source public` usa il bucket GCS storico anonimo.
- `--download_source eumetsat` richiede credenziali e accede a MSG15-RSS.
- `--download_source auto` preferisce i dati pubblici e usa EUMETSAT quando
  necessario e configurato.
- `--skip_inference` scarica/converte solo i frame.
- `--force` conserva i frame ma rigenera predizioni e video.

I worker e batch size sono intenzionalmente bassi per CPU e Windows. Aumentarli
soltanto dopo aver confermato memoria e stabilità su una breve finestra.
