<div align="center">
<h1> DeMeTra </h1>
<br>
<img src="moduli/videomae/misc/readme_img_earth.png" alt="Project Icon" width="400" />
<h3> Medicanes detection and tracking </h3>
</div>


## Installazione (Windows, Linux, CPU e CUDA)

Sono supportati Python 3.9–3.11 e un ambiente virtuale dedicato. Il primo
comando seleziona PyTorch: scegliere **una sola** variante; non modificare i
file requirements.

### Windows PowerShell, CPU

```powershell
git clone https://github.com/dandarm/Demetra.git
cd Demetra
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements.txt
winget install Gyan.FFmpeg
```

Chiudere e riaprire PowerShell dopo l'installazione di FFmpeg, poi riattivare
`.venv`. Se `winget` non è disponibile, installare FFmpeg da
[ffmpeg.org](https://ffmpeg.org/download.html) e passare il suo `bin` con
`--ffmpeg-path C:\percorso\ffmpeg\bin`.

### Linux, CPU

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements.txt
sudo apt install ffmpeg
```

### GPU NVIDIA

Al posto di `requirements-cpu.txt`, installare `requirements-cuda118.txt` su
un sistema con driver compatibile con CUDA 11.8, poi installare
`requirements.txt`. Per altre versioni CUDA, seguire la matrice ufficiale
PyTorch e mantenere `requirements.txt` invariato.

Il tracking VideoMAE-Large richiede molta RAM e, su CPU, può richiedere ore.
I default dell'inferenza sono volutamente conservativi (`batch=1`, `workers=0`)
e funzionano con Windows; aumentarli soltanto dopo una prima esecuzione
riuscita. Il download e gli artefatti necessitano almeno 20 GiB liberi come
controllo iniziale e spesso di più per intervalli lunghi.

## Model weights

The DeMeTrA v1 checkpoints are available on Zenodo:
[doi:10.5281/zenodo.23123832](https://doi.org/10.5281/zenodo.23123832).
They are distributed under the CC BY-NC 4.0 licence.

For inference, download these two files into `trained_models/`:

| File | Use |
| --- | --- |
| `firstpass_model.ckpt` | X3D-M first-pass cyclone detection and coarse centre localization |
| `checkpoint_new_tracking2.pth` | VideoMAE-Large supervised centre-tracking model |

```bash
mkdir -p trained_models

curl -L \
  'https://zenodo.org/records/23123832/files/firstpass_model.ckpt?download=1' \
  -o trained_models/firstpass_model.ckpt
curl -L \
  'https://zenodo.org/records/23123832/files/checkpoint_new_tracking2.pth?download=1' \
  -o trained_models/checkpoint_new_tracking2.pth
```

`checkpoint_large_new.pth` is also available in the Zenodo release. It is the
self-supervised VideoMAE-Large specialization checkpoint used to initialize
tracking training; it is not required for inference.

Verify downloaded files before use:

```bash
echo '0a841577b376a077cf9eb7856f5168f4be2043066779e241203fe49b3e0c48fa  trained_models/firstpass_model.ckpt' | sha256sum -c -
echo 'f5607edaccc5b802773dd67be9e69b1195e69bd42095a2ff8fceaa5bd2a34f4a  trained_models/checkpoint_new_tracking2.pth' | sha256sum -c -
```

Maintainers can publish a much smaller inference-only tracking artifact with:

```bash
python scripts/export_model_only_checkpoint.py \
  --input trained_models/checkpoint_new_tracking2.pth \
  --output trained_models/checkpoint_new_tracking2_model_only.pth
```

The command prints the exact size and SHA-256 to publish with the release. The
generated file is accepted by the current inference loader; it intentionally
contains no optimizer, RNG state, local paths, or training arguments.

## Quick start

### Verifica locale senza credenziali

Questo comando controlla le dipendenze Python e genera i manifest dal piccolo
dataset incluso; non scarica dati satellitari né carica checkpoint:

```bash
cd moduli/firstpass
python -m pytest tests/test_letterbox.py tests/test_manifest_cli.py
cd ../..
```

Controllare inoltre che gli entrypoint siano disponibili:

```bash
python scripts/predict_firstpass_and_track_from_folder.py --help
python scripts/download_and_track_range.py --help
```

### Download EUMETSAT e tracking

Per dati recenti (incluso marzo 2026) sono obbligatorie credenziali EUMETSAT.
Lo script si ferma prima del download se mancano credenziali, checkpoint,
FFmpeg o spazio su disco.

```bash
export EUMETSAT_CONSUMER_KEY=<your_consumer_key>
export EUMETSAT_CONSUMER_SECRET=<your_consumer_secret>

python scripts/download_and_track_range.py \
  --start 15-03-2026 --end 17-03-2026 \
  --firstpass_model_path trained_models/firstpass_model.ckpt \
  --tracking_model_path trained_models/checkpoint_new_tracking2.pth \
  --output_root output
```

Su Windows, con `.venv` attivo, usare PowerShell:

```powershell
$env:EUMETSAT_CONSUMER_KEY = '<your_consumer_key>'
$env:EUMETSAT_CONSUMER_SECRET = '<your_consumer_secret>'
python .\scripts\download_and_track_range.py --start 15-03-2026 --end 17-03-2026
```

Il primo avvio scarica inoltre il backbone X3D-M usato dal first-pass. CPU è
utile per il controllo d'installazione ma non è un'impostazione pratica per
produzioni estese; usare una GPU NVIDIA per range temporali grandi.



## Example using most common features

```bash
python scripts/predict_firstpass_and_track_from_folder.py   \
--input_dir /path/to/source_dataset/jolina  \
--output_dir output/jolina  \
--firstpass_model_path trained_models/firstpass_model.ckpt \
--tracking_model_path trained_models/checkpoint_new_tracking2.pth \
--firstpass_threshold 0.2 \
--make_video \
--ffmpeg_path /path/to/ffmpeg/bin \
--standard_tiling \
--video_coastlines \
--video_tracking_dot_only 
```

I file CSV LFS storici `all_manos_CL*.csv` non sono richiesti dal quick start
né dall'inferenza. Sono stati esclusi dalla distribuzione per evitare che un
clone pubblico dipenda da oggetti LFS non pubblicati.
