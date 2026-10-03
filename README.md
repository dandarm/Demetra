<div align="center">
<h1> DeMeTra </h1>
<br>
<img src="moduli/videomae/misc/readme_img_earth.png" alt="Project Icon" width="400" />
<h3> Medicanes detection and tracking </h3>
</div>


## Environment setup

Create a Python 3.9 Conda environment and install the repo dependencies:

```bash
conda create -n demetra python=3.9 -y
conda activate demetra

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


Note: the file currently pins `torch==1.12.1+cu113`, `torchvision==0.13.1+cu113`
and `torchaudio==0.12.1+cu113`. If your machine does not use CUDA 11.3, adjust
those lines before installing.

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

## Quick start
Launch the following script to download image data from Eumetsat (using your account keys) and track with DeMeTra

```bash
export EUMETSAT_CONSUMER_KEY=<your_consumer_key>
export EUMETSAT_CONSUMER_SECRET=<your_consumer_secret>

conda activate demetra

python scripts/download_and_track_range.py \
  --start 15-03-2026 --end 17-03-2026 \
  --firstpass_model_path trained_models/firstpass_model.ckpt \
  --tracking_model_path trained_models/checkpoint_new_tracking2.pth
```

the script will automatically download data from EUMETSAT using your account keys



## Example using most common features

```bash
python scripts/predict_firstpass_and_track_from_folder.py   \
--input_dir /media/isacDisk2/source_dataset_by_cyc/jolina  \
--output_dir /media/isacDisk2/demetra_output/jolina  \
--firstpass_model_path trained_models/firstpass_model.ckpt \
--tracking_model_path trained_models/checkpoint_new_tracking2.pth \
--firstpass_threshold 0.2 \
--make_video \
--ffmpeg_path /mnt/share/Demetra_files/VideoMAEv2/ffmpeg-7.0.2-amd64-static/ \
--standard_tiling \
--video_coastlines \
--video_tracking_dot_only 
```
