<div align="center">
<h1> DeMeTra </h1>
<br>
<img src="moduli/videomae/misc/readme_img_earth.png" alt="Project Icon" width="400" />
<h3> Medicanes detection and tracking </h3>
</div>


## Quick start
Launch the following script to download image data from Eumetsat (using your account keys) and track with DeMeTra

```bash
export EUMETSAT_CONSUMER_KEY=<your_consumer_key>
export EUMETSAT_CONSUMER_SECRET=<your_consumer_secret>
python scripts/download_and_track_range.py   --start 15-03-2026   --end 17-03-2026   --download_source eumetsat
```
