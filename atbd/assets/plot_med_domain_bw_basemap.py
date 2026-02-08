import os
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIDEOMAE_DIR = os.path.join(ROOT_DIR, "moduli", "videomae")
if VIDEOMAE_DIR not in sys.path:
    sys.path.insert(0, VIDEOMAE_DIR)

from medicane_utils.geo_const import latcorners, loncorners, create_basemap_obj

# Output
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "med_domain_bw_basemap.png")

# Figure
dpi = 150
fig = plt.figure(figsize=(8.5, 5.5), dpi=dpi)
ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])

# Basemap from existing project code
m = create_basemap_obj(ax=ax)

# Coastlines and grid
m.drawcoastlines(linewidth=0.8, color="black", zorder=2)

lat_min, lat_max = latcorners
lon_min, lon_max = loncorners

parallels = np.arange(lat_min, lat_max + 0.1, 2.0)
meridians = np.arange(lon_min, lon_max + 0.1, 2.0)

m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=8, color="0.5")
m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=8, color="0.5", rotation=45)

ax.set_title("Mediterranean domain")

fig.savefig(OUT_PNG, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
print(f"Saved: {OUT_PNG}")
