import os
import numpy as np
import shapefile  # pyshp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Domain corners (lat/lon)
LAT_MIN, LAT_MAX = 30, 48
LON_MIN, LON_MAX = -7, 46

# Paths
BASE_DIR = os.path.dirname(__file__)
SHP_PATH = os.path.join(BASE_DIR, "ne_110m_coastline", "ne_110m_coastline.shp")
OUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, "med_domain_bw.png")
OUT_SVG = os.path.join(OUT_DIR, "med_domain_bw.svg")

# Plot settings
fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
ax.set_facecolor("white")

# Coastlines
reader = shapefile.Reader(SHP_PATH)
for shape in reader.shapes():
    pts = shape.points
    parts = list(shape.parts) + [len(pts)]
    for i in range(len(parts) - 1):
        seg = pts[parts[i]:parts[i + 1]]
        if len(seg) < 2:
            continue
        lons, lats = zip(*seg)
        ax.plot(lons, lats, color="black", linewidth=0.6)

# Domain rectangle
rect = Rectangle(
    (LON_MIN, LAT_MIN),
    LON_MAX - LON_MIN,
    LAT_MAX - LAT_MIN,
    fill=False,
    edgecolor="black",
    linewidth=1.5,
)
ax.add_patch(rect)

# Grid (meridians/parallels)
lon_pad = 8
lat_pad = 5
ax.set_xlim(LON_MIN - lon_pad, LON_MAX + lon_pad)
ax.set_ylim(LAT_MIN - lat_pad, LAT_MAX + lat_pad)

lon_ticks = np.arange(-20, 61, 5)
lat_ticks = np.arange(20, 61, 5)
ax.set_xticks(lon_ticks)
ax.set_yticks(lat_ticks)
ax.grid(True, color="0.7", linewidth=0.5, linestyle="-")

ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")
ax.set_title("Mediterranean domain (lat/lon corners)")
ax.set_aspect("equal", adjustable="box")

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
fig.savefig(OUT_SVG)
print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_SVG}")
