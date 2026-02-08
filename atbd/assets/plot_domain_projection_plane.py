import os
import sys
import numpy as np
import shapefile  # pyshp
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIDEOMAE_DIR = os.path.join(ROOT_DIR, "moduli", "videomae")
if VIDEOMAE_DIR not in sys.path:
    sys.path.insert(0, VIDEOMAE_DIR)

from medicane_utils.geo_const import latcorners, loncorners, create_basemap_obj

# Output
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "med_domain_projected_on_plane.png")

# Coastlines shapefile (planar lon/lat plot)
COAST_SHP = os.path.join(os.path.dirname(__file__), "ne_110m_coastline", "ne_110m_coastline.shp")

# Build Basemap domain (geos) to get projected rectangle
m = create_basemap_obj()
lat_min, lat_max = latcorners
lon_min, lon_max = loncorners

x_min, y_min = m(lon_min, lat_min)
x_max, y_max = m(lon_max, lat_max)

# Sample edges of the projected rectangle in (x,y)
num = 400
x_vals = np.linspace(x_min, x_max, num)
y_vals = np.linspace(y_min, y_max, num)

# Bottom/top edges (y fixed)
xb = x_vals
xt = x_vals
yb = np.full_like(x_vals, y_min)
yt = np.full_like(x_vals, y_max)

# Left/right edges (x fixed)
yl = y_vals
yr = y_vals
xl = np.full_like(y_vals, x_min)
xr = np.full_like(y_vals, x_max)

# Inverse projection to lon/lat for each edge
lon_b, lat_b = m(xb, yb, inverse=True)
lon_t, lat_t = m(xt, yt, inverse=True)
lon_l, lat_l = m(xl, yl, inverse=True)
lon_r, lat_r = m(xr, yr, inverse=True)

# Plot in lon/lat plane
fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
ax.set_facecolor("white")

# Coastlines
reader = shapefile.Reader(COAST_SHP)
for shape in reader.shapes():
    pts = shape.points
    parts = list(shape.parts) + [len(pts)]
    for i in range(len(parts) - 1):
        seg = pts[parts[i]:parts[i + 1]]
        if len(seg) < 2:
            continue
        lons, lats = zip(*seg)
        ax.plot(lons, lats, color="black", linewidth=0.6)

# Projected rectangle (curved in lon/lat)
ax.plot(lon_b, lat_b, color="black", linewidth=1.5)
ax.plot(lon_t, lat_t, color="black", linewidth=1.5)
ax.plot(lon_l, lat_l, color="black", linewidth=1.5)
ax.plot(lon_r, lat_r, color="black", linewidth=1.5)

# Grid
ax.set_xlim(lon_min - 15, lon_max + 15)
ax.set_ylim(lat_min - 8, lat_max + 8)
ax.set_xticks(np.arange(-20, 61, 5))
ax.set_yticks(np.arange(20, 61, 5))
ax.grid(True, color="0.7", linewidth=0.5, linestyle="-")

ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")
ax.set_title("Mediterranean domain projected on lon/lat plane")
ax.set_aspect("equal", adjustable="box")

fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", pad_inches=0.05)
print(f"Saved: {OUT_PNG}")
