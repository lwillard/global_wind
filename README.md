# Global Wind — 3D GDAS ARL Wind Viewer

This project displays global wind fields from **GDAS ARL** (`.w4`) files on a rotating 3D globe.

> Main script: `global_wind.py`

---

## Features

- **ARL indexer**: scans `.w4` files and builds a random-access index of times/levels.
- **Interactive 3D globe** with PyOpenGL inside a PyQt5 window.
- **Wind rendering**: curved arrow segments with shadow/white layers for legibility.
- **Cursor readout**: live lat/lon, bilinear-interpolated wind (mph) and meteorological bearing.
- **Optional land mask**: textured shell from an equirectangular PNG.
- **Optional coastlines**: Natural Earth 50m outlines (downloaded on first run).

---

## Quick Start

### 1) Environment
Python 3.9–3.12 recommended.

```bash
# (optional) create a virtual environment
python -m venv .venv
# On macOS/Linux
source .venv/bin/activate
# On Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

> **Note:** GeoPandas/Shapely can be tricky to build from source. If you do NOT need coastlines,
> you can skip those two and the app will still run.

**Minimal (no coastlines, no land mask):**
```bash
pip install numpy PyQt5 PyOpenGL
```

**Recommended (with land mask; Pillow)**
```bash
pip install numpy PyQt5 PyOpenGL Pillow requests
```

**Full (with coastlines; adds GeoPandas/Shapely)**
```bash
pip install numpy PyQt5 PyOpenGL Pillow requests geopandas shapely
```
If installing `geopandas` or `shapely` fails, try a scientific Python distro (e.g., conda/mamba):

```bash
# using mamba (fast) or conda
mamba create -n global python=3.11 numpy pyqt pyopengl pillow requests geopandas shapely
mamba activate global
```

### 3) Get wind data (`.w4`)

Place a GDAS ARL file in the working directory. Example naming: `gdas1.aug25.w4` (daily bundles).
If you have a different filename, pass it as the first CLI argument.

### 4) Run

```bash
# Default filename: gdas1.aug25.w4
python global_wind.py

# Or specify your own ARL file:
python global_wind.py /path/to/your_data.w4
```

---

## Controls

- **Rotate:** Click+drag (left mouse).
- **Zoom:** Mouse wheel / trackpad scroll. Keyboard: `+`/`-`, `PageUp`/`PageDown`.
- **Time level:** Bottom “Time” slider.
- **Pressure level:** Bottom “Level” slider (hPa labels).

The status bar (bottom) shows:
- Cursor **Lat/Lon**
- **Wind direction** (° from which the wind blows; meteorological)
- **Wind speed** in **mph**

---

## Files & Layout

- `global_wind.py` — the main program for displaying global wind patterns.
- `land_mask_2048x1024.png` — optional equirectangular alpha mask (white=land). Place next to the script.
- Natural Earth 50m land shapefile — auto-downloaded (if GeoPandas/Shapely/requests installed).
---

## Packaging and Installation (optional)

You can install this script as a command-line tool using `setup.py` provided here.

```bash
pip install .
# This installs an entry point 'global-wind' into your environment.
global-wind              # will try DEFAULT_FILE
global-wind yourfile.w4  # runs with your ARL file
```

To build distribution artifacts:

```bash
pip install build
python -m build
```
This creates `dist/*.tar.gz` and `dist/*.whl` that you can install elsewhere:

```bash
pip install dist/global_wind-*.whl
```

---

## Troubleshooting

### Qt platform plugin errors (e.g., `xcb`, `cocoa`, or `windows` not loading)
- Ensure `PyQt5` is installed in the **current** environment.
- On Linux, install system packages for OpenGL/GLX and X11 (varies by distro).
- On WSL, use an X server or run in a desktop environment.

### OpenGL issues (blank window, very slow)
- Update graphics drivers.
- Try reducing sphere tessellation (search for `gluSphere(..., 64, 64)` and lower the segments).
- Disable line smoothing (toggle `self.line_smooth_enabled = False` in `Globe`).

### GeoPandas/Shapely install failures
- Prefer a conda/mamba environment: these packages ship prebuilt manylinux wheels there.
- Or skip coastlines entirely; the viewer will still run without them.

### Natural Earth download blocked
- Manually download **50m land** shapefile and extract in the working directory
  (so `ne_50m_land.shp` exists alongside the script).

### Pillow cannot load land mask
- Ensure `land_mask_2048x1024.png` exists; try converting to plain PNG with RGBA or L mode.

### Performance tuning
- Reduce `equator_samples` in `Globe` to draw fewer arrows.
- Increase `min_samples_per_lat` to avoid sparsity at high latitudes.
- Increase `arc_gain` slightly to make arrows longer (more visible), or decrease for speed.

---

## Developer Notes

- **Data model:** We index U/V records per `(time_index, level_index)` and lazily decode fields on demand.
- **Coordinate convention:** Rendering uses **west-negative** longitudes for camera feel; raw grids are **0..360 East**.
- **Interpolation:** Cursor winds are bilinear on the 1° grid — good enough for visualization.
- **Licenses:** Natural Earth data are in the public domain. Respect any local data license for your ARL files.

---

## FAQ

**Q: Can I change wind units?**  
A: Yes. See the cursor readout method; convert m/s to whatever you need.

**Q: Can I load other ARL variables?**  
A: Extend the indexer to also map `TTMP`, `RHUM`, etc., and add rendering overlays as needed.

**Q: Where can I get `.w4` data?**  
A: GDAS/ARL files come from atmospheric transport/dispersion workflows (e.g., NOAA ARL/HYSPLIT datasets).
Distribution sources vary; ensure you have permission and correct format.
https://www.ready.noaa.gov/data/archives/gdas1/ is where I retrieved my wind data from

---

## License
This code is provided as-is for educational purposes.
