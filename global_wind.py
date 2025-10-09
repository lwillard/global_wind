"""
global_wind.py — A step-by-step, tutorial-style 3D wind visualization globe

WHAT THIS PROGRAM DOES
----------------------
This program reads global wind fields from a GDAS ARL (.w4) file and displays them as
flowing arrow segments on a rotating 3D globe. It uses:
  • NumPy for fast array math
  • PyQt5 for the desktop GUI (window, sliders, status bar)
  • PyOpenGL for 3D rendering
  • (Optional) Pillow for a land-mask texture on the globe
  • (Optional) GeoPandas/Shapely for coastline outlines (Natural Earth 50m)

HOW TO READ THIS FILE
---------------------
This file is written as a teaching resource. Every class/method is documented.
Inline comments explain design decisions and graphics math (e.g., tangent frames
on the sphere, bilinear interpolation, etc.). You can skim top-to-bottom, or
jump to the section you’re curious about:
  1) ARL parsing helpers (parse_header, unpack_field)
  2) ARL indexing (ARLRecordRef, ARLIndex, ARLIndexer thread)
  3) Coastline loader (downloads Natural Earth if missing)
  4) OpenGL globe widget (draw ocean sphere, land mask, coastlines, winds)
  5) MainWindow (Qt GUI and interactions)
  6) main() entry point
"""

# ----- Standard library imports ----------------------------------------------
import sys, os, math, zipfile, datetime as dt
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# ----- Third-party libraries --------------------------------------------------
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
from PyQt5.QtOpenGL import QGLWidget

# PyOpenGL gives you direct access to OpenGL functions.
from OpenGL.GL import *
from OpenGL.GLU import *

# Optional: Pillow for the land-mask image. If missing, we just skip the mask.
try:
    from PIL import Image, ImageChops
    PIL_OK = True
except Exception:
    PIL_OK = False

# Optional: requests + GeoPandas + Shapely for coastlines. Also skippable.
try:
    import requests
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon
    GP_OK = True
except Exception:
    GP_OK = False

# ----- ARL (GDAS) constants ---------------------------------------------------
# GDAS 1° grid: 360 longitudes (0..359E) × 181 latitudes (-90..+90 inclusive).
NX = 360
NY = 181

# Each ARL record is fixed-length: 50-byte ASCII header + 1 byte per grid cell
REC_LEN = 50 + NX * NY

# If you run this file from the terminal without args, it tries this file name:
DEFAULT_FILE = "gdas1.aug25.w4"

# Optional external assets
MASK_FILENAME = "land_mask_2048x1024.png"  # equirectangular, alpha=land

# Pressure level lookup (ARL level indices → hPa). Useful for labeling.
LEVELS_HPA_TABLE = [
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
     750, 725, 700, 650, 600, 550, 500, 450, 400, 350,
     300, 250, 200, 150, 100,  70,  50,  30,  20,  10
]
LEVEL_TO_HPA = {i+1: h for i, h in enumerate(LEVELS_HPA_TABLE)}

# -----------------------------------------------------------------------------
# 1) ARL PARSING HELPERS
# -----------------------------------------------------------------------------
def parse_header(hdr_bytes: bytes) -> dict:
    """
    Parse the 50-byte ARL ASCII header into a Python dictionary.

    Header layout (by convention for GDAS ARL):
      - YYMMDDHH (8 chars): UTC timestamp of the analysis
      - FCST (2): forecast hour (-1 for index/meta records)
      - LVL  (2): vertical level index (1..30 here)
      - GRID (2): grid identifier
      - VAR  (4): variable name ('UWND', 'VWND', 'INDX', etc.)
      - EXP  (4): integer exponent controlling unpacking scale
      - PREC (14): float; absolute values below this are clamped to 0
      - VAL11(14): float; starting/reference value at grid (1,1)

    Returns keys:
      'time' (datetime, tz-aware UTC), 'fcst_hr' (int), 'level_index' (int),
      'grid_id' (int), 'var' (str), 'exp' (int), 'precision' (float), 'val11' (float).
    """
    s = hdr_bytes.decode('ascii', errors='ignore')
    year   = 2000 + int(s[0:2])
    month  = int(s[2:4])
    day    = int(s[4:6])
    hour   = int(s[6:8])
    fcst   = int(s[8:10])
    level  = int(s[10:12])
    grid   = int(s[12:14])
    var    = s[14:18].strip()
    exp    = int(s[18:22])
    prec   = float(s[22:36])
    val11  = float(s[36:50])
    time   = dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc)
    return {"time": time, "fcst_hr": fcst, "level_index": level, "grid_id": grid,
            "var": var, "exp": exp, "precision": prec, "val11": val11}


def unpack_field(data_bytes: bytes, exp: int, precision: float, val11: float) -> np.ndarray:
    """
    Convert the ARL "delta-encoded bytes" into a float32 grid.

    The file stores differences from neighbors as unsigned bytes (0..255).
    We shift by 127 to center around 0, then scale by 2^(7-exp). The first
    value (row 0, col 0) starts at 'val11', and values accumulate along
    rows and columns to reconstruct the field.

    Args:
        data_bytes: Raw NX*NY bytes.
        exp: Scale exponent from header.
        precision: Absolute values below this become 0 (noise clamp).
        val11: Starting value at (0,0).

    Returns:
        R: (NY, NX) float32 array (e.g., U or V component in m/s).
    """
    # 1) reshape the flat byte stream into a 2D array of small ints
    P = np.frombuffer(data_bytes, dtype=np.uint8).astype(np.int16).reshape(NY, NX)

    # 2) convert byte differences into real-value steps
    scale = 2.0 ** (7 - exp)

    # 3) accumulate along first column, then along rows
    R = np.empty((NY, NX), dtype=np.float32)
    R[0, 0] = val11
    R[1:, 0] = np.cumsum((P[1:, 0] - 127) / scale) + R[0, 0]
    for j in range(NY):
        diffs = (P[j, 1:] - 127) / scale
        R[j, 1:] = np.cumsum(diffs) + R[j, 0]

    # 4) clamp tiny values to zero for visual stability
    if precision > 0:
        R[np.abs(R) < precision] = 0.0
    return R

# -----------------------------------------------------------------------------
# 2) ARL INDEXING (THREAD)
# -----------------------------------------------------------------------------
@dataclass
class ARLRecordRef:
    """
    A lightweight pointer to a record in the ARL file.
    We store where it lives (offset) and how to unpack it later (exp, precision, val11).
    """
    offset: int
    exp: int
    precision: float
    val11: float


@dataclass
class ARLIndex:
    """
    A compact index for everything we care about in the ARL file:

    times: list of datetimes (one per time slice we discovered)
    levels_by_time: map time_index → sorted list of level indices available at that time
    uv_by_time_level: map (time_index, level_index) → {'UWND': ARLRecordRef, 'VWND': ARLRecordRef}
    present_levels: sorted list of *all* unique levels observed in the file
    """
    times: List[dt.datetime]
    levels_by_time: Dict[int, List[int]]
    uv_by_time_level: Dict[Tuple[int, int], Dict[str, ARLRecordRef]]
    present_levels: List[int]


class ARLIndexer(QThread):
    """
    A QThread that scans the file so the UI stays responsive.

    Signals:
        finished(ARLIndex): emitted when indexing completes successfully
        error(str): if something goes wrong (e.g., file missing or corrupt)
        progress(str): optional human-readable progress messages
    """
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        """Walk the file record-by-record and build an ARLIndex object."""
        if not os.path.exists(self.path):
            self.error.emit(f"File not found: {self.path}")
            return

        try:
            with open(self.path, "rb") as f:
                size = os.path.getsize(self.path)
                pos = 0

                times: List[dt.datetime] = []
                levels_by_time: Dict[int, List[int]] = {}
                uv_map: Dict[Tuple[int, int], Dict[str, ARLRecordRef]] = {}
                present_levels = set()
                current_ti: Optional[int] = None

                while pos + REC_LEN <= size:
                    f.seek(pos)
                    head = f.read(50)
                    if len(head) < 50:
                        break
                    meta = parse_header(head)
                    var = meta["var"]

                    # ARL convention: an "INDX" record (with fcst_hr != -1) indicates a real time slice
                    if var == "INDX" and meta["fcst_hr"] != -1:
                        if len(times) == 0 or meta["time"] != times[-1]:
                            times.append(meta["time"])
                            current_ti = len(times) - 1
                            levels_by_time[current_ti] = []

                    # For UWND/VWND, record a pointer to the bytes so we can load on demand
                    elif current_ti is not None and meta["fcst_hr"] != -1 and var in ("UWND", "VWND"):
                        lvl = meta["level_index"]
                        present_levels.add(lvl)
                        if lvl not in levels_by_time[current_ti]:
                            levels_by_time[current_ti].append(lvl)
                        uv_map.setdefault((current_ti, lvl), {})[var] = ARLRecordRef(
                            offset=pos, exp=meta["exp"], precision=meta["precision"], val11=meta["val11"]
                        )

                    pos += REC_LEN

                # Sort levels within each time for a tidy UI
                for ti in levels_by_time:
                    levels_by_time[ti].sort()

                index = ARLIndex(
                    times=times,
                    levels_by_time=levels_by_time,
                    uv_by_time_level=uv_map,
                    present_levels=sorted(present_levels),
                )
                self.finished.emit(index)

        except Exception as e:
            self.error.emit(f"ARL indexing failed: {e}")

# -----------------------------------------------------------------------------
# 3) COASTLINE LOADER (OPTIONAL)
# -----------------------------------------------------------------------------
class CoastLoader(QThread):
    """
    Background loader for Natural Earth 50m 'land' polygons.
    If GeoPandas/Shapely/requests are available, we ensure the shapefile exists
    (downloading it if necessary), then extract exterior rings for rendering.
    """
    finished = pyqtSignal(object)  # emits list of rings: each ring is list[(lon, lat), ...]
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def run(self):
        if not GP_OK:
            self.error.emit("Coastlines disabled (install geopandas, shapely, requests to enable).")
            return

        try:
            shp = "ne_50m_land.shp"
            if not os.path.exists(shp):
                self.progress.emit("Downloading Natural Earth 50m land…")
                url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/physical/ne_50m_land.zip"
                r = requests.get(url, stream=True); r.raise_for_status()
                with open("ne_50m_land.zip", "wb") as f:
                    for chunk in r.iter_content(8192): f.write(chunk)
                with zipfile.ZipFile("ne_50m_land.zip","r") as zf:
                    zf.extractall(".")
                os.remove("ne_50m_land.zip")

            self.progress.emit("Reading coastline shapefile…")
            gdf = gpd.read_file(shp)
            rings = []
            for _, row in gdf.iterrows():
                geom = row.geometry
                if isinstance(geom, Polygon):
                    coords = list(geom.exterior.coords)
                    if len(coords) > 3: rings.append(coords[:-1])  # drop duplicate closing vertex
                elif isinstance(geom, MultiPolygon):
                    for poly in geom.geoms:
                        coords = list(poly.exterior.coords)
                        if len(coords) > 3: rings.append(coords[:-1])
            self.finished.emit(rings)

        except Exception as e:
            self.error.emit(f"Coastline load error: {e}")

# -----------------------------------------------------------------------------
# 4) OPENGL GLOBE WIDGET
# -----------------------------------------------------------------------------
class Globe(QGLWidget):
    """
    The heart of the visualizer: an OpenGL widget that draws a globe and wind arrows.
    High-level draw order:
      (a) blue ocean sphere
      (b) optional white land-mask shell (semi-transparent)
      (c) optional coastline lines
      (d) wind arrows (shadow + white)

    TIP: OpenGL state is global. Each method tries to leave GL in a clean state
         so other methods are not surprised.
    """
    latlon_changed = pyqtSignal(float, float)  # emits (lat, lon) under the cursor

    def __init__(self, arl_path: str):
        super().__init__()
        self.arl_path = arl_path
        self.index: Optional[ARLIndex] = None

        # The selected time slice and level (indices into the index tables)
        self.time_idx = 0
        self.level_idx_in_list = 0

        # Camera controls
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.zoom = -5.0  # negative Z = pulled back
        self.dragging = False
        self.last_pos: Optional[QPoint] = None

        # Spheres and shells
        self.sphere_radius = 2.0
        self.mask_radius   = 2.003  # slightly above ocean sphere to avoid z-fighting
        self.coast_radius  = 2.006

        # Optional land-mask texture (RGBA)
        self.mask_tex = None
        self.mask_loaded = False
        self.mask_tex_offset = 0.75  # shift in texture coordinates to align meridians
        self.mask_lat_deg = -90.0    # rotate so texture poles match GL sphere

        # Optional coastline rings
        self.coast_rings: List[List[Tuple[float,float]]] = []

        # Wind render tuning
        self.wind_ref = 38.0          # reference speed for arrow scaling (m/s)
        self.arc_gain = 0.05          # how long the curved arrow segments are
        self.line_smooth_enabled = True

        # Dynamic sampling: more arrows near equator, fewer near poles
        self.equator_samples = 1000   # target samples per equator row
        self.min_samples_per_lat = 6  # never fewer than this
        self.sample_falloff = 1.0     # 0.0 = flat distribution, 1.0 = cos(lat)

        # Cache for decoded U/V arrays keyed by (time_idx, level_index)
        self._cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

        # Cache for prebuilt arrow line geometry keyed by selection + sampling params
        self._wind_mesh_cache = {}

        # Qt basics
        self.setMouseTracking(True)
        self._timer = QTimer(); self._timer.timeout.connect(self.updateGL); self._timer.start(16)

        # Coastlines load in the background (if dependencies exist)
        self.coast_loader = CoastLoader()
        self.coast_loader.finished.connect(self._on_coasts)
        self.coast_loader.error.connect(lambda msg: None)  # silently ignore errors
        self.coast_loader.start()

    # -- Qt/GL lifecycle hooks -------------------------------------------------
    def initializeGL(self):
        """Called once after the GL context is created. Set baseline state here."""
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)
        glClearColor(0.05, 0.05, 0.15, 1.0)

        # Load the land mask only after a GL context exists
        self._maybe_load_mask()

    def resizeGL(self, w, h):
        """Maintain a 45° perspective projection when the widget resizes."""
        if h == 0: h = 1
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); gluPerspective(45.0, w / h, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        """The main draw function — called repeatedly by the timer or interactions."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Camera transform: move back, then apply rotations
        glTranslatef(0.0, 0.0, self.zoom)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)

        # Draw in back-to-front order
        self._draw_ocean()
        if self.mask_loaded: self._draw_mask()
        if self.coast_rings: self._draw_coastlines()
        self._draw_winds()

    # -- Land / coast rendering ------------------------------------------------
    def _maybe_load_mask(self):
        """
        Load an RGBA texture whose alpha channel marks land areas.
        Why not colorize? Keeping it white lets the ocean color show through.
        """
        if self.mask_loaded or not PIL_OK or not os.path.exists(MASK_FILENAME):
            return
        try:
            img = Image.open(MASK_FILENAME)
            # Build RGBA with white color + combined alpha (luminance OR existing alpha)
            if img.mode in ("RGBA", "LA"):
                alpha = ImageChops.lighter(img.getchannel("A"), img.convert("L"))
            else:
                alpha = img.convert("L")
            white = Image.new("L", img.size, 255)
            rgba = Image.merge("RGBA", (white, white, white, alpha))

            # Respect GPU texture size limits
            max_size = int(glGetIntegerv(GL_MAX_TEXTURE_SIZE))
            w, h = rgba.size
            if w > max_size or h > max_size:
                scale = min(max_size / w, max_size / h)
                rgba = rgba.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                w, h = rgba.size

            data = rgba.tobytes("raw", "RGBA", 0, -1)

            # Upload to a GL texture object
            self.mask_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.mask_tex)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glEnable(GL_TEXTURE_2D)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
            glDisable(GL_TEXTURE_2D)
            self.mask_loaded = True
        except Exception:
            self.mask_loaded = False  # quietly give up

    def _draw_mask(self):
        """Render the land-mask as a thin white shell above the ocean sphere."""
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, self.mask_tex)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)

        # Shift the texture longitudinally for alignment
        glMatrixMode(GL_TEXTURE); glPushMatrix(); glLoadIdentity(); glTranslatef(self.mask_tex_offset, 0.0, 0.0)
        glMatrixMode(GL_MODELVIEW)

        glPushMatrix()
        glRotatef(self.mask_lat_deg, 1, 0, 0)  # rotate so the texture poles match the sphere poles
        quad = gluNewQuadric(); gluQuadricTexture(quad, GL_TRUE); gluQuadricNormals(quad, GLU_SMOOTH)
        gluSphere(quad, self.mask_radius, 128, 128); gluDeleteQuadric(quad)
        glPopMatrix()

        glMatrixMode(GL_TEXTURE); glPopMatrix(); glMatrixMode(GL_MODELVIEW)
        glDisable(GL_TEXTURE_2D); glDisable(GL_BLEND); glEnable(GL_CULL_FACE)

    def _on_coasts(self, rings):
        """Qt slot: store rings when CoastLoader finishes."""
        self.coast_rings = rings

    def _split_on_antimeridian(self, ring):
        """
        Lines that cross the 180° meridian can "wrap" across the globe.
        We split the polyline where jumps >180° appear to avoid long back-faces.
        """
        parts, cur = [], [ring[0]]
        prev = ring[0][0]
        for lon, lat in ring[1:]:
            if abs(lon - prev) > 180:
                parts.append(cur); cur = [(lon, lat)]
            else:
                cur.append((lon, lat))
            prev = lon
        if cur: parts.append(cur)
        return parts

    def _draw_coastlines(self):
        """Render coastline polylines a hair above the land-mask shell."""
        glDisable(GL_CULL_FACE)
        glLineWidth(1.2)
        glColor3f(0.07, 0.07, 0.07)
        for ring in self.coast_rings:
            if len(ring) < 2: continue
            for part in self._split_on_antimeridian(ring):
                if len(part) < 2: continue
                glBegin(GL_LINE_STRIP)
                for lon, lat in part:
                    # Our globe uses west-negative longitudes for camera alignment
                    x, y, z = self._latlon_to_xyz(lat, -lon, self.coast_radius)
                    glVertex3f(x, y, z)
                glEnd()
        glEnable(GL_CULL_FACE)

    def _draw_ocean(self):
        """A simple solid-color sphere for the ocean background."""
        glEnable(GL_CULL_FACE)
        glColor3f(0.22, 0.46, 0.72)
        quad = gluNewQuadric(); gluQuadricNormals(quad, GLU_SMOOTH)
        gluSphere(quad, self.sphere_radius, 64, 64)
        gluDeleteQuadric(quad)

    # -- Wind data access + rendering -----------------------------------------
    def _get_current_uv(self) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Fetch (and cache) U and V arrays for the selected (time, level).
        Returns (U, V, level_index) or None if data are not ready.
        """
        if not self.index or not self.index.times:
            return None
        ti = max(0, min(self.time_idx, len(self.index.times)-1))
        levels = self.index.levels_by_time.get(ti, [])
        if not levels:
            return None
        li = max(0, min(self.level_idx_in_list, len(levels)-1))
        lvl = levels[li]
        key = (ti, lvl)
        if key not in self._cache:
            recs = self.index.uv_by_time_level.get(key, {})
            ru, rv = recs.get("UWND"), recs.get("VWND")
            if not ru or not rv:
                return None
            with open(self.arl_path, "rb") as f:
                # read U
                f.seek(ru.offset); f.read(50)
                U = unpack_field(f.read(NX*NY), ru.exp, ru.precision, ru.val11)
                # read V
                f.seek(rv.offset); f.read(50)
                V = unpack_field(f.read(NX*NY), rv.exp, rv.precision, rv.val11)
            self._cache[key] = (U, V)
        U, V = self._cache[key]
        return U, V, lvl

    def _draw_winds(self):
        """
        Build curved polyline arrows that follow tangent directions on the sphere.
        For a point P on the sphere, we build a local tangent basis and step along it.
        We draw a thin black "shadow" under a white line to help contrast.
        """
        res = self._get_current_uv()
        if not res:
            return
        U, V, lvl_index = res

        # Latitude/longitude grids for sampling (1° spacing)
        lats = np.linspace(-90.0, 90.0, NY)
        lons = np.arange(0, 360, 1.0)

        # Visual parameters
        line_w_shadow = 0.9
        line_w_white  = 1.4
        shadow_offset = 0.003
        R_base = self.coast_radius + 0.002

        # Cache key includes sampling settings to reuse geometry if unchanged
        key = (self.time_idx, lvl_index, self.equator_samples, self.min_samples_per_lat, self.sample_falloff)
        mesh = self._wind_mesh_cache.get(key)

        if mesh is None:
            lines_shadow = []
            lines_white  = []
            heads_shadow = []
            heads_white  = []

            lat_sin = np.sin(np.radians(lats))
            lat_cos = np.cos(np.radians(lats))

            for j in range(NY):
                lat = float(lats[j])
                sin_lat = float(lat_sin[j])
                cos_lat = float(lat_cos[j])

                # More samples near equator (cosine falloff toward poles)
                coslat = max(0.0, math.cos(math.radians(abs(lat))))
                target = self.equator_samples * (coslat ** self.sample_falloff)
                samples = max(self.min_samples_per_lat, int(round(target)))
                stride_lon = max(1, int(round(NX / samples)))

                for i in range(0, NX, stride_lon):
                    lon_e = float(lons[i])        # 0..360 East
                    lon = -(((lon_e + 180) % 360) - 180)  # convert to west-negative

                    uu = float(U[j, i])
                    vv = float(V[j, i])
                    spd = math.hypot(uu, vv)
                    if spd < 0.08:  # skip calm points
                        continue

                    # Position on unit sphere in world space
                    lon_r = math.radians(lon)
                    px = cos_lat * math.cos(lon_r)
                    py = sin_lat
                    pz = cos_lat * math.sin(lon_r)

                    # Local tangent frame at P:
                    # ex = east direction tangent, n = north direction tangent
                    ex, ey, ez = -math.sin(lon_r), 0.0, math.cos(lon_r)
                    nx, ny, nz = -sin_lat*math.cos(lon_r), cos_lat, -sin_lat*math.sin(lon_r)

                    # Wind vector in tangent plane (combine U & V onto ex and n)
                    vx = uu * ex + vv * nx
                    vy = uu * ey + vv * ny
                    vz = uu * ez + vv * nz
                    vmag = math.sqrt(vx*vx + vy*vy + vz*vz)
                    if vmag == 0.0:
                        continue
                    tx, ty, tz = vx/vmag, vy/vmag, vz/vmag  # normalized tangent direction

                    # Make a short curved arc along great-circle-ish path
                    s = self.arc_gain * (spd / self.wind_ref)  # radians along the unit sphere
                    s = max(0.005, min(s, 0.18))               # clamp to nice visuals
                    segs = 6                                    # fixed segments per arrow

                    prev = None
                    for k in range(segs+1):
                        a = s * (k / segs)
                        # Rotate around tangent direction (small-angle approximation)
                        cx = px*math.cos(a) + tx*math.sin(a)
                        cy = py*math.cos(a) + ty*math.sin(a)
                        cz = pz*math.cos(a) + tz*math.sin(a)
                        # renormalize back to the sphere
                        inv = 1.0 / math.sqrt(cx*cx + cy*cy + cz*cz)
                        cx *= inv; cy *= inv; cz *= inv
                        # project to two radii (shadow and white)
                        sx, sy, sz = (R_base + shadow_offset)*cx, (R_base + shadow_offset)*cy, (R_base + shadow_offset)*cz
                        wx, wy, wz = R_base*cx, R_base*cy, R_base*cz
                        if prev is not None:
                            psx, psy, psz, pwx, pwy, pwz = prev
                            lines_shadow.extend([psx, psy, psz, sx, sy, sz])
                            lines_white.extend([pwx, pwy, pwz, wx, wy, wz])
                        prev = (sx, sy, sz, wx, wy, wz)

                    # Arrowhead: two short lines at the arc tip
                    pxs = px*math.cos(s) + tx*math.sin(s)
                    pys = py*math.cos(s) + ty*math.sin(s)
                    pzs = pz*math.cos(s) + tz*math.sin(s)
                    inv = 1.0 / math.sqrt(pxs*pxs + pys*pys + pzs*pzs)
                    pxs*=inv; pys*=inv; pzs*=inv

                    txs = -px*math.sin(s) + tx*math.cos(s)
                    tys = -py*math.sin(s) + ty*math.cos(s)
                    tzs = -pz*math.sin(s) + tz*math.cos(s)
                    inv = 1.0 / math.sqrt(txs*txs + tys*tys + tzs*tzs)
                    txs*=inv; tys*=inv; tzs*=inv

                    # orthogonal for the "V" head
                    cxs = tys*pzs - tzs*pys
                    cys = tzs*pxs - txs*pzs
                    czs = txs*pys - tys*pxs
                    inv = math.sqrt(cxs*cxs + cys*cys + czs*czs) or 1.0
                    cxs/=inv; cys/=inv; czs/=inv

                    head_len = min(0.06, 0.45 * s)

                    # shadow
                    ax, ay, az = (R_base+shadow_offset)*pxs, (R_base+shadow_offset)*pys, (R_base+shadow_offset)*pzs
                    lx = ax - head_len*(0.9*txs + 0.5*cxs)
                    ly = ay - head_len*(0.9*tys + 0.5*cys)
                    lz = az - head_len*(0.9*tzs + 0.5*czs)
                    rx = ax - head_len*(0.9*txs - 0.5*cxs)
                    ry = ay - head_len*(0.9*tys - 0.5*cys)
                    rz = az - head_len*(0.9*tzs - 0.5*czs)
                    heads_shadow.extend([ax,ay,az, lx,ly,lz,  ax,ay,az, rx,ry,rz])

                    # white
                    ax, ay, az = R_base*pxs, R_base*pys, R_base*pzs
                    lx = ax - head_len*(0.9*txs + 0.5*cxs)
                    ly = ay - head_len*(0.9*tys + 0.5*cys)
                    lz = az - head_len*(0.9*tzs + 0.5*czs)
                    rx = ax - head_len*(0.9*txs - 0.5*cxs)
                    ry = ay - head_len*(0.9*tys - 0.5*cys)
                    rz = az - head_len*(0.9*tzs - 0.5*czs)
                    heads_white.extend([ax,ay,az, lx,ly,lz,  ax,ay,az, rx,ry,rz])

            mesh = {
                "lines_shadow": np.array(lines_shadow, dtype=np.float32),
                "lines_white":  np.array(lines_white,  dtype=np.float32),
                "heads_shadow": np.array(heads_shadow, dtype=np.float32),
                "heads_white":  np.array(heads_white,  dtype=np.float32),
            }
            self._wind_mesh_cache[key] = mesh

        # Actual OpenGL drawing using client-side arrays
        glDisable(GL_CULL_FACE)
        if self.line_smooth_enabled:
            glEnable(GL_LINE_SMOOTH); glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        else:
            glDisable(GL_LINE_SMOOTH)

        glEnableClientState(GL_VERTEX_ARRAY)

        if mesh["lines_shadow"].size:
            glColor3f(0.0, 0.0, 0.0); glLineWidth(line_w_shadow)
            glVertexPointer(3, GL_FLOAT, 0, mesh["lines_shadow"])
            glDrawArrays(GL_LINES, 0, mesh["lines_shadow"].size // 3)

        if mesh["heads_shadow"].size:
            glColor3f(0.0, 0.0, 0.0); glLineWidth(line_w_shadow)
            glVertexPointer(3, GL_FLOAT, 0, mesh["heads_shadow"])
            glDrawArrays(GL_LINES, 0, mesh["heads_shadow"].size // 3)

        if mesh["lines_white"].size:
            glColor3f(1.0, 1.0, 1.0); glLineWidth(line_w_white)
            glVertexPointer(3, GL_FLOAT, 0, mesh["lines_white"])
            glDrawArrays(GL_LINES, 0, mesh["lines_white"].size // 3)

        if mesh["heads_white"].size:
            glColor3f(1.0, 1.0, 1.0); glLineWidth(line_w_white)
            glVertexPointer(3, GL_FLOAT, 0, mesh["heads_white"])
            glDrawArrays(GL_LINES, 0, mesh["heads_white"].size // 3)

        glDisableClientState(GL_VERTEX_ARRAY)
        glEnable(GL_CULL_FACE)

    # -- Interaction (mouse/keyboard) -----------------------------------------
    def mousePressEvent(self, e):
        """Start left-button dragging to rotate the globe. Turn off smoothing while dragging for speed."""
        if e.button() == Qt.LeftButton:
            self.dragging = True; self.last_pos = e.pos()
            self.line_smooth_enabled = False

    def mouseReleaseEvent(self, e):
        """Stop dragging; re-enable line smoothing."""
        if e.button() == Qt.LeftButton:
            self.dragging = False
            self.line_smooth_enabled = True

    def mouseMoveEvent(self, e):
        """If dragging, update rotations. Always emit the lat/lon under the cursor if we hit the globe."""
        if self.dragging and self.last_pos:
            dx = e.x() - self.last_pos.x(); dy = e.y() - self.last_pos.y()
            self.rotation_y += dx * 0.5
            self.rotation_x = max(-90, min(90, self.rotation_x + dy * 0.5))
            self.last_pos = e.pos(); self.updateGL()
        self.updateGL()
        lat, lon = self._latlon_under_cursor(e.x(), e.y())
        if lat is not None: self.latlon_changed.emit(lat, lon)

    def keyPressEvent(self, e):
        """Keyboard zoom shortcuts for accessibility (also see wheelEvent)."""
        key = e.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal, Qt.Key_Up):
            self.zoom = min(-0.5, self.zoom + 0.5)
        elif key in (Qt.Key_Minus, Qt.Key_Underscore, Qt.Key_Down):
            self.zoom = max(-20.0, self.zoom - 0.5)
        elif key == Qt.Key_PageUp:
            self.zoom = min(-0.5, self.zoom + 1.0)
        elif key == Qt.Key_PageDown:
            self.zoom = max(-20.0, self.zoom - 1.0)
        else:
            return super().keyPressEvent(e)
        self.updateGL()

    def wheelEvent(self, e):
        """Smooth mouse-wheel/trackpad zoom."""
        steps = e.angleDelta().y() / 120.0
        if steps == 0: return
        self.zoom += -0.4 * steps
        self.zoom = max(-20.0, min(-0.5, self.zoom))
        self.updateGL()

    # -- Math helpers ----------------------------------------------------------
    def _latlon_to_xyz(self, lat, lon, R):
        """Convert (lat, lon in degrees; lon east-positive) to Cartesian coordinates on radius R sphere."""
        lat_r = math.radians(lat); lon_r = math.radians(lon)
        return (R*math.cos(lat_r)*math.cos(lon_r),
                R*math.sin(lat_r),
                R*math.cos(lat_r)*math.sin(lon_r))

    def _latlon_under_cursor(self, mx, my):
        """
        Ray-cast from the camera through the mouse position and intersect with the sphere.
        If we hit, return (lat, lon) in degrees (lon is west-negative to match rendering).
        """
        model = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        winX = float(mx); winY = float(viewport[3] - my)

        near = gluUnProject(winX, winY, 0.0, model, proj, viewport)
        far  = gluUnProject(winX, winY, 1.0, model, proj, viewport)
        if near is None or far is None:
            return None, None

        nx, ny, nz = near; fx, fy, fz = far
        dx, dy, dz = fx - nx, fy - ny, fz - nz
        A = dx*dx + dy*dy + dz*dz
        B = 2*(nx*dx + ny*dy + nz*dz)
        C = nx*nx + ny*ny + nz*nz - (self.sphere_radius * self.sphere_radius)
        disc = B*B - 4*A*C
        if disc < 0:
            return None, None
        t = (-B - math.sqrt(disc)) / (2*A)
        if t < 0:
            t = (-B + math.sqrt(disc)) / (2*A)
            if t < 0:
                return None, None

        ix, iy, iz = nx + t*dx, ny + t*dy, nz + t*dz
        r = math.sqrt(ix*ix + iy*iy + iz*iz) or 1.0
        lat = math.degrees(math.asin(iy / r))
        lon = math.degrees(math.atan2(iz, ix))  # east-positive
        lon = ((-lon + 180.0) % 360.0) - 180.0  # flip to west-negative for display
        return lat, lon

# -----------------------------------------------------------------------------
# 5) MAIN WINDOW (PyQt GUI)
# -----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    """
    A basic Qt window that embeds the Globe widget and adds UI controls:
      • A top info bar showing current counts and cache info
      • Bottom sliders to change the time index and pressure level
      • A status bar with live cursor lat/lon and interpolated wind at cursor
    """
    def __init__(self, path: str):
        super().__init__()
        self.setWindowTitle("GDAS ARL Global Winds")
        self.resize(1200, 900)

        # Central OpenGL view
        self.gl = Globe(path)

        # --- Layout -----------------------------------------------------------
        central = QWidget(); layout = QVBoxLayout(central); layout.setSpacing(0); self.setCentralWidget(central)

        # Info bar (text-only; refreshed by a timer)
        self.info = QLabel("Indexing ARL…")
        self.info.setStyleSheet("background:#f7f7f7;padding:6px;border-bottom:1px solid #ccc;")
        layout.addWidget(self.info)

        # The globe occupies most of the window
        layout.addWidget(self.gl, 1)

        # Controls row: time + level sliders with labels
        row = QWidget(); row_l = QHBoxLayout(row); row_l.setContentsMargins(8,8,8,8)
        self.tlabel = QLabel("Time: —")
        self.tslider = QSlider(Qt.Horizontal); self.tslider.setMinimum(0); self.tslider.valueChanged.connect(self._on_time)
        self.llabel = QLabel("Level: — hPa")
        self.lslider = QSlider(Qt.Horizontal); self.lslider.setMinimum(0); self.lslider.valueChanged.connect(self._on_level)
        row_l.addWidget(self.tlabel); row_l.addWidget(self.tslider, 1); row_l.addSpacing(16)
        row_l.addWidget(self.llabel); row_l.addWidget(self.lslider, 1)
        layout.addWidget(row)

        # Status bar shows live cursor readout and the current time/level
        sb: QStatusBar = self.statusBar()
        self.latlon_txt = QLabel("Lat: —  Lon: —"); self.latlon_txt.setTextInteractionFlags(Qt.TextSelectableByMouse|Qt.TextSelectableByKeyboard)
        self.whenlev_txt = QLabel("Time: —  Level: — hPa"); self.whenlev_txt.setTextInteractionFlags(Qt.TextSelectableByMouse|Qt.TextSelectableByKeyboard)
        sb.addPermanentWidget(self.latlon_txt, 1); sb.addPermanentWidget(self.whenlev_txt, 1)

        # Wire the globe's cursor signal to our readout
        self.gl.latlon_changed.connect(self._on_mouse_latlon)

        # Kick off the background indexer
        self.indexer = ARLIndexer(path)
        self.indexer.error.connect(self._on_index_error)
        self.indexer.finished.connect(self._on_index_ready)
        self.indexer.start()

        # Periodic info bar refresh (every 0.5s)
        self._timer = QTimer(); self._timer.timeout.connect(self._refresh_info); self._timer.start(500)

    # -- Indexing callbacks ----------------------------------------------------
    def _on_index_error(self, msg: str):
        self.info.setText(f"Error: {msg}")

    def _on_index_ready(self, index: ARLIndex):
        """Receive the ARLIndex, configure sliders, and show an initial status."""
        self.gl.index = index
        nt = len(index.times)
        self.tslider.setMaximum(max(0, nt-1))
        self._update_time_label()

        # For the first time index, configure the level slider
        levs = index.levels_by_time.get(0, [])
        self.lslider.setMaximum(max(0, len(levs)-1))
        self.gl.level_idx_in_list = 0
        self._update_level_label()
        self._update_whenlev_status()

        # Summarize what we loaded
        self.info.setText(f"Loaded: {nt} times — levels present: {index.present_levels[:8]}{'…' if len(index.present_levels)>8 else ''}")

    # -- Slider handlers -------------------------------------------------------
    def _on_time(self, value: int):
        """Update the active time slice and adjust level slider bounds accordingly."""
        self.gl.time_idx = value
        levs = self.gl.index.levels_by_time.get(self.gl.time_idx, []) if self.gl.index else []
        self.lslider.setMaximum(max(0, len(levs)-1))
        self.gl.level_idx_in_list = min(self.gl.level_idx_in_list, max(0, len(levs)-1))
        self._update_time_label(); self._update_level_label(); self._update_whenlev_status()

    def _on_level(self, value: int):
        """Choose a level within the current time slice."""
        self.gl.level_idx_in_list = value
        self._update_level_label(); self._update_whenlev_status()

    # -- Labels / status helpers ----------------------------------------------
    def _update_time_label(self):
        if not self.gl.index or not self.gl.index.times:
            self.tlabel.setText("Time: —"); return
        t = self.gl.index.times[self.gl.time_idx]
        self.tlabel.setText(f"Time: {t.isoformat()}")

    def _update_level_label(self):
        if not self.gl.index:
            self.llabel.setText("Level: — hPa"); return
        levs = self.gl.index.levels_by_time.get(self.gl.time_idx, [])
        if not levs:
            self.llabel.setText("Level: — hPa"); return
        lvl = levs[self.gl.level_idx_in_list]
        hpa = LEVEL_TO_HPA.get(lvl)
        self.llabel.setText(f"Level: {hpa if hpa else f'idx {lvl}'} hPa")

    def _update_whenlev_status(self):
        if not self.gl.index or not self.gl.index.times:
            self.whenlev_txt.setText("Time: —  Level: — hPa"); return
        t = self.gl.index.times[self.gl.time_idx]
        levs = self.gl.index.levels_by_time.get(self.gl.time_idx, [])
        if not levs:
            self.whenlev_txt.setText(f"Time: {t.isoformat()}  Level: — hPa"); return
        lvl = levs[self.gl.level_idx_in_list]
        hpa = LEVEL_TO_HPA.get(lvl)
        self.whenlev_txt.setText(f"Time: {t.isoformat()}  Level: {hpa if hpa else f'idx {lvl} hPa'}")

    def _refresh_info(self):
        """Lightweight dashboard of what's loaded and toggled."""
        if not self.gl.index: return
        nt = len(self.gl.index.times)
        levs = self.gl.index.levels_by_time.get(self.gl.time_idx, [])
        mask = "on" if self.gl.mask_loaded else "off"
        self.info.setText(f"Times: {nt} | Levels now: {len(levs)} | Cache: {len(self.gl._cache)} | Land mask: {mask} | Coasts: {len(self.gl.coast_rings)}")

    # -- Cursor readout: bilinear interpolation at the mouse ------------------
    def _interp_uv(self, lat: float, lon: float, U, V):
        """
        Bilinear interpolation on the 1° grid. Our displayed longitudes are
        west-negative, but the grid is 0..360 East, so we convert first.
        """
        lon_e = (-lon) % 360.0
        x = lon_e
        y = ((lat + 90.0) / 180.0) * (NY - 1)
        i0 = int(x) % 360; i1 = (i0 + 1) % 360
        j0 = int(y); j1 = min(NY - 1, j0 + 1)
        tx = x - int(x); ty = y - j0
        u00 = float(U[j0, i0]); v00 = float(V[j0, i0])
        u10 = float(U[j0, i1]); v10 = float(V[j0, i1])
        u01 = float(U[j1, i0]); v01 = float(V[j1, i0])
        u11 = float(U[j1, i1]); v11 = float(V[j1, i1])
        u0 = u00 * (1.0 - tx) + u10 * tx
        u1 = u01 * (1.0 - tx) + u11 * tx
        v0 = v00 * (1.0 - tx) + v10 * tx
        v1 = v01 * (1.0 - tx) + v11 * tx
        u = u0 * (1.0 - ty) + u1 * ty
        v = v0 * (1.0 - ty) + v1 * ty
        return u, v

    def _on_mouse_latlon(self, lat: float, lon: float):
        """Update the status bar with precise cursor lat/lon and local wind (mph + bearing)."""
        self.latlon_txt.setText(f"Lat: {lat:.5f}  Lon: {lon:.5f}")
        try:
            res = self.gl._get_current_uv()
            if not res:
                return
            U, V, _lvl = res
            uu, vv = self._interp_uv(lat, lon, U, V)
            speed_ms = math.hypot(uu, vv)
            speed_mph = speed_ms * 2.236936
            theta = math.degrees(math.atan2(-uu, -vv))  # meteorological: from-N clockwise
            direction_deg = (theta + 360.0) % 360.0
            self.latlon_txt.setText(
                f"Lat: {lat:.5f}  Lon: {lon:.5f}   Dir: {direction_deg:6.1f}°   Spd: {speed_mph:6.2f} mph"
            )
        except Exception:
            pass

# -----------------------------------------------------------------------------
# 6) ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    """
    Run the Qt application. If you pass a path to a .w4 file as the first
    command-line argument, we’ll open that; otherwise we try DEFAULT_FILE.
    """
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    app = QApplication(sys.argv)
    w = MainWindow(path); w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
