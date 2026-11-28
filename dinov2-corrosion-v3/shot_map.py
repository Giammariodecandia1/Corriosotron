"""Static map of the shooting point (optional, for the PDF report).

The coordinates come only from the photo's EXIF GPS (photo_meta). Map tiles are taken from
OpenStreetMap (internet needed); only the tile coordinates are sent, never the photo. Tiles are
cached on disk, so the same area is downloaded once. The map must keep the OSM attribution.
"""
import io
import math
import threading
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
TILE_DIR = ROOT / "model_cache" / "osm_tiles"
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "Rustron-CorrosionIntelligence/1.0 (local inspection tool; occasional single-point maps)"
ATTRIBUTION = "© OpenStreetMap contributors"
_LOCK = threading.Lock()


def google_maps_url(lat, lon):
    return f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"


def _tile(z, x, y, timeout):
    path = TILE_DIR / str(z) / str(x) / f"{y}.png"
    if path.is_file():
        return Image.open(path).convert("RGB")
    req = urllib.request.Request(TILE_URL.format(z=z, x=x, y=y), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return Image.open(io.BytesIO(data)).convert("RGB")


def static_map(lat, lon, width=640, height=440, zoom=17, timeout=8):
    """Return (PIL image, None) or (None, reason). Marker at the exact point, attribution included."""
    try:
        lat_r = math.radians(max(-85.0, min(85.0, lat)))
        n = 2 ** zoom
        px = (lon + 180.0) / 360.0 * n * 256
        py = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n * 256
        x0, y0 = px - width / 2, py - height / 2
        out = Image.new("RGB", (width, height), (230, 230, 230))
        with _LOCK:
            for tx in range(int(x0 // 256), int((x0 + width) // 256) + 1):
                for ty in range(int(y0 // 256), int((y0 + height) // 256) + 1):
                    if 0 <= ty < n:
                        out.paste(_tile(zoom, tx % n, ty, timeout), (int(tx * 256 - x0), int(ty * 256 - y0)))
    except Exception as exc:  # no internet, server refused, ...
        return None, f"mappa non scaricata ({type(exc).__name__}: {str(exc)[:80]})"
    d = ImageDraw.Draw(out)
    cx, cy = width / 2, height / 2
    # pin: white halo, red disc, white centre
    d.ellipse((cx - 11, cy - 11, cx + 11, cy + 11), fill=(255, 255, 255))
    d.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=(220, 40, 60))
    d.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 255, 255))
    tw = d.textlength(ATTRIBUTION)
    d.rectangle((width - tw - 10, height - 16, width, height), fill=(255, 255, 255))
    d.text((width - tw - 5, height - 14), ATTRIBUTION, fill=(60, 60, 60))
    return out, None
