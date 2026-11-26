"""Shot date/time and GPS position read from the photo itself.

Sources, in order:
- date/time: EXIF DateTimeOriginal, DateTimeDigitized, DateTime; then a timestamp in the
  file name (e.g. "Camera_6_2026-05-08_10.04.57.000.jpg") as fallback.
- coordinates: EXIF GPS latitude/longitude. Nothing is typed by the operator.
"""
import re
from datetime import datetime

EXIF_IFD = 0x8769          # Exif sub-IFD
GPS_IFD = 0x8825           # GPS sub-IFD
DATETIME = 0x0132          # IFD0 DateTime
DATETIME_ORIGINAL = 0x9003
DATETIME_DIGITIZED = 0x9004
GPS_LAT_REF, GPS_LAT, GPS_LON_REF, GPS_LON = 1, 2, 3, 4

FILENAME_TIME = re.compile(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})\.(\d{2})\.(\d{2})")


def _exif_datetime(text):
    try:
        return datetime.strptime(str(text).strip()[:19], "%Y:%m:%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def _to_degrees(value, ref):
    d, m, s = (float(x) for x in value)
    degrees = d + m / 60 + s / 3600
    return -degrees if str(ref).upper() in ("S", "W") else degrees


def _gps(exif):
    try:
        gps = exif.get_ifd(GPS_IFD)
    except (AttributeError, KeyError, ValueError, TypeError):
        return None
    try:
        lat = _to_degrees(gps[GPS_LAT], gps[GPS_LAT_REF])
        lon = _to_degrees(gps[GPS_LON], gps[GPS_LON_REF])
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return None
    if lat == 0 and lon == 0:
        return None
    return lat, lon


def format_coordinates(lat, lon):
    """Italian decimal style, with hemisphere letters: 45,123456 N, 9,123456 E."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.6f}".replace(".", ",") + f" {ns}, " + f"{abs(lon):.6f}".replace(".", ",") + f" {ew}"


def read_shot_meta(image, filename=""):
    """Return {"captured_at", "date_source", "coordinates"}; missing values are None.

    `image` must be the original, not yet rotated, PIL image so EXIF is intact.
    """
    captured, source = None, None
    try:
        exif = image.getexif()
    except (AttributeError, ValueError, OSError):
        exif = {}
    try:
        sub = exif.get_ifd(EXIF_IFD)
    except (AttributeError, KeyError, ValueError, TypeError):
        sub = {}
    for store, tag in ((sub, DATETIME_ORIGINAL), (sub, DATETIME_DIGITIZED), (exif, DATETIME)):
        value = store.get(tag) if store else None
        parsed = _exif_datetime(value) if value else None
        if parsed:
            captured, source = parsed, "EXIF"
            break
    if captured is None:
        match = FILENAME_TIME.search(filename or "")
        if match:
            captured = datetime(*(int(g) for g in match.groups()))
            source = "nome file"
    gps = _gps(exif) if exif else None
    return {
        "captured_at": captured.strftime("%d/%m/%Y %H:%M:%S") if captured else None,
        "date_source": source,
        "coordinates": format_coordinates(*gps) if gps else None,
        "latitude": round(gps[0], 7) if gps else None,
        "longitude": round(gps[1], 7) if gps else None,
    }
