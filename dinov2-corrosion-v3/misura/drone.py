"""Riconoscimento del drone dai metadati della foto e scelta del profilo camera.

I metadati si leggono dai byte originali della foto (``meta_source`` se c'è), ma gli intrinseci
si scalano sulla dimensione dell'immagine **analizzata**, che può essere una copia ridotta.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

from .distix.calibration_manager import parse_dewarp_data
from .distix.metadata_reader import parse_dji_xmp, parse_image_exif

ROOT = Path(__file__).resolve().parents[1]
CARTELLA_PROFILI = ROOT / "profili_drone"
DIAGONALE_35MM = 43.267  # mm, diagonale del fotogramma 36×24 (definizione CIPA dell'equivalente 35 mm)
TOLLERANZA_ZOOM = 0.05

@dataclass
class FotoMeta:
    make: Optional[str] = None
    model: Optional[str] = None
    image_source: Optional[str] = None
    width: Optional[int] = None          # immagine analizzata
    height: Optional[int] = None
    native_w: Optional[int] = None       # file da cui si leggono i metadati
    native_h: Optional[int] = None
    orientation: Optional[int] = None
    focal_mm: Optional[float] = None
    focal35_mm: Optional[float] = None
    digital_zoom: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    abs_alt: Optional[float] = None
    rel_alt: Optional[float] = None
    gimbal_pitch: Optional[float] = None
    gimbal_yaw: Optional[float] = None
    gimbal_roll: Optional[float] = None
    rtk_flag: Optional[int] = None
    rtk_std_h: Optional[float] = None
    rtk_std_v: Optional[float] = None
    lrf_status: Optional[str] = None
    lrf_m: Optional[float] = None
    dewarp: Optional[str] = None
    xmp: dict = field(default_factory=dict)

@dataclass
class Profilo:
    id: str
    nome: str
    make: Optional[str]
    model: Optional[str]
    focal_mm: Optional[float]
    sensor_w_mm: Optional[float]
    native_w: Optional[int]
    native_h: Optional[int]
    has_rtk: bool
    has_lrf: bool
    lrf_range_m: tuple
    sigma_gimbal_deg: float
    sigma_focal_rel: float
    calibrazione: Optional[dict]
    generico: bool = False
    image_source: Optional[str] = None

    @classmethod
    def generico_default(cls, meta: Optional[FotoMeta] = None) -> "Profilo":
        nome = " ".join(x for x in ((meta.make if meta else None), (meta.model if meta else None)) if x) or "drone sconosciuto"
        return cls(id="generico", nome=f"{nome} · profilo generico", make=meta.make if meta else None,
                   model=meta.model if meta else None, focal_mm=None, sensor_w_mm=None, native_w=None, native_h=None,
                   has_rtk=False, has_lrf=False, lrf_range_m=(3.0, 1200.0), sigma_gimbal_deg=2.0,
                   sigma_focal_rel=0.05, calibrazione=None, generico=True)

def _num(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None

def _int(value) -> Optional[int]:
    out = _num(value)
    return int(out) if out is not None else None

def _gps_exif(gps: dict) -> tuple[Optional[float], Optional[float]]:
    def deg(v, ref):
        try:
            d, m, s = (float(x) for x in v)
        except (TypeError, ValueError):
            return None
        out = d + m / 60 + s / 3600
        return -out if str(ref).upper().startswith(("S", "W")) else out
    lat = deg(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef", "N"))
    lon = deg(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef", "E"))
    return lat, lon

def leggi_foto(meta_bytes: bytes, analysed_size: tuple[int, int]) -> FotoMeta:
    """Legge EXIF + XMP DJI dai byte della foto. Valori mancanti = None, mai inventati."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(meta_bytes)
        xmp = parse_dji_xmp(path)
        try:
            native_w, native_h, exif, gps = parse_image_exif(path)
        except Exception:
            native_w = native_h = None
            exif, gps = {}, {}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    lat, lon = _gps_exif(gps) if gps else (None, None)
    if lat is None or lon is None:
        lat = _num(xmp.get("GpsLatitude"))
        lon = _num(xmp.get("GpsLongitude") or xmp.get("GpsLongtitude"))
    rtk_lat, rtk_lon = _num(xmp.get("RtkStdLat")), _num(xmp.get("RtkStdLon"))
    std_h = max(v for v in (rtk_lat, rtk_lon) if v is not None) if (rtk_lat is not None or rtk_lon is not None) else None
    return FotoMeta(
        make=exif.get("Make") or None, model=exif.get("Model") or None,
        image_source=xmp.get("ImageSource"),
        width=int(analysed_size[0]), height=int(analysed_size[1]), native_w=native_w, native_h=native_h,
        orientation=exif.get("Orientation"),
        focal_mm=exif.get("FocalLength"), focal35_mm=exif.get("FocalLengthIn35mmFilm"),
        digital_zoom=exif.get("DigitalZoomRatio") or _num(xmp.get("DigitalZoomRatio")),
        lat=lat, lon=lon, abs_alt=_num(xmp.get("AbsoluteAltitude")), rel_alt=_num(xmp.get("RelativeAltitude")),
        gimbal_pitch=_num(xmp.get("GimbalPitchDegree")), gimbal_yaw=_num(xmp.get("GimbalYawDegree")),
        gimbal_roll=_num(xmp.get("GimbalRollDegree")),
        rtk_flag=_int(xmp.get("RtkFlag")), rtk_std_h=std_h, rtk_std_v=_num(xmp.get("RtkStdHgt")),
        lrf_status=xmp.get("LRFStatus"), lrf_m=_num(xmp.get("LRFTargetDistance")),
        dewarp=xmp.get("DewarpData"), xmp=xmp)

def _carica_profili(cartella: Path) -> list[Profilo]:
    nomi = {f.name for f in fields(Profilo)}
    out = []
    for path in sorted(Path(cartella).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["lrf_range_m"] = tuple(data.get("lrf_range_m") or (3.0, 1200.0))
            out.append(Profilo(**{k: v for k, v in data.items() if k in nomi}))
        except (OSError, ValueError, TypeError):
            continue  # un file rotto non deve fermare la misura: si torna al profilo generico
    return out

def scegli_profilo(meta: FotoMeta, cartella: Path = CARTELLA_PROFILI) -> Profilo:
    make, model = (meta.make or "").strip().lower(), (meta.model or "").strip().lower()
    if make and model:
        for p in _carica_profili(cartella):
            if (p.make or "").lower() == make and (p.model or "").lower() == model:
                return p
    return Profilo.generico_default(meta)

DISTORSIONE = ("k1", "k2", "p1", "p2", "k3")


def intrinseci(meta: FotoMeta, profilo: Profilo) -> Optional[dict]:
    """fx, fy, cx, cy in pixel dell'immagine analizzata, fonte, distorsione e se la camera è calibrata.

    «calibrata» vale solo per una calibrazione completa dal profilo (anche k1, k2, p1, p2, k3, eventualmente 0)
    oppure per la calibrazione di fabbrica DJI (DewarpData). La distorsione si applica ai raggi solo se
    l'immagine non è già raddrizzata (DewarpFlag diverso da 1). None se la focale non è ricavabile.
    """
    w, h = meta.width, meta.height
    if not w or not h:
        return None
    cal = profilo.calibrazione
    if cal and cal.get("fx"):
        s = w / float(cal.get("image_width") or profilo.native_w or w)
        completa = all(cal.get(k) is not None for k in ("fx", "fy", "cx", "cy") + DISTORSIONE)
        dist = tuple(float(cal[k]) for k in DISTORSIONE) if completa else None
        return {"fx": cal["fx"] * s, "fy": cal.get("fy", cal["fx"]) * s, "cx": cal.get("cx", w / s / 2) * s,
                "cy": cal.get("cy", h / s / 2) * s, "fonte": "calibrazione", "calibrata": completa, "dist": dist}
    if meta.dewarp and meta.native_w and meta.native_h:
        d = parse_dewarp_data(meta.dewarp, meta.native_w, meta.native_h)
        if d:
            s = w / meta.native_w
            raddrizzata = str(meta.xmp.get("DewarpFlag", "")).strip() == "1"
            return {"fx": d["fx"] * s, "fy": d["fy"] * s, "cx": d["cx"] * s, "cy": d["cy"] * s, "fonte": "dewarp",
                    "calibrata": True, "dist": None if raddrizzata else tuple(float(d[k]) for k in DISTORSIONE)}
    if meta.focal35_mm:
        fx = meta.focal35_mm * math.hypot(w, h) / DIAGONALE_35MM
        return {"fx": fx, "fy": fx, "cx": w / 2, "cy": h / 2, "fonte": "35mm", "calibrata": False, "dist": None}
    if meta.focal_mm and profilo.sensor_w_mm:
        fx = meta.focal_mm / profilo.sensor_w_mm * w
        return {"fx": fx, "fy": fx, "cx": w / 2, "cy": h / 2, "fonte": "sensore", "calibrata": False, "dist": None}
    return None

def etichetta_rtk(meta: FotoMeta) -> str:
    stato = {50: "RTK FIX", 34: "RTK FLOAT", 16: "GNSS singolo", 0: "senza RTK"}.get(meta.rtk_flag, "RTK sconosciuto")
    if meta.rtk_std_h is not None:
        return f"{stato} · ±{max(1, round(meta.rtk_std_h * 100))} cm"
    return stato


def misurabile(meta: FotoMeta, profilo: Profilo) -> tuple[bool, str]:
    if (meta.gimbal_pitch is None or meta.gimbal_yaw is None or meta.rel_alt is None
            or meta.lat is None or meta.lon is None):
        return False, "metadati di posizione/gimbal mancanti"
    if intrinseci(meta, profilo) is None:
        return False, "metadati della camera mancanti: focale non ricavabile"
    if meta.orientation not in (None, 1):
        return False, "foto ruotata: usare l'originale del drone"
    src = (meta.image_source or "").lower()
    if "infrared" in src or "thermal" in src:
        return False, "foto termica: si misura solo con la camera grandangolare"
    zoom = "zoom attivo: si misura solo con la camera grandangolare senza zoom"
    if "zoom" in src or "tele" in src:
        return False, zoom
    if meta.digital_zoom is not None and meta.digital_zoom > 1.01:
        return False, zoom
    if not profilo.generico and profilo.focal_mm and meta.focal_mm:
        if abs(meta.focal_mm / profilo.focal_mm - 1) > TOLLERANZA_ZOOM:
            return False, zoom
    return True, ""
