"""
DISTIX - GNSS / RTK Quality Validator
Valutazione rigorosa della qualità GNSS/RTK/PPK registrata nei metadati.
Non applica mai rtk_fix=True forzato e non inventa deviazioni standard mancanti.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


# Codici standard DJI RtkFlag / MRK Q-flag
RTK_FLAG_NONE = 0
RTK_FLAG_SINGLE = 16
RTK_FLAG_FLOAT = 34
RTK_FLAG_FIX = 50


def rtk_flag_label(flag: Optional[int]) -> str:
    if flag == RTK_FLAG_FIX:
        return "FIX (Centimetrico)"
    if flag == RTK_FLAG_FLOAT:
        return "FLOAT (Decimetrico)"
    if flag == RTK_FLAG_SINGLE:
        return "SINGLE / GPS Autonomo (No RTK)"
    if flag == RTK_FLAG_NONE:
        return "NESSUN POSIZIONAMENTO"
    if flag is None:
        return "SCONOSCIUTO"
    return f"ALTRO (Flag={flag})"


@dataclass
class GNSSQuality:
    """Stato e tracciabilità completa del posizionamento GNSS della camera."""

    mode: str = "MISSING"  # "RTK", "PPK", "GNSS_STANDALONE", "DESCR_CSV", "MISSING"
    rtk_flag: Optional[int] = None
    is_fix: bool = False
    lat_deg: Optional[float] = None
    lon_deg: Optional[float] = None
    alt_m: Optional[float] = None
    relative_alt_m: Optional[float] = None
    altitude_type: str = "Unknown"  # "RtkAlt", "Ellipsoidal", "Orthometric", "Metric_CSV", "Relative", "Unknown"
    std_lat_m: Optional[float] = None
    std_lon_m: Optional[float] = None
    std_hgt_m: Optional[float] = None
    diff_age_s: Optional[float] = None
    gps_status_raw: Optional[str] = None
    surveying_mode: Optional[int] = None
    source_file: Optional[str] = None
    source_sha256: Optional[str] = None
    match_method: Optional[str] = None
    match_dt_s: Optional[float] = None
    lever_arm_nev_mm: Optional[Tuple[float, float, float]] = None
    # Tracciabilità pre-PPK
    original_rtk_flag: Optional[int] = None
    original_lat_deg: Optional[float] = None
    original_lon_deg: Optional[float] = None
    original_alt_m: Optional[float] = None
    original_std_lat_m: Optional[float] = None
    original_std_lon_m: Optional[float] = None
    original_std_hgt_m: Optional[float] = None
    ppk_shift_enu_m: Optional[Tuple[float, float, float]] = None
    is_validated: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def max_horiz_std_m(self) -> Optional[float]:
        vals = [v for v in (self.std_lat_m, self.std_lon_m) if v is not None]
        return max(vals) if vals else None

    @property
    def max_std_m(self) -> Optional[float]:
        vals = [v for v in (self.std_lat_m, self.std_lon_m, self.std_hgt_m) if v is not None]
        return max(vals) if vals else None

    def summary_label(self) -> str:
        if self.mode == "PPK":
            if self.is_fix and self.max_horiz_std_m is not None:
                return f"PPK FIX (±{self.max_horiz_std_m*100:.1f} cm H / ±{(self.std_hgt_m or 0)*100:.1f} cm V)"
            return f"PPK {rtk_flag_label(self.rtk_flag)}"
        if self.mode == "RTK":
            if self.is_fix and self.max_horiz_std_m is not None:
                return f"RTK FIX (±{self.max_horiz_std_m*100:.1f} cm H / ±{(self.std_hgt_m or 0)*100:.1f} cm V)"
            if self.rtk_flag == RTK_FLAG_FLOAT:
                return f"RTK FLOAT (Flag 34, non fissato)"
            return f"NO RTK FIX ({rtk_flag_label(self.rtk_flag)})"
        if self.mode == "GNSS_STANDALONE":
            return "GPS Standard (Flag RTK assente)"
        if self.mode == "DESCR_CSV":
            return "descr.csv (AlignmentMode 3D - GNSS/RTK non documentato)"
        return "Posizionamento mancante"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "rtk_flag": self.rtk_flag,
            "rtk_flag_label": rtk_flag_label(self.rtk_flag),
            "is_fix": self.is_fix,
            "lat_deg": self.lat_deg,
            "lon_deg": self.lon_deg,
            "alt_m": self.alt_m,
            "relative_alt_m": self.relative_alt_m,
            "altitude_type": self.altitude_type,
            "std_lat_m": self.std_lat_m,
            "std_lon_m": self.std_lon_m,
            "std_hgt_m": self.std_hgt_m,
            "diff_age_s": self.diff_age_s,
            "gps_status_raw": self.gps_status_raw,
            "surveying_mode": self.surveying_mode,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "match_method": self.match_method,
            "match_dt_s": self.match_dt_s,
            "lever_arm_nev_mm": list(self.lever_arm_nev_mm) if self.lever_arm_nev_mm else None,
            "ppk_shift_enu_m": [round(x, 4) for x in self.ppk_shift_enu_m] if self.ppk_shift_enu_m else None,
            "is_validated": self.is_validated,
            "notes": list(self.notes),
        }


def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        f = float(str(val).strip())
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(float(str(val).strip()))
    except (TypeError, ValueError):
        return None


def _dms_to_deg(dms: Any, ref: Optional[str]) -> Optional[float]:
    try:
        d, m, s = [float(x) for x in dms]
        deg = abs(d) + m / 60.0 + s / 3600.0
        if ref and str(ref).upper() in ("S", "W"):
            deg = -deg
        return deg if np.isfinite(deg) else None
    except Exception:
        return None


def evaluate_dji_xmp_gnss(xmp: Dict[str, str], exif_gps: Optional[Dict[str, Any]] = None) -> GNSSQuality:
    """
    Estrae e valida i dati GNSS/RTK dai tag XMP DJI (e da EXIF GPS solo se XMP assente).
    Non sostituisce mai coordinate o quote assenti con zero o con RelativeAltitude.
    """
    notes: List[str] = []
    lat = _safe_float(xmp.get("GpsLatitude"))
    lon = _safe_float(xmp.get("GpsLongitude"))
    abs_alt = _safe_float(xmp.get("AbsoluteAltitude"))
    rel_alt = _safe_float(xmp.get("RelativeAltitude"))
    alt_type = str(xmp.get("AltitudeType") or "").strip() or "Unknown"

    pos_from_exif = False
    if (lat is None or lon is None) and exif_gps:
        lat_exif = _dms_to_deg(exif_gps.get("GPSLatitude"), exif_gps.get("GPSLatitudeRef"))
        lon_exif = _dms_to_deg(exif_gps.get("GPSLongitude"), exif_gps.get("GPSLongitudeRef"))
        if lat_exif is not None and lon_exif is not None:
            lat, lon = lat_exif, lon_exif
            pos_from_exif = True
            notes.append("Coordinate Lat/Lon lette da fallback EXIF GPSInfo.")
    if abs_alt is None and exif_gps:
        alt_exif = _safe_float(exif_gps.get("GPSAltitude"))
        if alt_exif is not None:
            # GPSAltitudeRef == 1 o b'\x01' significa sotto il livello di riferimento
            alt_ref = exif_gps.get("GPSAltitudeRef")
            if alt_ref in (1, b"\x01"):
                alt_exif = -alt_exif
            abs_alt = alt_exif
            if alt_type == "Unknown":
                alt_type = "EXIF_GPSAltitude"
            notes.append("Quota letta da fallback EXIF GPSAltitude.")

    if lat is None or lon is None:
        notes.append("Latitudine o longitudine mancanti nei metadati.")
    if abs_alt is None:
        if rel_alt is not None:
            notes.append(
                "AbsoluteAltitude mancante (presente solo RelativeAltitude dal decollo, non mescolabile)."
            )
        else:
            notes.append("Quota assoluta (AbsoluteAltitude) mancante nei metadati.")

    rtk_flag = _safe_int(xmp.get("RtkFlag"))
    gps_status_raw = xmp.get("GpsStatus")
    std_lat = _safe_float(xmp.get("RtkStdLat"))
    std_lon = _safe_float(xmp.get("RtkStdLon"))
    std_hgt = _safe_float(xmp.get("RtkStdHgt"))
    diff_age = _safe_float(xmp.get("RtkDiffAge"))
    surv_mode = _safe_int(xmp.get("SurveyingMode"))

    if lat is None or lon is None or abs_alt is None:
        mode = "MISSING"
    elif rtk_flag is not None:
        mode = "RTK"
    elif gps_status_raw and "RTK" in str(gps_status_raw).upper():
        mode = "RTK"
        notes.append("GpsStatus indica RTK ma RtkFlag numerico è assente.")
    else:
        mode = "GNSS_STANDALONE" if not pos_from_exif else "GNSS_STANDALONE"
        notes.append("RtkFlag assente: posizionamento trattato come GPS non-RTK.")

    is_fix = (rtk_flag == RTK_FLAG_FIX)

    if rtk_flag is not None and not is_fix:
        notes.append(f"Stato RTK non FIX: RtkFlag={rtk_flag} ({rtk_flag_label(rtk_flag)}).")

    if std_lat is None or std_lon is None or std_hgt is None:
        notes.append("Deviazioni standard RTK (RtkStdLat/Lon/Hgt) incomplete o assenti.")

    if surv_mode == 0:
        notes.append("SurveyingMode=0 (fotogramma marcato dal drone come non raccomandato per rilievi).")

    has_coords = (lat is not None and lon is not None and abs_alt is not None)
    has_stds = (
        std_lat is not None and std_lat > 0
        and std_lon is not None and std_lon > 0
        and std_hgt is not None and std_hgt > 0
    )
    is_validated = bool(has_coords and mode == "RTK" and is_fix and has_stds)

    return GNSSQuality(
        mode=mode,
        rtk_flag=rtk_flag,
        is_fix=is_fix,
        lat_deg=lat,
        lon_deg=lon,
        alt_m=abs_alt,
        relative_alt_m=rel_alt,
        altitude_type=alt_type,
        std_lat_m=std_lat,
        std_lon_m=std_lon,
        std_hgt_m=std_hgt,
        diff_age_s=diff_age,
        gps_status_raw=gps_status_raw,
        surveying_mode=surv_mode,
        original_rtk_flag=rtk_flag,
        original_lat_deg=lat,
        original_lon_deg=lon,
        original_alt_m=abs_alt,
        original_std_lat_m=std_lat,
        original_std_lon_m=std_lon,
        original_std_hgt_m=std_hgt,
        is_validated=is_validated,
        notes=notes,
    )
