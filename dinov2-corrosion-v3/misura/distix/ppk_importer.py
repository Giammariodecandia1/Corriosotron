"""
DISTIX - Traceable PPK Importer
Importazione deterministica e tracciabile di traiettorie/eventi PPK (DJI .MRK, RTKLIB .pos, CSV).
Garantisce:
- Verifica integrità file tramite hash SHA-256
- Associazione univoca e verificata alle fotografie (filename esatto, timestamp GPS/UTC <= 50 ms)
- Nessuna promozione arbitraria a FIX: usa i flag di qualità e le std effettive
- Controllo coerenza datum altimetrico (ellissoidica vs ortometrica/relativa)
- Conservazione completa dei dati GNSS originali e del vettore spostamento ENU pre/post PPK
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
import csv
import hashlib
import os
import re
import numpy as np

from .rtk_validator import (
    GNSSQuality,
    RTK_FLAG_NONE,
    RTK_FLAG_SINGLE,
    RTK_FLAG_FLOAT,
    RTK_FLAG_FIX,
    rtk_flag_label,
)
from .coordinate_system import CoordinateManager


GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
GPS_UTC_LEAP_SECONDS = 18.0  # Differenza GPS-UTC valida dal 2017 ad oggi


def compute_file_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def gps_week_tow_to_gps_unix(week: int, tow_s: float) -> float:
    """Converte (GPS week, GPS TOW) in secondi dall'epoca Unix su scala tempo GPS continua."""
    dt = GPS_EPOCH + timedelta(weeks=int(week), seconds=float(tow_s))
    return dt.timestamp()


def gps_week_tow_to_utc_unix(week: int, tow_s: float, leap_seconds: float = GPS_UTC_LEAP_SECONDS) -> float:
    """Converte (GPS week, GPS TOW) in timestamp Unix UTC sottraendo i leap seconds."""
    return gps_week_tow_to_gps_unix(week, tow_s) - float(leap_seconds)


def parse_iso_or_exif_datetime_to_unix(value: Optional[str]) -> Optional[float]:
    """Analizza timestamp ISO-8601 o EXIF (YYYY:MM:DD HH:MM:SS) in timestamp Unix (secondi)."""
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None

    # Formato ISO 8601 (es. DJI UTCAtExposure: 2026-11-09T12:43:44.256549 o con Z / +01:00)
    try:
        iso_norm = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_norm)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        pass

    # Formato RTKLIB / EXIF con eventuale frazione
    for fmt in (
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y:%m:%d %H:%M:%S.%f",
        "%Y:%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def extract_dji_seq_from_filename(filename: str) -> Optional[int]:
    """Estrae il numero sequenza DJI (es. DJI_20261109134324_0001_V.JPG -> 1)."""
    base = os.path.basename(filename)
    m = re.search(r"_(\d{4})_[A-Za-z0-9]+\.(?:jpg|jpeg|dng|tif|tiff)$", base, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _parse_quality_token(token: Any, source_format: str) -> Optional[int]:
    if token is None:
        return None
    s = str(token).strip().upper()
    if not s:
        return None

    if s in ("FIX", "FIXED", "RTK_FIX", "Q1"):
        return RTK_FLAG_FIX
    if s in ("FLOAT", "RTK_FLOAT", "Q2"):
        return RTK_FLAG_FLOAT
    if s in ("SINGLE", "SPP", "STANDALONE", "DGPS", "Q4", "Q5"):
        return RTK_FLAG_SINGLE
    if s in ("NONE", "NO_FIX", "INVALID", "Q0"):
        return RTK_FLAG_NONE

    try:
        val = int(float(s))
    except ValueError:
        return None

    if source_format == "RTKLIB_POS":
        # RTKLIB standard Q: 1=fix, 2=float, 3=sbas, 4=dgps, 5=single, 6=ppp
        if val == 1:
            return RTK_FLAG_FIX
        if val == 2:
            return RTK_FLAG_FLOAT
        if val in (3, 4, 5, 6):
            return RTK_FLAG_SINGLE
        return RTK_FLAG_NONE

    # DJI MRK / CSV numerico diretto o Q 1..5
    if val in (0, 16, 34, 50):
        return val
    if val == 1:
        return RTK_FLAG_FIX
    if val == 2:
        return RTK_FLAG_FLOAT
    if val in (3, 4, 5):
        return RTK_FLAG_SINGLE
    return val


@dataclass
class PPKRecord:
    record_id: str
    lat_deg: float
    lon_deg: float
    alt_m: float
    rtk_flag: int
    source_format: str
    altitude_type: str = "Unknown"
    filename: Optional[str] = None
    seq_id: Optional[int] = None
    gps_week: Optional[int] = None
    gps_tow_s: Optional[float] = None
    gpst_unix_s: Optional[float] = None
    utc_unix_s: Optional[float] = None
    std_lat_m: Optional[float] = None
    std_lon_m: Optional[float] = None
    std_hgt_m: Optional[float] = None
    diff_age_s: Optional[float] = None
    lever_arm_nev_mm: Optional[Tuple[float, float, float]] = None
    num_satellites: Optional[int] = None
    raw_quality: Optional[str] = None

    @property
    def is_fix(self) -> bool:
        return self.rtk_flag == RTK_FLAG_FIX

    @property
    def has_valid_stds(self) -> bool:
        return (
            self.std_lat_m is not None and self.std_lat_m > 0
            and self.std_lon_m is not None and self.std_lon_m > 0
            and self.std_hgt_m is not None and self.std_hgt_m > 0
        )


@dataclass
class PPKImportReport:
    source_path: str
    source_sha256: str
    source_format: str
    records_parsed: int
    matched_count: int
    fix_count: int
    validated_count: int
    unmatched_cameras: List[str] = field(default_factory=list)
    unmatched_records: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    matched_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": os.path.basename(self.source_path),
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_format": self.source_format,
            "records_parsed": self.records_parsed,
            "matched_count": self.matched_count,
            "fix_count": self.fix_count,
            "validated_count": self.validated_count,
            "unmatched_cameras": list(self.unmatched_cameras),
            "unmatched_records": list(self.unmatched_records),
            "warnings": list(self.warnings),
            "matched_details": list(self.matched_details),
        }


def parse_dji_mrk(mrk_path: str) -> List[PPKRecord]:
    """
    Analizza un file evento DJI *_D.MRK.
    Formato tipico:
    1  132224.256550  [2444]  151,N  -44,E  85,V  43.11360069,Lat  12.37258394,Lon  475.053,Ellh  0.103786, 0.094933, 0.190110  16,Q
    """
    records: List[PPKRecord] = []
    with open(mrk_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue

            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if len(parts) < 9:
                # Fallback spazi multipli se il file è stato riesportato senza tab
                parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
            if len(parts) < 9:
                continue

            try:
                seq_id = int(parts[0])
                tow_s = float(parts[1])
                week_m = re.search(r"\[?(\d+)\]?", parts[2])
                gps_week = int(week_m.group(1)) if week_m else None

                # Lever arm N, E, V in mm (opzionale rispetto a parsing tollerante)
                lever_n = float(parts[3].split(",")[0])
                lever_e = float(parts[4].split(",")[0])
                lever_v = float(parts[5].split(",")[0])

                lat_deg = float(parts[6].split(",")[0])
                lon_deg = float(parts[7].split(",")[0])

                alt_tokens = parts[8].split(",")
                alt_m = float(alt_tokens[0])
                alt_tag = alt_tokens[1].strip() if len(alt_tokens) > 1 else "Ellh"
                alt_type = "Ellipsoidal" if "ell" in alt_tag.lower() else (
                    "Orthometric" if any(k in alt_tag.lower() for k in ("ortho", "msl", "egm")) else alt_tag
                )

                std_lat = std_lon = std_hgt = None
                if len(parts) >= 10:
                    std_vals = [x.strip() for x in parts[9].split(",") if x.strip()]
                    if len(std_vals) >= 3:
                        std_lat = float(std_vals[0])
                        std_lon = float(std_vals[1])
                        std_hgt = float(std_vals[2])

                raw_q = parts[10].split(",")[0].strip() if len(parts) >= 11 else "0"
                rtk_flag = _parse_quality_token(raw_q, "DJI_MRK")
                if rtk_flag is None:
                    rtk_flag = RTK_FLAG_NONE

                gpst_unix = gps_week_tow_to_gps_unix(gps_week, tow_s) if gps_week is not None else None
                utc_unix = gps_week_tow_to_utc_unix(gps_week, tow_s) if gps_week is not None else None

                records.append(
                    PPKRecord(
                        record_id=f"MRK#{seq_id}",
                        seq_id=seq_id,
                        gps_week=gps_week,
                        gps_tow_s=tow_s,
                        gpst_unix_s=gpst_unix,
                        utc_unix_s=utc_unix,
                        lat_deg=lat_deg,
                        lon_deg=lon_deg,
                        alt_m=alt_m,
                        altitude_type=alt_type,
                        rtk_flag=rtk_flag,
                        std_lat_m=std_lat,
                        std_lon_m=std_lon,
                        std_hgt_m=std_hgt,
                        lever_arm_nev_mm=(lever_n, lever_e, lever_v),
                        source_format="DJI_MRK",
                        raw_quality=raw_q,
                    )
                )
            except (ValueError, IndexError):
                continue

    return records


def parse_rtklib_pos(pos_path: str) -> List[PPKRecord]:
    """
    Analizza un file RTKLIB (.pos o _events.pos).
    Supporta sia intestazione temporale YYYY/MM/DD HH:MM:SS.SSS (GPST o UTC)
    sia formato GPS week + TOW.
    """
    time_scale = "GPST"
    alt_type = "Ellipsoidal"
    records: List[PPKRecord] = []

    with open(pos_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("%"):
                low = line.lower()
                if "utc" in low:
                    time_scale = "UTC"
                elif "gpst" in low:
                    time_scale = "GPST"
                if "msl" in low or "orthometric" in low or "geoid" in low:
                    alt_type = "Orthometric"
                elif "ellipsoidal" in low or "height" in low:
                    alt_type = "Ellipsoidal"
                continue

            cols = line.split()
            if len(cols) < 6:
                continue

            gpst_unix = None
            utc_unix = None
            gps_week = None
            gps_tow = None
            idx_lat = 2

            # Rileva formato data/ora vs GPS week/TOW
            if "/" in cols[0] or "-" in cols[0]:
                ts_str = f"{cols[0]} {cols[1]}"
                ts_val = parse_iso_or_exif_datetime_to_unix(ts_str)
                if ts_val is None:
                    continue
                if time_scale == "UTC":
                    utc_unix = ts_val
                    gpst_unix = ts_val + GPS_UTC_LEAP_SECONDS
                else:
                    gpst_unix = ts_val
                    utc_unix = ts_val - GPS_UTC_LEAP_SECONDS
                idx_lat = 2
            else:
                try:
                    gps_week = int(cols[0])
                    gps_tow = float(cols[1])
                except ValueError:
                    continue
                gpst_unix = gps_week_tow_to_gps_unix(gps_week, gps_tow)
                utc_unix = gps_week_tow_to_utc_unix(gps_week, gps_tow)
                idx_lat = 2

            try:
                lat_deg = float(cols[idx_lat])
                lon_deg = float(cols[idx_lat + 1])
                alt_m = float(cols[idx_lat + 2])
                raw_q = cols[idx_lat + 3]
                rtk_flag = _parse_quality_token(raw_q, "RTKLIB_POS")
                if rtk_flag is None:
                    rtk_flag = RTK_FLAG_NONE
                ns = int(cols[idx_lat + 4]) if len(cols) > idx_lat + 4 else None
                std_lat = float(cols[idx_lat + 5]) if len(cols) > idx_lat + 5 else None  # sdn
                std_lon = float(cols[idx_lat + 6]) if len(cols) > idx_lat + 6 else None  # sde
                std_hgt = float(cols[idx_lat + 7]) if len(cols) > idx_lat + 7 else None  # sdu
                age_s = float(cols[idx_lat + 11]) if len(cols) > idx_lat + 11 else None
            except (ValueError, IndexError):
                continue

            records.append(
                PPKRecord(
                    record_id=f"POS#{line_no}",
                    gps_week=gps_week,
                    gps_tow_s=gps_tow,
                    gpst_unix_s=gpst_unix,
                    utc_unix_s=utc_unix,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    alt_m=alt_m,
                    altitude_type=alt_type,
                    rtk_flag=rtk_flag,
                    std_lat_m=std_lat,
                    std_lon_m=std_lon,
                    std_hgt_m=std_hgt,
                    diff_age_s=age_s,
                    num_satellites=ns,
                    source_format="RTKLIB_POS",
                    raw_quality=raw_q,
                )
            )

    return records


def parse_ppk_csv(csv_path: str) -> List[PPKRecord]:
    """
    Analizza un file CSV/TSV PPK strutturato con colonne flessibili:
    Filename/Photo, Timestamp/UTC/GPS_TOW, GPSWeek, Lat, Lon, Alt/Ellh, AltitudeType,
    Q/RtkFlag/Status, StdLat/Sdn, StdLon/Sde, StdHgt/StdAlt/Sdu, DiffAge.
    """
    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            return []

        norm_map = {re.sub(r"[^a-z0-9]", "", col.lower()): col for col in reader.fieldnames}

        def get_col(row: Dict[str, str], candidates: Tuple[str, ...]) -> Optional[str]:
            for cand in candidates:
                orig = norm_map.get(cand)
                if orig and row.get(orig) is not None and str(row[orig]).strip() != "":
                    return str(row[orig]).strip()
            return None

        records: List[PPKRecord] = []
        for idx, row in enumerate(reader, start=1):
            lat_s = get_col(row, ("lat", "latitude", "gpslatitude", "latdeg"))
            lon_s = get_col(row, ("lon", "long", "longitude", "gpslongitude", "londeg"))
            alt_s = get_col(row, ("alt", "altitude", "height", "ellh", "absolutealtitude", "altm"))
            if lat_s is None or lon_s is None or alt_s is None:
                continue

            try:
                lat_deg = float(lat_s)
                lon_deg = float(lon_s)
                alt_m = float(alt_s)
            except ValueError:
                continue

            fname = get_col(row, ("filename", "photo", "image", "file", "name"))
            seq_s = get_col(row, ("seq", "seqid", "sequence", "event", "index"))
            seq_id = int(seq_s) if seq_s and seq_s.isdigit() else None

            q_raw = get_col(row, ("rtkflag", "q", "quality", "fix", "status", "gpsstatus"))
            rtk_flag = _parse_quality_token(q_raw, "PPK_CSV")
            if rtk_flag is None:
                rtk_flag = RTK_FLAG_NONE

            alt_type = get_col(row, ("altitudetype", "alttype", "datum")) or "Unknown"

            def opt_float(cands: Tuple[str, ...]) -> Optional[float]:
                val = get_col(row, cands)
                if val is None:
                    return None
                try:
                    fv = float(val)
                    return fv if np.isfinite(fv) else None
                except ValueError:
                    return None

            std_lat = opt_float(("stdlat", "stdlatm", "sdn", "sigman", "sigmalat"))
            std_lon = opt_float(("stdlon", "stdlonm", "sde", "sigmae", "sigmalon"))
            std_hgt = opt_float(("stdhgt", "stdhgtm", "stdalt", "stdz", "sdu", "sigmau", "sigmaz"))
            diff_age = opt_float(("diffage", "age", "rtkdiffage"))

            gps_week_s = get_col(row, ("gpsweek", "week"))
            gps_tow = opt_float(("gpstow", "tow", "sow"))
            gps_week = int(gps_week_s) if gps_week_s and gps_week_s.isdigit() else None

            ts_raw = get_col(row, ("timestamp", "utctime", "utcatexposure", "datetime", "gpstime"))
            gpst_unix = None
            utc_unix = None
            if gps_week is not None and gps_tow is not None:
                gpst_unix = gps_week_tow_to_gps_unix(gps_week, gps_tow)
                utc_unix = gps_week_tow_to_utc_unix(gps_week, gps_tow)
            elif ts_raw is not None:
                parsed_t = parse_iso_or_exif_datetime_to_unix(ts_raw)
                if parsed_t is not None:
                    utc_unix = parsed_t
                    gpst_unix = parsed_t + GPS_UTC_LEAP_SECONDS
                else:
                    try:
                        num_t = float(ts_raw)
                        utc_unix = num_t
                        gpst_unix = num_t
                    except ValueError:
                        pass

            records.append(
                PPKRecord(
                    record_id=fname or f"CSV#{idx}",
                    filename=os.path.basename(fname) if fname else None,
                    seq_id=seq_id,
                    gps_week=gps_week,
                    gps_tow_s=gps_tow,
                    gpst_unix_s=gpst_unix,
                    utc_unix_s=utc_unix,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    alt_m=alt_m,
                    altitude_type=alt_type,
                    rtk_flag=rtk_flag,
                    std_lat_m=std_lat,
                    std_lon_m=std_lon,
                    std_hgt_m=std_hgt,
                    diff_age_s=diff_age,
                    source_format="PPK_CSV",
                    raw_quality=q_raw,
                )
            )

    return records


def parse_ppk_file(file_path: str) -> Tuple[str, List[PPKRecord]]:
    """Rileva automaticamente il formato PPK (.MRK / .pos / .csv) e ne estrae i record."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File PPK non trovato: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".mrk":
        return "DJI_MRK", parse_dji_mrk(file_path)
    if ext == ".pos":
        return "RTKLIB_POS", parse_rtklib_pos(file_path)
    if ext in (".csv", ".tsv", ".txt"):
        # Controlla se è un MRK o POS salvato come .txt
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(1024)
        if re.search(r"\[\d+\]", head) and re.search(r"\d\s*,\s*Lat\b", head) and re.search(r"\d\s*,\s*Lon\b", head):
            return "DJI_MRK", parse_dji_mrk(file_path)
        if head.lstrip().startswith("%"):
            return "RTKLIB_POS", parse_rtklib_pos(file_path)
        return "PPK_CSV", parse_ppk_csv(file_path)

    raise ValueError(f"Formato file PPK non supportato: {ext}")


def _altitudes_compatible(cam_alt_type: str, ppk_alt_type: str) -> Tuple[bool, Optional[str]]:
    """
    Verifica la compatibilità del riferimento altimetrico tra metadati camera e soluzione PPK.
    Su DJI Enterprise in RTK/PPK, 'RtkAlt' e 'Ellipsoidal' (Ellh) condividono il datum ellissoidico WGS84.
    """
    c = (cam_alt_type or "Unknown").strip().lower()
    p = (ppk_alt_type or "Unknown").strip().lower()

    if c == "relative" or p == "relative":
        return False, "Riferimento altimetrico relativo incompatibile con quota assoluta GNSS."

    c_is_ortho = any(k in c for k in ("ortho", "msl", "egm"))
    p_is_ortho = any(k in p for k in ("ortho", "msl", "egm"))
    c_is_ell = any(k in c for k in ("ell", "rtkalt"))
    p_is_ell = any(k in p for k in ("ell", "rtkalt"))

    if (c_is_ortho and p_is_ell) or (c_is_ell and p_is_ortho):
        return False, (
            f"Incoerenza datum altimetrico tra foto ({cam_alt_type}) e PPK ({ppk_alt_type}): "
            "rischio offset geoidico non compensato."
        )
    return True, None


def apply_ppk_to_cameras(
    cameras: Dict[str, Any],
    ppk_path: str,
    coord_mgr: Optional[CoordinateManager] = None,
    max_time_diff_s: float = 0.05,
    apply_fix_mode: bool = True,
) -> Tuple[PPKImportReport, CoordinateManager]:
    """
    Associa in modo tracciabile i record di un file PPK (.MRK, .pos, .csv) alle camere caricate.
    Criteri di associazione rigorosi (in ordine di priorità):
    1. Corrispondenza esatta del nome file (PPK CSV)
    2. Corrispondenza temporale deterministica (dt <= max_time_diff_s) su UTCAtExposure/CreateDate,
       con eventuale verifica coerenza seq_id per file .MRK.
    Non accetta associazioni ambigue o fuori tolleranza temporale.
    """
    source_sha = compute_file_sha256(ppk_path)
    src_format, records = parse_ppk_file(ppk_path)

    if not records:
        raise ValueError(f"Nessun record PPK valido estratto da {ppk_path}")

    # Indici di supporto
    used_record_indices = set()
    matched_assignments: Dict[str, Tuple[PPKRecord, str, Optional[float]]] = {}
    warnings: List[str] = []

    cam_names = list(cameras.keys())

    # Step 1: Match esatto per nome file
    by_filename: Dict[str, List[int]] = {}
    for r_idx, rec in enumerate(records):
        if rec.filename:
            by_filename.setdefault(rec.filename.lower(), []).append(r_idx)

    for cname in cam_names:
        key = os.path.basename(cname).lower()
        cand_indices = by_filename.get(key, [])
        if len(cand_indices) == 1:
            r_idx = cand_indices[0]
            used_record_indices.add(r_idx)
            matched_assignments[cname] = (records[r_idx], "FILENAME_EXACT", 0.0)
        elif len(cand_indices) > 1:
            warnings.append(f"Record multipli nel file PPK per {cname}: associazione ambigua saltata.")

    # Step 2: Match temporale rigoroso per le camere rimanenti
    remaining_cams = [c for c in cam_names if c not in matched_assignments]
    # Estrae tutti i possibili riferimenti temporali della camera:
    # - gpst_unix_s (es. UTCAtExposure DJI interpretato come GPS continuous time, verificato 1 us vs MRK)
    # - utc_unix_s (es. CreateDate o vero UTC)
    cam_candidates: List[Tuple[str, float, int, str, float]] = []
    for cname in remaining_cams:
        cam = cameras[cname]
        meta = getattr(cam, "metadata", {}) or {}
        cam_gpst = meta.get("exposure_gpst_unix_s")
        cam_utc = meta.get("exposure_utc_unix_s")
        cam_seq = meta.get("dji_seq_id") or extract_dji_seq_from_filename(cname)

        for r_idx, rec in enumerate(records):
            if r_idx in used_record_indices:
                continue

            best_dt = None
            best_mode = None

            if cam_gpst is not None and rec.gpst_unix_s is not None:
                dt_gpst = abs(cam_gpst - rec.gpst_unix_s)
                if dt_gpst <= max_time_diff_s:
                    best_dt = dt_gpst
                    best_mode = "TIMESTAMP_GPST"

            if cam_utc is not None and rec.utc_unix_s is not None:
                dt_utc = abs(cam_utc - rec.utc_unix_s)
                if dt_utc <= max_time_diff_s and (best_dt is None or dt_utc < best_dt):
                    best_dt = dt_utc
                    best_mode = "TIMESTAMP_UTC"

            if best_dt is not None and best_mode is not None:
                # Se entrambi hanno seq_id e non coincidono, avvisa ma controlla
                if cam_seq is not None and rec.seq_id is not None and cam_seq != rec.seq_id:
                    warnings.append(
                        f"Timestamp compatibile ({best_dt*1000:.2f} ms) tra {cname} (seq {cam_seq}) "
                        f"e record {rec.record_id} (seq {rec.seq_id}): seq discordante."
                    )
                method_str = f"{best_mode}_SEQ_VERIFIED" if (cam_seq is not None and rec.seq_id == cam_seq) else best_mode
                cam_candidates.append((best_dt, cname, r_idx, method_str, best_dt))

    # Assegna in ordine crescente di delta t garantendo univocità 1-a-1
    cam_candidates.sort(key=lambda item: item[0])
    # Verifica ambiguità: se per la stessa camera due record hanno dt quasi uguale entro tolerance
    for dt_val, cname, r_idx, method_str, exact_dt in cam_candidates:
        if cname in matched_assignments or r_idx in used_record_indices:
            continue
        # Controlla se esiste un altro record non usato entro 5 ms dallo stesso scatto
        ambiguous = [
            o for o in cam_candidates
            if o[1] == cname and o[2] != r_idx and o[2] not in used_record_indices and abs(o[0] - dt_val) < 0.005
        ]
        if ambiguous:
            warnings.append(f"Match temporale ambiguo per {cname}: più record PPK entro 5 ms.")
            continue
        used_record_indices.add(r_idx)
        matched_assignments[cname] = (records[r_idx], method_str, exact_dt)

    # Inizializza CoordinateManager se mancante (ENU locale tangente al Nord Vero)
    if coord_mgr is None:
        first_rec = None
        for cname in cam_names:
            if cname in matched_assignments:
                first_rec = matched_assignments[cname][0]
                break
        if first_rec is None:
            first_rec = records[0]
        utm_zone = int((first_rec.lon_deg + 180.0) / 6.0) + 1
        is_north = (first_rec.lat_deg >= 0.0)
        coord_mgr = CoordinateManager(utm_zone=utm_zone, is_northern=is_north)
        x0, y0, z0 = coord_mgr.wgs84_to_utm(first_rec.lon_deg, first_rec.lat_deg, first_rec.alt_m)
        coord_mgr.set_origin(np.array([x0, y0, z0], dtype=np.float64))
        coord_mgr.set_local_enu_origin(first_rec.lon_deg, first_rec.lat_deg, first_rec.alt_m)

    # Applicazione delle soluzioni PPK alle camere matchate
    matched_details: List[Dict[str, Any]] = []
    fix_count = 0
    validated_count = 0

    for cname in cam_names:
        if cname not in matched_assignments:
            continue
        cam = cameras[cname]
        rec, method_str, dt_val = matched_assignments[cname]
        old_gnss: Optional[GNSSQuality] = getattr(cam, "gnss_quality", None)

        # Verifica coerenza datum altimetrico
        old_alt_type = old_gnss.altitude_type if old_gnss else "Unknown"
        datum_ok, datum_warn = _altitudes_compatible(old_alt_type, rec.altitude_type)
        if not datum_ok and datum_warn:
            warnings.append(f"{cname}: {datum_warn}")

        # Salva valori pre-PPK per audit
        orig_lat = old_gnss.original_lat_deg if old_gnss and old_gnss.original_lat_deg is not None else (
            old_gnss.lat_deg if old_gnss else None
        )
        orig_lon = old_gnss.original_lon_deg if old_gnss and old_gnss.original_lon_deg is not None else (
            old_gnss.lon_deg if old_gnss else None
        )
        orig_alt = old_gnss.original_alt_m if old_gnss and old_gnss.original_alt_m is not None else (
            old_gnss.alt_m if old_gnss else None
        )
        orig_flag = old_gnss.original_rtk_flag if old_gnss else getattr(cam, "rtk_flag", None)

        new_local = coord_mgr.wgs84_to_local(rec.lon_deg, rec.lat_deg, rec.alt_m)

        shift_enu = None
        if orig_lat is not None and orig_lon is not None and orig_alt is not None:
            orig_local = coord_mgr.wgs84_to_local(orig_lon, orig_lat, orig_alt)
            shift_enu = tuple((new_local - orig_local).tolist())

        # Controllo anomalie di quota o salto esagerato (> 25 m indica spesso errore geoidico o match errato)
        if shift_enu is not None and abs(shift_enu[2]) > 25.0:
            datum_ok = False
            warnings.append(
                f"{cname}: variazione quota post-PPK anomala (dZ={shift_enu[2]:+.2f} m): "
                "possibile disallineamento ellissoide/geoide."
            )

        has_stds = rec.has_valid_stds
        if not has_stds:
            warnings.append(f"{cname}: record PPK {rec.record_id} privo di deviazioni standard valide.")

        is_validated = bool(datum_ok and rec.is_fix and has_stds)
        notes_list: List[str] = [
            f"PPK importato da {os.path.basename(ppk_path)} ({src_format}, match={method_str}, dt={dt_val:.6f}s)."
        ]
        if not rec.is_fix:
            notes_list.append(f"Soluzione PPK non FIX: Flag={rec.rtk_flag} ({rtk_flag_label(rec.rtk_flag)}).")
        if not has_stds:
            notes_list.append("Deviazioni standard PPK assenti o nulle.")
        if not datum_ok and datum_warn:
            notes_list.append(datum_warn)

        new_gnss = GNSSQuality(
            mode="PPK",
            rtk_flag=rec.rtk_flag,
            is_fix=rec.is_fix,
            lat_deg=rec.lat_deg,
            lon_deg=rec.lon_deg,
            alt_m=rec.alt_m,
            relative_alt_m=old_gnss.relative_alt_m if old_gnss else None,
            altitude_type=rec.altitude_type if rec.altitude_type != "Unknown" else old_alt_type,
            std_lat_m=rec.std_lat_m,
            std_lon_m=rec.std_lon_m,
            std_hgt_m=rec.std_hgt_m,
            diff_age_s=rec.diff_age_s,
            gps_status_raw=f"PPK:{src_format}",
            surveying_mode=old_gnss.surveying_mode if old_gnss else None,
            source_file=os.path.basename(ppk_path),
            source_sha256=source_sha,
            match_method=method_str,
            match_dt_s=dt_val,
            lever_arm_nev_mm=rec.lever_arm_nev_mm,
            original_rtk_flag=orig_flag,
            original_lat_deg=orig_lat,
            original_lon_deg=orig_lon,
            original_alt_m=orig_alt,
            original_std_lat_m=old_gnss.original_std_lat_m if old_gnss else None,
            original_std_lon_m=old_gnss.original_std_lon_m if old_gnss else None,
            original_std_hgt_m=old_gnss.original_std_hgt_m if old_gnss else None,
            ppk_shift_enu_m=shift_enu,
            is_validated=is_validated,
            notes=notes_list,
        )

        # Aggiorna la camera senza inventare valori se apply_fix_mode è attivo
        if apply_fix_mode:
            cam.pos = new_local
            if hasattr(cam, "nominal_pos"):
                cam.nominal_pos = new_local.copy()
            cam.rtk_flag = rec.rtk_flag
            cam.rtk_fix = rec.is_fix
            if has_stds:
                cam.rtk_std = float(max(rec.std_lat_m, rec.std_lon_m))
            else:
                cam.rtk_std = None
            cam.gnss_quality = new_gnss

        if rec.is_fix:
            fix_count += 1
        if is_validated:
            validated_count += 1

        matched_details.append(
            {
                "camera_name": cname,
                "record_id": rec.record_id,
                "match_method": method_str,
                "match_dt_s": round(dt_val, 6) if dt_val is not None else None,
                "pre_rtk_flag": orig_flag,
                "post_rtk_flag": rec.rtk_flag,
                "post_fix": rec.is_fix,
                "std_lat_m": rec.std_lat_m,
                "std_lon_m": rec.std_lon_m,
                "std_hgt_m": rec.std_hgt_m,
                "shift_enu_m": [round(v, 4) for v in shift_enu] if shift_enu else None,
                "is_validated": is_validated,
            }
        )

    unmatched_cams = [c for c in cam_names if c not in matched_assignments]
    unmatched_recs = [records[i].record_id for i in range(len(records)) if i not in used_record_indices]

    report = PPKImportReport(
        source_path=os.path.abspath(ppk_path),
        source_sha256=source_sha,
        source_format=src_format,
        records_parsed=len(records),
        matched_count=len(matched_details),
        fix_count=fix_count,
        validated_count=validated_count,
        unmatched_cameras=unmatched_cams,
        unmatched_records=unmatched_recs,
        warnings=warnings,
        matched_details=matched_details,
    )
    return report, coord_mgr
