"""
DISTIX - Metadata Reader & DJI EXIF/XMP Parser
Estrazione rigorosa e validazione dei metadati di volo, GNSS/RTK, calibrazione ottica
e orientamento drone/gimbal sia da immagini DJI originali (M3E / M4E / RTK) sia da descr.csv.
Principio operativo: nessun dato mancante viene sostituito con zero o valori presunti.
"""

import csv
import glob
import os
import re
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from PIL import Image, ExifTags

from .camera_model import CameraModel
from .coordinate_system import CoordinateManager
from .calibration_manager import (
    OpticalCalibration,
    resolve_dji_calibration,
    load_documented_profiles,
)
from .rtk_validator import (
    GNSSQuality,
    evaluate_dji_xmp_gnss,
    RTK_FLAG_FIX,
)
from .ppk_importer import (
    parse_iso_or_exif_datetime_to_unix,
    extract_dji_seq_from_filename,
)


def parse_dji_xmp(image_path: str) -> Dict[str, str]:
    """
    Estrae i blocchi XMP incorporati nel file JPEG (namespace drone-dji, tiff, xmp, exif).
    """
    xmp_data: Dict[str, str] = {}
    try:
        with open(image_path, "rb") as f:
            content = f.read()
        xmp_start = content.find(b"<x:xmpmeta")
        xmp_end = content.find(b"</x:xmpmeta>")
        if xmp_start != -1 and xmp_end != -1:
            xmp_str = content[xmp_start : xmp_end + 12].decode("utf-8", errors="ignore")

            # 1. Attributi XML su qualsiasi prefisso noto DJI/XMP/EXIF/TIFF
            matches_attr = re.findall(
                r'(?:drone-dji|tiff|xmp|exif|Camera):([A-Za-z0-9_-]+)=["\']([^"\']*)["\']',
                xmp_str,
            )
            for key, val in matches_attr:
                val_clean = val.strip()
                if val_clean != "" and key not in xmp_data:
                    xmp_data[key] = val_clean

            # 2. Tag XML annidati (precedenza ai tag drone-dji diretti)
            matches_tag = re.findall(
                r'<(?:drone-dji|tiff|xmp|exif|Camera):([A-Za-z0-9_-]+)>([^<]+)</(?:drone-dji|tiff|xmp|exif|Camera):\1>',
                xmp_str,
            )
            for key, val in matches_tag:
                val_clean = val.strip()
                if val_clean != "":
                    xmp_data[key] = val_clean
    except Exception:
        pass
    return xmp_data


def _rational_to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        if isinstance(val, tuple) and len(val) == 2:
            num, den = float(val[0]), float(val[1])
            return num / den if den != 0 else None
        f = float(val)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def parse_image_exif(image_path: str) -> Tuple[int, int, Dict[str, Any], Dict[str, Any]]:
    """
    Legge dimensioni effettive dell'immagine, EXIF standard e GPSInfo tramite PIL.
    Restituisce (width, height, exif_dict, exif_gps_dict).
    """
    exif_dict: Dict[str, Any] = {}
    exif_gps: Dict[str, Any] = {}

    with Image.open(image_path) as im:
        width, height = im.size
        raw_exif = im.getexif()
        if raw_exif:
            for tag_id, val in raw_exif.items():
                name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if name in ("FocalLength", "DigitalZoomRatio", "ExposureTime", "FNumber"):
                    exif_dict[name] = _rational_to_float(val)
                elif name == "FocalLengthIn35mmFilm":
                    exif_dict[name] = _rational_to_float(val)
                elif name == "Orientation":
                    exif_dict[name] = int(val) if val is not None else 1
                elif name in ("Make", "Model", "DateTime", "DateTimeOriginal", "LensModel"):
                    exif_dict[name] = str(val).strip().rstrip("\x00")
                elif name == "GPSInfo" and isinstance(val, dict):
                    for g_id, g_val in val.items():
                        g_name = ExifTags.GPSTAGS.get(g_id, str(g_id))
                        exif_gps[g_name] = g_val

            # Sub-IFD EXIF (0x8769) e GPS (0x8825) in Pillow >= 10
            try:
                exif_sub = raw_exif.get_ifd(0x8769)
                for tag_id, val in exif_sub.items():
                    name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if name in ("FocalLength", "DigitalZoomRatio", "ExposureTime", "FNumber"):
                        exif_dict[name] = _rational_to_float(val)
                    elif name == "FocalLengthIn35mmFilm":
                        exif_dict[name] = _rational_to_float(val)
                    elif name in ("DateTimeOriginal", "DateTimeDigitized", "SubSecTimeOriginal", "LensModel"):
                        exif_dict[name] = str(val).strip().rstrip("\x00")
            except Exception:
                pass

            try:
                gps_sub = raw_exif.get_ifd(0x8825)
                if gps_sub:
                    for g_id, g_val in gps_sub.items():
                        g_name = ExifTags.GPSTAGS.get(g_id, str(g_id))
                        exif_gps[g_name] = g_val
            except Exception:
                pass

    return int(width), int(height), exif_dict, exif_gps


def dji_euler_to_vectors(
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converte gli angoli di gimbal DJI (Pitch, Yaw, Roll)
    in vettori unitari Front e Up nel sistema cartesiano ENU (East=X, North=Y, Up=Z).
    - Yaw: gradi da Nord in senso orario (0=N, 90=E, 180=S, -90/270=W)
    - Pitch: 0 = orizzontale, -90 = nadirale (verso il basso)
    - Roll: rollio attorno all'asse ottico della camera (risolve anche gimbal lock a Pitch=-90° con Roll=180°)
    """
    pitch_r = np.deg2rad(float(pitch_deg))
    yaw_r = np.deg2rad(float(yaw_deg))

    sin_p = np.sin(pitch_r)
    cos_p = np.cos(pitch_r)
    sin_y = np.sin(yaw_r)
    cos_y = np.cos(yaw_r)

    front = np.array([cos_p * sin_y, cos_p * cos_y, sin_p], dtype=np.float64)
    front = front / np.linalg.norm(front)

    up = np.array([-sin_p * sin_y, -sin_p * cos_y, cos_p], dtype=np.float64)
    up = up / np.linalg.norm(up)

    if abs(roll_deg) > 1e-4:
        from scipy.spatial.transform import Rotation as R
        r_rot = R.from_rotvec(np.deg2rad(float(roll_deg)) * front)
        up = r_rot.apply(up)
        up = up / np.linalg.norm(up)

    return front, up


def _opt_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        f = float(str(val).strip())
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _parse_timestamps(xmp: Dict[str, str], exif: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Estrae i timestamp dallo scatto senza inventarli:
    - exposure_gpst_unix_s: derivato da DJI UTCAtExposure (che in DJI Enterprise M3E/M4E usa GPS time)
    - exposure_utc_unix_s: da CreateDate / DateTimeOriginal (+ SubSec) o fallback UTCAtExposure
    - timestamp_iso: stringa leggibile
    """
    utc_exp_str = xmp.get("UTCAtExposure")
    gpst_unix = parse_iso_or_exif_datetime_to_unix(utc_exp_str) if utc_exp_str else None

    create_str = xmp.get("CreateDate") or exif.get("DateTimeOriginal") or exif.get("DateTime")
    utc_unix = parse_iso_or_exif_datetime_to_unix(create_str) if create_str else None
    if utc_unix is not None and exif.get("SubSecTimeOriginal"):
        try:
            sub = float("0." + str(exif["SubSecTimeOriginal"]).strip())
            utc_unix += sub
        except ValueError:
            pass

    if utc_unix is None and gpst_unix is not None:
        utc_unix = gpst_unix

    iso_label = utc_exp_str or create_str
    return gpst_unix, utc_unix, iso_label


class DatasetImporter:
    last_load_report: Dict[str, Any] = {}

    @staticmethod
    def load_from_descr_csv(
        csv_path: str,
        image_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, CameraModel], CoordinateManager]:
        """
        Carica i metadati delle camere dal file descr.csv (compatibilità sistema Brasile).
        Verifica colonne obbligatorie, finitudine numerica, ortogonalità Front/Up,
        coerenza FOV e presenza opzionale di dimensioni immagine o colonne RTK reali.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File {csv_path} non trovato.")

        required_cols = {
            "X", "Y", "Z",
            "FrontX", "FrontY", "FrontZ",
            "UpX", "UpY", "UpZ",
            "VFOV", "HFOV", "FocalLength",
            "Timestamp", "AlignmentMode", "Filename",
        }

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            missing_cols = sorted(required_cols - fieldnames)
            if missing_cols:
                raise ValueError(
                    f"Il file descr.csv non contiene tutte le colonne richieste: {missing_cols}"
                )
            raw_rows = list(reader)

        if not raw_rows:
            raise ValueError(f"Il file {csv_path} è vuoto.")

        search_dir = image_dir or os.path.dirname(os.path.abspath(csv_path))

        # Determina origine locale dal primo punto valido per stabilizzare la precisione
        first_x = float(raw_rows[0]["X"])
        first_y = float(raw_rows[0]["Y"])
        first_z = float(raw_rows[0]["Z"])
        if not (np.isfinite(first_x) and np.isfinite(first_y) and np.isfinite(first_z)):
            raise ValueError("Coordinate XYZ non valide nella prima riga di descr.csv.")

        coord_mgr = CoordinateManager(origin_xyz=np.array([first_x, first_y, first_z], dtype=np.float64))

        cameras: Dict[str, CameraModel] = {}
        warnings: List[str] = []
        verified_dim_count = 0

        for idx, row in enumerate(raw_rows, start=1):
            filename = str(row["Filename"]).strip()
            if not filename:
                raise ValueError(f"Riga {idx} di descr.csv priva di Filename.")

            x = float(row["X"])
            y = float(row["Y"])
            z = float(row["Z"])
            fx = float(row["FrontX"])
            fy = float(row["FrontY"])
            fz = float(row["FrontZ"])
            ux = float(row["UpX"])
            uy = float(row["UpY"])
            uz = float(row["UpZ"])
            vfov = float(row["VFOV"])
            hfov = float(row["HFOV"])
            focal = float(row["FocalLength"])
            ts = float(row["Timestamp"])
            align_mode = str(row.get("AlignmentMode", "")).strip()

            vals = [x, y, z, fx, fy, fz, ux, uy, uz, vfov, hfov, focal, ts]
            if not all(np.isfinite(vals)):
                raise ValueError(f"Valori non finiti (NaN/Inf) in descr.csv alla riga {idx} ({filename}).")
            if vfov <= 0.0 or hfov <= 0.0 or focal <= 0.0:
                raise ValueError(f"Parametri ottici non positivi in descr.csv alla riga {idx} ({filename}).")

            v_front = np.array([fx, fy, fz], dtype=np.float64)
            v_up = np.array([ux, uy, uz], dtype=np.float64)
            nf = np.linalg.norm(v_front)
            nu = np.linalg.norm(v_up)
            if abs(nf - 1.0) > 0.05 or abs(nu - 1.0) > 0.05:
                warnings.append(f"{filename}: vettori Front/Up non unitari in descr.csv (|F|={nf:.3f}, |U|={nu:.3f}).")
            dot_fu = abs(np.dot(v_front / max(nf, 1e-9), v_up / max(nu, 1e-9)))
            if dot_fu > 0.05:
                raise ValueError(
                    f"{filename}: vettori Front e Up in descr.csv non ortogonali (dot={dot_fu:.3f})."
                )

            # Verifica presenza dimensioni reali da disco o colonne opzionali ImageWidth/ImageHeight
            img_w = None
            img_h = None
            dim_verified = False
            col_w = _opt_float(row.get("ImageWidth"))
            col_h = _opt_float(row.get("ImageHeight"))
            if col_w and col_h and col_w > 0 and col_h > 0:
                img_w, img_h = int(col_w), int(col_h)
                dim_verified = True
            else:
                img_candidate = os.path.join(search_dir, filename)
                if os.path.exists(img_candidate):
                    try:
                        with Image.open(img_candidate) as im:
                            img_w, img_h = im.size
                        dim_verified = True
                    except Exception:
                        pass

            tan_h = float(np.tan(np.deg2rad(hfov / 2.0)))
            tan_v = float(np.tan(np.deg2rad(vfov / 2.0)))
            fov_aspect = tan_h / tan_v

            if dim_verified and img_w and img_h:
                verified_dim_count += 1
                img_aspect = img_w / img_h
                if abs(img_aspect - fov_aspect) > 0.03:
                    warnings.append(
                        f"{filename}: discrepanza aspect ratio immagine ({img_aspect:.3f}) vs FOV CSV ({fov_aspect:.3f})."
                    )
                cal_source = "DESCR_CSV_WITH_VERIFIED_DIMS"
            else:
                # Ricava una griglia di riferimento proporzionata esattamente a HFOV/VFOV senza fingere che il file immagine sia stato letto
                img_h = 3956
                img_w = int(round(img_h * fov_aspect))
                cal_source = "DESCR_CSV_FOV_ONLY"

            fx_pix = (img_w / 2.0) / tan_h
            fy_pix = (img_h / 2.0) / tan_v

            cal = OpticalCalibration(
                image_width=img_w,
                image_height=img_h,
                fx_pix=fx_pix,
                fy_pix=fy_pix,
                cx_pix=img_w / 2.0,
                cy_pix=img_h / 2.0,
                distortion_active=False,
                focal_length_mm=focal,
                hfov_deg=hfov,
                vfov_deg=vfov,
                source=cal_source,
                is_complete=True,
                is_validated=False,  # Distorsione / modello camera completo non presente in descr.csv standard
                image_dimensions_verified=dim_verified,
                notes=[
                    "Parametri HFOV/VFOV/FocalLength letti da descr.csv.",
                    "Dimensioni sensore verificate da file/colonne." if dim_verified else (
                        "File immagine assente: usata griglia geometrica coerente con VFOV/HFOV (non verificata in pixel reali)."
                    ),
                    "Distorsione ottica non inclusa nello schema descr.csv: validità metrologica limitata.",
                ],
            )

            # Controlla eventuali colonne GNSS/RTK opzionali se presenti in descr.csv esteso
            opt_flag = row.get("RtkFlag")
            rtk_flag_val = int(float(opt_flag)) if opt_flag not in (None, "") else None
            std_lat = _opt_float(row.get("RtkStdLat"))
            std_lon = _opt_float(row.get("RtkStdLon"))
            std_hgt = _opt_float(row.get("RtkStdHgt"))
            is_fix = (rtk_flag_val == RTK_FLAG_FIX) if rtk_flag_val is not None else False
            has_stds = all(s is not None and s > 0 for s in (std_lat, std_lon, std_hgt))

            gnss_mode = "RTK" if rtk_flag_val is not None else "DESCR_CSV"
            gnss = GNSSQuality(
                mode=gnss_mode,
                rtk_flag=rtk_flag_val,
                is_fix=is_fix,
                altitude_type="Metric_CSV",
                std_lat_m=std_lat,
                std_lon_m=std_lon,
                std_hgt_m=std_hgt,
                source_file=os.path.basename(csv_path),
                is_validated=bool(rtk_flag_val == RTK_FLAG_FIX and has_stds),
                notes=[
                    f"AlignmentMode='{align_mode or 'N/D'}'.",
                    "Colonne RTK reali presenti in CSV." if rtk_flag_val is not None else (
                        "Stato RTK/PPK e deviazioni standard GNSS non documentati nello schema descr.csv standard."
                    ),
                ],
            )

            pos_local = coord_mgr.to_local(np.array([x, y, z], dtype=np.float64))
            cam = CameraModel(
                name=filename,
                pos=pos_local,
                front=v_front,
                up=v_up,
                hfov_deg=hfov,
                vfov_deg=vfov,
                focal_length_mm=focal,
                image_width=img_w,
                image_height=img_h,
                fx_pix=fx_pix,
                fy_pix=fy_pix,
                cx_pix=img_w / 2.0,
                cy_pix=img_h / 2.0,
                calibration=cal,
                rtk_fix=is_fix,
                rtk_flag=rtk_flag_val,
                rtk_std=gnss.max_horiz_std_m,
                rtk_std_hgt=std_hgt,
                gnss_quality=gnss,
                timestamp=ts,
                metadata={
                    "source_type": "descr.csv",
                    "source_path": os.path.abspath(csv_path),
                    "alignment_mode": align_mode,
                    "exposure_utc_unix_s": ts,
                    "exposure_gpst_unix_s": ts,
                    "timestamp_iso": str(ts),
                    "pose_source": "DESCR_CSV_FRONT_UP",
                    "warnings": list(warnings),
                },
            )
            cameras[filename] = cam

        if verified_dim_count == 0:
            warnings.append(
                "Nessuna immagine fisica corrispondente trovata accanto a descr.csv: "
                "impossibile verificare le dimensioni reali in pixel né eseguire marcatura su foto originali."
            )

        DatasetImporter.last_load_report = {
            "source_type": "descr.csv",
            "source_path": os.path.abspath(csv_path),
            "total_loaded": len(cameras),
            "rejected_files": [],
            "warnings": warnings,
            "detected_ppk_files": [],
        }
        return cameras, coord_mgr

    @staticmethod
    def load_from_image_files(
        image_paths: List[str],
        coord_mgr: Optional[CoordinateManager] = None,
        documented_profiles: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, CameraModel], CoordinateManager]:
        """
        Carica e valida i metadati reali dai file JPG DJI (EXIF + XMP).
        Non sostituisce mai valori mancanti (coordinate, altitudine, gimbal, focali, flag RTK) con zeri o default.
        """
        if not image_paths:
            raise ValueError("Nessuna immagine fornita.")

        profiles = documented_profiles if documented_profiles is not None else load_documented_profiles()
        parsed_entries: List[Dict[str, Any]] = []
        rejected_files: List[Dict[str, str]] = []
        global_warnings: List[str] = []

        for img_path in sorted(image_paths):
            fname = os.path.basename(img_path)
            if not os.path.exists(img_path):
                rejected_files.append({"file": fname, "reason": "File non trovato su disco."})
                continue

            xmp = parse_dji_xmp(img_path)
            width, height, exif_dict, exif_gps = parse_image_exif(img_path)

            # Controllo tag Orientation EXIF
            orientation_tag = exif_dict.get("Orientation", 1)
            if orientation_tag not in (1, None):
                global_warnings.append(
                    f"{fname}: EXIF Orientation={orientation_tag} != 1; verificare corrispondenza assi immagine/gimbal."
                )

            # 1. Calibrazione ottica reale
            calibration = resolve_dji_calibration(
                xmp=xmp,
                exif=exif_dict,
                image_width=width,
                image_height=height,
                documented_profiles=profiles,
            )

            # 2. Valutazione GNSS / RTK reale
            gnss = evaluate_dji_xmp_gnss(xmp=xmp, exif_gps=exif_gps)
            if gnss.lat_deg is None or gnss.lon_deg is None or gnss.alt_m is None:
                reason = "; ".join(gnss.notes) or "Coordinate 3D GNSS mancanti."
                rejected_files.append({"file": fname, "reason": f"Esclusa (no zero-fallback): {reason}"})
                continue

            # Check null-island (0,0) o coordinate nulle sospette
            if abs(gnss.lat_deg) < 1e-6 and abs(gnss.lon_deg) < 1e-6:
                rejected_files.append(
                    {"file": fname, "reason": "Esclusa: coordinate (0.0, 0.0) non valide."}
                )
                continue

            # 3. Orientamento reale Gimbal & Drone (nessun fallback -90/0/0)
            g_pitch = _opt_float(xmp.get("GimbalPitchDegree"))
            g_yaw = _opt_float(xmp.get("GimbalYawDegree"))
            g_roll = _opt_float(xmp.get("GimbalRollDegree"))

            if g_pitch is None or g_yaw is None or g_roll is None:
                rejected_files.append(
                    {
                        "file": fname,
                        "reason": "Angoli di orientamento Gimbal (Pitch/Yaw/Roll) incompleti nei metadati XMP.",
                    }
                )
                continue

            front, up = dji_euler_to_vectors(g_pitch, g_yaw, g_roll)

            f_pitch = _opt_float(xmp.get("FlightPitchDegree"))
            f_yaw = _opt_float(xmp.get("FlightYawDegree"))
            f_roll = _opt_float(xmp.get("FlightRollDegree"))

            # 4. Timestamp ed informazioni camera/sensore
            gpst_unix, utc_unix, iso_ts = _parse_timestamps(xmp, exif_dict)
            seq_id = extract_dji_seq_from_filename(fname)

            drone_model = (
                xmp.get("ProductName")
                or xmp.get("DroneModel")
                or exif_dict.get("Model")
                or "Modello non dichiarato"
            )
            make = exif_dict.get("Make") or xmp.get("Make") or "N/D"
            shutter_type = xmp.get("ShutterType") or "Non dichiarato"
            image_source = xmp.get("ImageSource") or "Non dichiarato"
            cam_serial = xmp.get("CameraSerialNumber") or xmp.get("DroneSerialNumber") or "N/D"

            lrf_info = None
            if xmp.get("LRFStatus") or xmp.get("LRFTargetDistance"):
                lrf_info = {
                    "status": xmp.get("LRFStatus"),
                    "distance_m": _opt_float(xmp.get("LRFTargetDistance")),
                    "target_lat": _opt_float(xmp.get("LRFTargetLat")),
                    "target_lon": _opt_float(xmp.get("LRFTargetLon")),
                    "target_abs_alt": _opt_float(xmp.get("LRFTargetAbsAlt")),
                }

            parsed_entries.append(
                {
                    "name": fname,
                    "path": os.path.abspath(img_path),
                    "gnss": gnss,
                    "calibration": calibration,
                    "front": front,
                    "up": up,
                    "metadata": {
                        "source_type": "DJI_IMAGE",
                        "source_path": os.path.abspath(img_path),
                        "make": make,
                        "drone_model": drone_model,
                        "camera_serial": cam_serial,
                        "image_source": image_source,
                        "shutter_type": shutter_type,
                        "exif_orientation": orientation_tag,
                        "gimbal_euler_deg": {"pitch": g_pitch, "yaw": g_yaw, "roll": g_roll},
                        "flight_euler_deg": {"pitch": f_pitch, "yaw": f_yaw, "roll": f_roll},
                        "exposure_gpst_unix_s": gpst_unix,
                        "exposure_utc_unix_s": utc_unix,
                        "timestamp_iso": iso_ts,
                        "dji_seq_id": seq_id,
                        "lrf": lrf_info,
                        "pose_source": "DJI_GIMBAL_XMP",
                    },
                }
            )

        if not parsed_entries:
            detail_reasons = "; ".join(f"{r['file']}: {r['reason']}" for r in rejected_files)
            raise ValueError(
                "Nessuna fotografia valida caricabile (dati geografici o orientamento gimbal assenti). "
                f"Dettagli: {detail_reasons}"
            )

        # Inizializza CoordinateManager sul primo punto valido (ENU tangente al Nord Vero per coerenza con GimbalYaw DJI)
        if coord_mgr is None:
            ref_gnss: GNSSQuality = parsed_entries[0]["gnss"]
            utm_zone = int((ref_gnss.lon_deg + 180.0) / 6.0) + 1
            is_north = (ref_gnss.lat_deg >= 0.0)
            coord_mgr = CoordinateManager(utm_zone=utm_zone, is_northern=is_north)
            x0, y0, z0 = coord_mgr.wgs84_to_utm(ref_gnss.lon_deg, ref_gnss.lat_deg, ref_gnss.alt_m)
            coord_mgr.set_origin(np.array([x0, y0, z0], dtype=np.float64))
            coord_mgr.set_local_enu_origin(ref_gnss.lon_deg, ref_gnss.lat_deg, ref_gnss.alt_m)

        # Controllo omogeneità sistema altimetrico tra le immagini
        alt_types = {e["gnss"].altitude_type for e in parsed_entries}
        if len(alt_types) > 1:
            global_warnings.append(
                f"Tipi di altitudine misti nel dataset ({sorted(alt_types)}): verificare omogeneità delle quote."
            )

        cameras: Dict[str, CameraModel] = {}
        for entry in parsed_entries:
            g: GNSSQuality = entry["gnss"]
            cal: OpticalCalibration = entry["calibration"]
            pos_local = coord_mgr.wgs84_to_local(g.lon_deg, g.lat_deg, g.alt_m)

            if np.linalg.norm(pos_local[:2]) > 5000.0:
                global_warnings.append(
                    f"{entry['name']}: stazione distante > 5 km dall'origine locale ({np.linalg.norm(pos_local[:2]):.0f} m)."
                )

            cam = CameraModel(
                name=entry["name"],
                pos=pos_local,
                front=entry["front"],
                up=entry["up"],
                calibration=cal,
                gnss_quality=g,
                timestamp=entry["metadata"].get("exposure_utc_unix_s"),
                metadata=entry["metadata"],
            )
            cameras[entry["name"]] = cam

        # Rileva eventuali file PPK / raw presenti nella cartella delle immagini (senza applicarli di nascosto)
        folders = {os.path.dirname(os.path.abspath(p)) for p in image_paths}
        detected_ppk: List[str] = []
        for fld in sorted(folders):
            for pat in ("*.MRK", "*.mrk", "*.pos", "*.POS", "*ppk*.csv", "*PPK*.csv"):
                for found in glob.glob(os.path.join(fld, pat)):
                    detected_ppk.append(found)

        DatasetImporter.last_load_report = {
            "source_type": "DJI_IMAGE",
            "total_loaded": len(cameras),
            "rejected_files": rejected_files,
            "warnings": global_warnings,
            "detected_ppk_files": sorted(set(detected_ppk)),
        }
        return cameras, coord_mgr

    @staticmethod
    def export_to_descr_csv(
        cameras: Dict[str, CameraModel],
        coord_mgr: CoordinateManager,
        csv_path: str,
        default_alignment_mode: str = "3D",
    ) -> str:
        """
        Esporta le camere nel formato standard descr.csv compatibile con il sistema Brasile.
        Converte posizioni locali -> metriche globali tramite coord_mgr.to_world.
        """
        fieldnames = [
            "Index", "Timestamp", "X", "Y", "Z",
            "FrontX", "FrontY", "FrontZ",
            "UpX", "UpY", "UpZ",
            "VFOV", "HFOV", "FocalLength", "AlignmentMode", "Filename",
        ]
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, (cname, cam) in enumerate(cameras.items()):
                if cam.vfov_deg is None or cam.hfov_deg is None:
                    raise ValueError(
                        f"Impossibile esportare {cname} in descr.csv: VFOV/HFOV reali mancanti."
                    )
                if cam.focal_length_mm is None:
                    raise ValueError(
                        f"Impossibile esportare {cname} in descr.csv: FocalLength in mm mancante."
                    )
                if cam.timestamp is None:
                    raise ValueError(
                        f"Impossibile esportare {cname} in descr.csv: Timestamp dello scatto mancante."
                    )
                world_xyz = coord_mgr.to_world(cam.pos)
                vfov = float(cam.vfov_deg)
                hfov = float(cam.hfov_deg)
                focal = float(cam.focal_length_mm)
                ts = float(cam.timestamp)
                align = cam.metadata.get("alignment_mode") or default_alignment_mode
                writer.writerow(
                    {
                        "Index": idx,
                        "Timestamp": f"{ts:.3f}",
                        "X": f"{world_xyz[0]:.4f}",
                        "Y": f"{world_xyz[1]:.4f}",
                        "Z": f"{world_xyz[2]:.4f}",
                        "FrontX": f"{cam.front[0]:.9f}",
                        "FrontY": f"{cam.front[1]:.9f}",
                        "FrontZ": f"{cam.front[2]:.9f}",
                        "UpX": f"{cam.up[0]:.9f}",
                        "UpY": f"{cam.up[1]:.9f}",
                        "UpZ": f"{cam.up[2]:.9f}",
                        "VFOV": f"{vfov:.6f}",
                        "HFOV": f"{hfov:.6f}",
                        "FocalLength": f"{focal:.3f}",
                        "AlignmentMode": align,
                        "Filename": cname,
                    }
                )
        return os.path.abspath(csv_path)
