"""
DISTIX - Optical Calibration Manager
Gestione rigorosa della calibrazione ottica per fotocamere DJI (Mavic 3 Enterprise / Matrice 4E)
e formati compatibili (descr.csv).
Nessun valore intrinseco predefinito viene assunto come dato reale se assente dai metadati
o da un profilo di calibrazione documentato.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import json
import os
import re
import numpy as np


@dataclass
class OpticalCalibration:
    """Rappresenta i parametri intrinseci e di distorsione ottica di una fotocamera."""

    image_width: int
    image_height: int
    fx_pix: Optional[float] = None
    fy_pix: Optional[float] = None
    cx_pix: Optional[float] = None
    cy_pix: Optional[float] = None
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    dewarp_flag: Optional[int] = None
    dewarp_date: Optional[str] = None
    distortion_active: bool = False
    focal_length_mm: Optional[float] = None
    focal_length_35mm: Optional[float] = None
    digital_zoom_ratio: Optional[float] = None
    hfov_deg: Optional[float] = None
    vfov_deg: Optional[float] = None
    source: str = "MISSING"
    is_complete: bool = False
    is_validated: bool = False
    image_dimensions_verified: bool = True
    notes: List[str] = field(default_factory=list)

    @property
    def distortion_coeffs(self) -> np.ndarray:
        """Restituisce il vettore OpenCV [k1, k2, p1, p2, k3] effettivamente applicabile."""
        if not self.distortion_active:
            return np.zeros(5, dtype=np.float64)
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    @property
    def camera_matrix(self) -> Optional[np.ndarray]:
        if None in (self.fx_pix, self.fy_pix, self.cx_pix, self.cy_pix):
            return None
        return np.array(
            [
                [float(self.fx_pix), 0.0, float(self.cx_pix)],
                [0.0, float(self.fy_pix), float(self.cy_pix)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def compute_fov_if_available(self):
        """Calcola HFOV e VFOV effettivi dai parametri intrinseci reali."""
        if self.fx_pix and self.fx_pix > 0 and self.image_width > 0:
            self.hfov_deg = float(2.0 * np.rad2deg(np.arctan((self.image_width / 2.0) / self.fx_pix)))
        if self.fy_pix and self.fy_pix > 0 and self.image_height > 0:
            self.vfov_deg = float(2.0 * np.rad2deg(np.arctan((self.image_height / 2.0) / self.fy_pix)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_dimensions_verified": self.image_dimensions_verified,
            "fx_pix": round(self.fx_pix, 4) if self.fx_pix is not None else None,
            "fy_pix": round(self.fy_pix, 4) if self.fy_pix is not None else None,
            "cx_pix": round(self.cx_pix, 4) if self.cx_pix is not None else None,
            "cy_pix": round(self.cy_pix, 4) if self.cy_pix is not None else None,
            "distortion_coeffs": [float(x) for x in self.distortion_coeffs],
            "distortion_active": self.distortion_active,
            "dewarp_flag": self.dewarp_flag,
            "dewarp_date": self.dewarp_date,
            "focal_length_mm": self.focal_length_mm,
            "focal_length_35mm": self.focal_length_35mm,
            "digital_zoom_ratio": self.digital_zoom_ratio,
            "hfov_deg": round(self.hfov_deg, 4) if self.hfov_deg is not None else None,
            "vfov_deg": round(self.vfov_deg, 4) if self.vfov_deg is not None else None,
            "source": self.source,
            "is_complete": self.is_complete,
            "is_validated": self.is_validated,
            "notes": list(self.notes),
        }


def parse_dewarp_data(raw_value: str, image_width: int, image_height: int) -> Optional[Dict[str, Any]]:
    """
    Analizza la stringa DJI DewarpData:
    Formato atteso: 'YYYY-MM-DD;fx,fy,cx,cy,k1,k2,p1,p2,k3' (data opzionale).
    In DJI XMP, cx e cy sono offset rispetto al centro geometrico (W/2, H/2),
    salvo casi in cui risultino espressi in coordinate assolute vicine a (W/2, H/2).
    """
    if not raw_value or not isinstance(raw_value, str):
        return None
    text = raw_value.strip()
    dewarp_date = None
    if ";" in text:
        date_part, params_part = text.split(";", 1)
        dewarp_date = date_part.strip() or None
    else:
        params_part = text

    tokens = [t.strip() for t in re.split(r"[,\s]+", params_part.strip()) if t.strip()]
    if len(tokens) < 9:
        return None

    try:
        vals = [float(x) for x in tokens[:9]]
    except ValueError:
        return None

    if not all(np.isfinite(vals)):
        return None

    fx, fy, cx_raw, cy_raw, k1, k2, p1, p2, k3 = vals
    if fx <= 0 or fy <= 0:
        return None

    # Distinzione sicura tra offset centrato DJI (tipicamente |offset| << W/4) e coord. assolute
    if abs(cx_raw) < 0.25 * image_width and abs(cy_raw) < 0.25 * image_height:
        cx_px = (image_width / 2.0) + cx_raw
        cy_px = (image_height / 2.0) + cy_raw
        center_convention = "center_offset"
    elif 0.25 * image_width <= cx_raw <= 0.75 * image_width and 0.25 * image_height <= cy_raw <= 0.75 * image_height:
        cx_px = cx_raw
        cy_px = cy_raw
        center_convention = "absolute_pixel"
    else:
        return None

    return {
        "dewarp_date": dewarp_date,
        "fx": fx,
        "fy": fy,
        "cx": cx_px,
        "cy": cy_px,
        "cx_raw": cx_raw,
        "cy_raw": cy_raw,
        "center_convention": center_convention,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "k3": k3,
    }


def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        f = float(val)
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


def load_documented_profiles(profile_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Carica eventuali profili di calibrazione ottica documentati esternamente (JSON).
    Non genera mai profili presunti automaticamente.
    """
    if profile_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        profile_path = os.path.join(base_dir, "config", "documented_calibrations.json")
    if not os.path.exists(profile_path):
        return []
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("profiles"), list):
            return data["profiles"]
    except Exception:
        return []
    return []


def match_documented_profile(
    profiles: List[Dict[str, Any]],
    camera_model: Optional[str],
    camera_serial: Optional[str],
    image_width: int,
    image_height: int,
    image_source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Cerca un profilo documentato compatibile per seriale o modello + risoluzione."""
    for prof in profiles:
        if int(prof.get("image_width", -1)) != int(image_width):
            continue
        if int(prof.get("image_height", -1)) != int(image_height):
            continue
        prof_serial = prof.get("camera_serial")
        if prof_serial:
            if camera_serial and str(prof_serial).strip() == str(camera_serial).strip():
                return prof
            continue
        prof_model = prof.get("camera_model")
        if prof_model and camera_model and str(prof_model).strip().lower() == str(camera_model).strip().lower():
            prof_src = prof.get("image_source")
            if prof_src and image_source and str(prof_src).strip().lower() != str(image_source).strip().lower():
                continue
            return prof
    return None


def resolve_dji_calibration(
    xmp: Dict[str, str],
    exif: Dict[str, Any],
    image_width: int,
    image_height: int,
    documented_profiles: Optional[List[Dict[str, Any]]] = None,
) -> OpticalCalibration:
    """
    Risolve i parametri di calibrazione ottica dai metadati DJI XMP/EXIF.
    Regole di precedenza e validità:
    1. Controlla DigitalZoomRatio (deve essere ~1.0, senza zoom digitale).
    2. Se DewarpFlag e DewarpData sono presenti e validi:
       - DewarpFlag == 0: immagine non corretta in camera -> applica DewarpData (fx,fy,cx,cy,k1,k2,p1,p2,k3).
       - DewarpFlag == 1: immagine già corretta in camera -> NON riapplica distorsione (evita doppie correzioni),
         usa CalibratedFocalLength/OpticalCenter se presenti, altrimenti fx,fy,cx,cy di DewarpData con dist=0.
    3. Se DewarpFlag == 1 senza DewarpData ma con CalibratedFocalLength + CalibratedOpticalCenterX/Y validi:
       - Immagine rettificata in camera, usa parametri pinhole calibrati con dist=0.
    4. Se mancano DewarpFlag/DewarpData ma è presente CalibratedFocalLength (+ centro ottico):
       - Estrae fx,fy,cx,cy senza valori inventati; cerca profilo documentato applicabile;
         se assente, segnala calibrazione parziale / NON validata (distorsione non documentata).
    5. Se mancano sia DewarpData sia CalibratedFocalLength:
       - Cerca profilo documentato; se assente, restituisce calibrazione mancante e non validata.
    """
    notes: List[str] = []
    focal_mm = _safe_float(exif.get("FocalLength"))
    focal_35mm = _safe_float(exif.get("FocalLengthIn35mmFilm"))
    zoom_ratio = _safe_float(exif.get("DigitalZoomRatio"))

    zoom_ok = True
    if zoom_ratio is not None and abs(zoom_ratio - 1.0) > 0.01:
        zoom_ok = False
        notes.append(f"Zoom digitale rilevato (DigitalZoomRatio={zoom_ratio:.2f} != 1.0): calibrazione invalidata.")

    dewarp_flag = _safe_int(xmp.get("DewarpFlag"))
    dewarp_raw = xmp.get("DewarpData")
    parsed_dewarp = parse_dewarp_data(dewarp_raw, image_width, image_height) if dewarp_raw else None

    cal_focal = _safe_float(xmp.get("CalibratedFocalLength"))
    cal_cx = _safe_float(xmp.get("CalibratedOpticalCenterX"))
    cal_cy = _safe_float(xmp.get("CalibratedOpticalCenterY"))

    if dewarp_raw and parsed_dewarp is None:
        notes.append("Tag DewarpData presente ma non valido o malformato.")

    cal = OpticalCalibration(
        image_width=int(image_width),
        image_height=int(image_height),
        dewarp_flag=dewarp_flag,
        focal_length_mm=focal_mm,
        focal_length_35mm=focal_35mm,
        digital_zoom_ratio=zoom_ratio,
        notes=notes,
    )

    # Caso 1: DewarpFlag == 0 con DewarpData valido -> usa distorsione completa
    if dewarp_flag == 0 and parsed_dewarp is not None:
        cal.fx_pix = parsed_dewarp["fx"]
        cal.fy_pix = parsed_dewarp["fy"]
        cal.cx_pix = parsed_dewarp["cx"]
        cal.cy_pix = parsed_dewarp["cy"]
        cal.k1 = parsed_dewarp["k1"]
        cal.k2 = parsed_dewarp["k2"]
        cal.p1 = parsed_dewarp["p1"]
        cal.p2 = parsed_dewarp["p2"]
        cal.k3 = parsed_dewarp["k3"]
        cal.dewarp_date = parsed_dewarp["dewarp_date"]
        cal.distortion_active = True
        cal.source = "DJI_DEWARP_DATA_RAW"
        cal.is_complete = True
        cal.is_validated = zoom_ok
        cal.compute_fov_if_available()
        if zoom_ok:
            cal.notes.append("Calibrazione DJI completa (DewarpFlag=0, distorsione Brown-Conrady attiva).")
        return cal

    # Caso 2: DewarpFlag == 1 (già dewarped in camera) -> evita doppia correzione!
    if dewarp_flag == 1:
        if cal_focal and cal_focal > 0 and cal_cx is not None and cal_cy is not None:
            cal.fx_pix = cal_focal
            cal.fy_pix = cal_focal
            cal.cx_pix = cal_cx
            cal.cy_pix = cal_cy
            cal.source = "DJI_DEWARPED_CALIBRATED_FOCAL"
        elif parsed_dewarp is not None:
            cal.fx_pix = parsed_dewarp["fx"]
            cal.fy_pix = parsed_dewarp["fy"]
            cal.cx_pix = parsed_dewarp["cx"]
            cal.cy_pix = parsed_dewarp["cy"]
            cal.dewarp_date = parsed_dewarp["dewarp_date"]
            cal.source = "DJI_DEWARPED_DEWARP_DATA"
            if cal_focal and cal_focal > 0:
                cal.fx_pix = cal_focal
                cal.fy_pix = cal_focal
        else:
            cal.source = "DJI_DEWARPED_NO_INTRINSICS"
            cal.is_complete = False
            cal.is_validated = False
            cal.notes.append("DewarpFlag=1 ma mancano i parametri intrinseci (CalibratedFocalLength/DewarpData).")
            return cal

        cal.distortion_active = False
        cal.is_complete = True
        cal.is_validated = zoom_ok
        cal.compute_fov_if_available()
        if zoom_ok:
            cal.notes.append("Immagine già rettificata in camera (DewarpFlag=1): modello pinhole senza doppia distorsione.")
        return cal

    # Caso 3: DewarpData presente ma DewarpFlag assente -> ambiguità!
    if parsed_dewarp is not None and dewarp_flag is None:
        cal.fx_pix = parsed_dewarp["fx"]
        cal.fy_pix = parsed_dewarp["fy"]
        cal.cx_pix = parsed_dewarp["cx"]
        cal.cy_pix = parsed_dewarp["cy"]
        cal.k1 = parsed_dewarp["k1"]
        cal.k2 = parsed_dewarp["k2"]
        cal.p1 = parsed_dewarp["p1"]
        cal.p2 = parsed_dewarp["p2"]
        cal.k3 = parsed_dewarp["k3"]
        cal.dewarp_date = parsed_dewarp["dewarp_date"]
        cal.distortion_active = False
        cal.source = "DJI_DEWARP_DATA_MISSING_FLAG"
        cal.is_complete = False
        cal.is_validated = False
        cal.compute_fov_if_available()
        cal.notes.append(
            "DewarpData presente ma DewarpFlag mancante: stato di correzione dell'immagine non verificabile."
        )
        return cal

    # Caso 4: Profilo di calibrazione documentato (se fornito)
    profiles = documented_profiles if documented_profiles is not None else load_documented_profiles()
    cam_model = xmp.get("ProductName") or xmp.get("DroneModel") or exif.get("Model")
    cam_serial = xmp.get("CameraSerialNumber")
    img_source = xmp.get("ImageSource")
    matched_prof = match_documented_profile(
        profiles, cam_model, cam_serial, image_width, image_height, img_source
    )
    if matched_prof is not None:
        cal.fx_pix = float(matched_prof["fx"])
        cal.fy_pix = float(matched_prof["fy"])
        cal.cx_pix = float(matched_prof["cx"])
        cal.cy_pix = float(matched_prof["cy"])
        cal.k1 = float(matched_prof.get("k1", 0.0))
        cal.k2 = float(matched_prof.get("k2", 0.0))
        cal.p1 = float(matched_prof.get("p1", 0.0))
        cal.p2 = float(matched_prof.get("p2", 0.0))
        cal.k3 = float(matched_prof.get("k3", 0.0))
        cal.distortion_active = bool(matched_prof.get("distortion_active", False))
        cal.dewarp_date = matched_prof.get("calibration_date")
        cal.source = f"DOCUMENTED_PROFILE ({matched_prof.get('id', 'custom')})"
        cal.is_complete = True
        cal.is_validated = zoom_ok
        cal.compute_fov_if_available()
        cal.notes.append(f"Applicato profilo di calibrazione documentato: {matched_prof.get('id', 'custom')}.")
        return cal

    # Caso 5: Solo CalibratedFocalLength (+ OpticalCenterX/Y) senza DewarpData/DewarpFlag (es. M4E senza DewarpData)
    if cal_focal and cal_focal > 0:
        cal.fx_pix = cal_focal
        cal.fy_pix = cal_focal
        if cal_cx is not None and cal_cy is not None:
            cal.cx_pix = cal_cx
            cal.cy_pix = cal_cy
            cal.source = "DJI_CALIBRATED_FOCAL_PARTIAL"
            cal.is_complete = True
            cal.notes.append(
                "Presenti CalibratedFocalLength e centro ottico, ma assenti DewarpData/DewarpFlag: "
                "distorsione residua non documentata, calibrazione non validata."
            )
        else:
            cal.source = "DJI_CALIBRATED_FOCAL_NO_CENTER"
            cal.is_complete = False
            cal.notes.append("CalibratedFocalLength presente ma centro ottico e DewarpData/DewarpFlag mancanti.")
        cal.distortion_active = False
        cal.is_validated = False
        cal.compute_fov_if_available()
        return cal

    # Caso 6: Nessuna calibrazione valida trovata
    cal.source = "MISSING"
    cal.is_complete = False
    cal.is_validated = False
    cal.notes.append(
        "Parametri di calibrazione ottica mancanti (assenti DewarpData, CalibratedFocalLength e profilo documentato)."
    )
    return cal


def compute_file_sha256(file_path: str) -> str:
    """Restituisce l'impronta SHA-256 esadecimale di un file immagine/calibrazione."""
    import hashlib

    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_dewarp_profile_from_image(image_path: str) -> Dict[str, Any]:
    """
    Estrae un profilo di calibrazione documentato dai metadati DewarpData reali di una foto DJI
    (es. scatto di calibrazione / missione mapping con Dewarping OFF dello stesso sensore).
    """
    from .metadata_reader import parse_dji_xmp, parse_image_exif

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File immagine non trovato: {image_path}")

    xmp = parse_dji_xmp(image_path)
    width, height, exif, _ = parse_image_exif(image_path)
    raw_dewarp = xmp.get("DewarpData")
    if not raw_dewarp:
        raise ValueError(
            "La fotografia selezionata non contiene il tag XMP DewarpData.\n"
            "Per acquisire DewarpData su DJI Matrice 4E, esegui uno scatto con "
            "Dewarping = OFF (Impostazioni Fotocamera -> ... -> Dewarping) o in missione Fotogrammetria/Area."
        )

    parsed = parse_dewarp_data(raw_dewarp, width, height)
    if parsed is None:
        raise ValueError(f"Tag DewarpData presente ma malformato in {os.path.basename(image_path)}.")

    cam_model = xmp.get("ProductName") or xmp.get("DroneModel") or exif.get("Model") or "DJI Camera"
    cam_serial = xmp.get("CameraSerialNumber") or None
    img_source = xmp.get("ImageSource") or "WideCamera"
    dewarp_flag = _safe_int(xmp.get("DewarpFlag"))
    sha = compute_file_sha256(image_path)
    date_tag = parsed.get("dewarp_date") or "ND"
    serial_slug = re.sub(r"[^A-Za-z0-9_-]", "", cam_serial or "GENERIC")
    prof_id = f"{serial_slug}_{width}x{height}_DEWARPDATA"

    return {
        "id": prof_id,
        "camera_model": cam_model,
        "camera_serial": cam_serial,
        "image_source": img_source,
        "image_width": int(width),
        "image_height": int(height),
        "fx": float(parsed["fx"]),
        "fy": float(parsed["fy"]),
        "cx": float(parsed["cx"]),
        "cy": float(parsed["cy"]),
        "k1": float(parsed["k1"]),
        "k2": float(parsed["k2"]),
        "p1": float(parsed["p1"]),
        "p2": float(parsed["p2"]),
        "k3": float(parsed["k3"]),
        "distortion_active": True,
        "source_dewarp_flag": dewarp_flag,
        "calibration_date": date_tag,
        "reference": f"Estratto da DewarpData di {os.path.basename(image_path)} (SHA256:{sha[:16]})",
    }


def _default_documented_profiles_path() -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_dir, "config", "documented_calibrations.json")


def save_documented_profile(profile: Dict[str, Any], profile_path: Optional[str] = None) -> str:
    """
    Salva o aggiorna un profilo di calibrazione documentato nel file JSON di configurazione.
    Sostituisce eventuali profili esistenti con uguale id o medesima combinazione seriale/risoluzione.
    """
    path = profile_path or _default_documented_profiles_path()
    required = ("id", "image_width", "image_height", "fx", "fy", "cx", "cy")
    missing = [k for k in required if profile.get(k) is None]
    if missing:
        raise ValueError(f"Profilo di calibrazione incompleto, campi mancanti: {missing}")

    ref = str(profile.get("reference") or "").strip()
    if not ref:
        raise ValueError("Il campo 'reference' (riferimento/verifica) è obbligatorio per registrare un profilo.")

    data: Dict[str, Any]
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
            else:
                data = {"schema_version": "2.0", "profiles": []}
        except Exception:
            data = {"schema_version": "2.0", "profiles": []}
    else:
        data = {
            "schema_version": "2.0",
            "description": "Archivio opzionale di profili di calibrazione ottica documentati esternamente.",
            "profiles": [],
        }

    profiles_list = list(data.get("profiles") or [])
    new_id = str(profile["id"]).strip()
    new_serial = str(profile.get("camera_serial") or "").strip()
    new_w = int(profile["image_width"])
    new_h = int(profile["image_height"])
    new_src = str(profile.get("image_source") or "").strip().lower()

    filtered: List[Dict[str, Any]] = []
    for p in profiles_list:
        pid = str(p.get("id") or "").strip()
        p_serial = str(p.get("camera_serial") or "").strip()
        p_w = int(p.get("image_width", -1))
        p_h = int(p.get("image_height", -1))
        p_src = str(p.get("image_source") or "").strip().lower()
        same_serial_dim = (
            bool(new_serial)
            and p_serial == new_serial
            and p_w == new_w
            and p_h == new_h
            and (not new_src or not p_src or p_src == new_src)
        )
        if pid == new_id or same_serial_dim:
            continue
        filtered.append(p)

    clean_entry = {
        "id": new_id,
        "camera_model": profile.get("camera_model"),
        "camera_serial": profile.get("camera_serial"),
        "image_source": profile.get("image_source"),
        "image_width": new_w,
        "image_height": new_h,
        "fx": float(profile["fx"]),
        "fy": float(profile["fy"]),
        "cx": float(profile["cx"]),
        "cy": float(profile["cy"]),
        "k1": float(profile.get("k1", 0.0)),
        "k2": float(profile.get("k2", 0.0)),
        "p1": float(profile.get("p1", 0.0)),
        "p2": float(profile.get("p2", 0.0)),
        "k3": float(profile.get("k3", 0.0)),
        "distortion_active": bool(profile.get("distortion_active", False)),
        "calibration_date": str(profile.get("calibration_date") or ""),
        "reference": ref,
    }
    filtered.append(clean_entry)
    data["profiles"] = filtered

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return os.path.abspath(path)


def remove_documented_profile(profile_id: str, profile_path: Optional[str] = None) -> bool:
    """Rimuove dal file JSON un profilo documentato identificato da profile_id."""
    path = profile_path or _default_documented_profiles_path()
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), list):
        return False
    before = len(data["profiles"])
    data["profiles"] = [p for p in data["profiles"] if str(p.get("id") or "").strip() != str(profile_id).strip()]
    if len(data["profiles"]) == before:
        return False
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return True
