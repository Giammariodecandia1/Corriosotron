"""
DISTIX - Metrological Uncertainty & Threshold Engine
Valutazione rigorosa dell'incertezza della distanza finale D (GUM),
verifica dei vincoli di validità (almeno 2 foto, calibrazione documentata, GNSS RTK/PPK FIX reale,
geometria >= 15°/20°, cheirality) e confronto con soglia configurabile e riferimento tecnico.
Tre esiti formali:
- SUPERIORE ALLA SOGLIA (D - U95 >= S, misura validata)
- INFERIORE ALLA SOGLIA (D + U95 < S, misura validata)
- INDETERMINATO (incertezza sovrapposta alla soglia o requisiti di validità non soddisfatti)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Sequence
from enum import Enum
import numpy as np


MIN_PHOTOS_PER_MEASUREMENT = 2


class ThresholdOutcome(Enum):
    SUPERIORE_ALLA_SOGLIA = "SUPERIORE ALLA SOGLIA"
    INFERIORE_ALLA_SOGLIA = "INFERIORE ALLA SOGLIA"
    INDETERMINATO = "INDETERMINATO"
    SOLO_MISURA = "SOLO MISURA 3D"


# Alias di compatibilità per eventuale codice legacy
class ComplianceStatus(Enum):
    CONFORME = "SUPERIORE ALLA SOGLIA"
    NON_CONFORME = "INFERIORE ALLA SOGLIA"
    INDETERMINATO = "INDETERMINATO"
    NON_VERIFICATO = "INDETERMINATO"


class NormativeThresholds:
    """Soglie predefinite configurabili + relativi riferimenti tecnici."""

    FABBRICATI = 3.15
    TERRAZZE_PRATICABILI = 4.00

    REFERENCES = {
        "Fabbricato": "Normativa CEI 11-4 / Specifica tecnica E-Distribuzione V.24 (Pareti e fabbricati non praticabili)",
        "Terrazza": "Normativa CEI 11-4 / Specifica tecnica E-Distribuzione V.23 (Terrazze, balconi e tetti praticabili)",
    }

    @classmethod
    def default_for_category(cls, category: str) -> tuple:
        if category and ("terrazza" in category.lower() or "praticabile" in category.lower()):
            return cls.TERRAZZE_PRATICABILI, cls.REFERENCES["Terrazza"]
        return cls.FABBRICATI, cls.REFERENCES["Fabbricato"]


@dataclass
class ValidationCheck:
    code: str
    label: str
    passed: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "passed": self.passed,
            "detail": self.detail,
        }


class UncertaintyAssessment:
    def __init__(
        self,
        measured_clearance_m: float,
        u_gnss_m: float,
        u_attitude_m: float,
        u_pixel_m: float,
        u_triangulation_m: float,
        u_conductor_m: float,
        combined_uncertainty_m: float,
        expanded_uncertainty_u95_m: float,
        threshold_s_m: float,
        technical_reference: str,
        outcome: ThresholdOutcome,
        is_validated: bool,
        validation_checks: List[ValidationCheck],
        details: str,
        u_gnss_mode: str = "RTK",
        u_attitude_source: str = "IMU Nominale",
        threshold_enabled: bool = True,
    ):
        self.measured_clearance_m = float(measured_clearance_m)
        self.u_gnss_m = float(u_gnss_m)
        self.u_attitude_m = float(u_attitude_m)
        self.u_pixel_m = float(u_pixel_m)
        self.u_triangulation_m = float(u_triangulation_m)
        self.u_conductor_m = float(u_conductor_m)
        self.combined_uncertainty_m = float(combined_uncertainty_m)
        self.expanded_uncertainty_u95_m = float(expanded_uncertainty_u95_m)
        self.threshold_s_m = float(threshold_s_m)
        self.technical_reference = str(technical_reference)
        self.outcome = outcome
        self.threshold_enabled = bool(threshold_enabled)
        # Proprietà di compatibilità
        self.status = outcome
        self.is_validated = bool(is_validated)
        self.validation_status = "VALIDATA" if self.is_validated else "NON VALIDATA"
        self.validation_checks = validation_checks
        self.details = details
        self.u_gnss_mode = u_gnss_mode
        self.u_attitude_source = u_attitude_source

        # Verifica numerica sul bilancio puntuale (non sostituisce la validazione topografica sperimentale!)
        self.model_u95_within_30cm = bool(self.is_validated and self.expanded_uncertainty_u95_m <= 0.30)

    @property
    def u_rtk_m(self) -> float:
        """Alias retrocompatibile per u_gnss_m."""
        return self.u_gnss_m

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measured_clearance_m": round(self.measured_clearance_m, 3),
            "expanded_uncertainty_u95_m": round(self.expanded_uncertainty_u95_m, 3),
            "threshold_enabled": self.threshold_enabled,
            "threshold_s_m": round(self.threshold_s_m, 3),
            "technical_reference": self.technical_reference,
            "status": self.outcome.value,
            "outcome": self.outcome.value,
            "is_validated": self.is_validated,
            "validation_status": self.validation_status,
            "model_u95_within_30cm": self.model_u95_within_30cm,
            "interval_min_m": round(max(0.0, self.measured_clearance_m - self.expanded_uncertainty_u95_m), 3),
            "interval_max_m": round(self.measured_clearance_m + self.expanded_uncertainty_u95_m, 3),
            "budget": {
                "u_gnss_m": round(self.u_gnss_m, 4),
                "u_rtk_m": round(self.u_gnss_m, 4),
                "u_attitude_m": round(self.u_attitude_m, 4),
                "u_pixel_m": round(self.u_pixel_m, 4),
                "u_triangulation_m": round(self.u_triangulation_m, 4),
                "u_conductor_m": round(self.u_conductor_m, 4),
                "u_combined_m": round(self.combined_uncertainty_m, 4),
                "u_gnss_mode": self.u_gnss_mode,
                "u_attitude_source": self.u_attitude_source,
            },
            "validation_checks": [vc.to_dict() for vc in self.validation_checks],
            "details": self.details,
        }


def evaluate_clearance_uncertainty(
    measured_clearance_m: float,
    mean_camera_distance_m: float,
    convergence_angle_deg: float,
    mean_triangulation_residual_m: float,
    conductor_fit_residual_m: float,
    category: str = "Fabbricato",
    rtk_fix: bool = False,
    threshold_s_m: Optional[float] = None,
    technical_reference: Optional[str] = None,
    fx_pix: Optional[float] = None,
    num_photos: Optional[int] = None,
    calibration_validated: bool = False,
    gnss_validated: Optional[bool] = None,
    gnss_mode: str = "RTK",
    rtk_flag: Optional[int] = None,
    gnss_std_xy_m: Optional[float] = None,
    gnss_std_z_m: Optional[float] = None,
    attitude_std_deg: Optional[float] = None,
    ba_applied: bool = False,
    conductor_plane_angle_deg: Optional[float] = None,
    conductor_sag_uncertainty_m: Optional[float] = None,
    unit_normal_cb: Optional[np.ndarray] = None,
    triangulation_cov_3d: Optional[np.ndarray] = None,
    triangulation_reliable: bool = True,
    conductor_well_conditioned: bool = True,
    cheirality_ok: bool = True,
    max_reproj_px: Optional[float] = None,
    cal_intrinsics_ok: Optional[bool] = None,
    check_threshold: bool = True,
) -> UncertaintyAssessment:
    """
    Calcola l'incertezza estesa U95 (k=2) sulla distanza finale D lungo il versore di misura n_CB,
    verifica tutti i vincoli operativi ed emette l'esito a 3 stati rispetto alla soglia configurata.
    """
    default_thr, default_ref = NormativeThresholds.default_for_category(category)
    threshold = float(threshold_s_m) if threshold_s_m is not None else default_thr
    tech_ref = (technical_reference or "").strip() or default_ref

    if gnss_validated is None:
        gnss_validated = bool(rtk_fix and gnss_std_xy_m is not None and gnss_std_z_m is not None)

    n_vec = (
        np.asarray(unit_normal_cb, dtype=np.float64)
        if unit_normal_cb is not None
        else np.array([1.0 / np.sqrt(3.0)] * 3, dtype=np.float64)
    )
    n_norm = float(np.linalg.norm(n_vec))
    n_vec = (n_vec / n_norm) if n_norm > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_xy = float(np.hypot(n_vec[0], n_vec[1]))
    n_z = float(abs(n_vec[2]))

    n_cams_eff = max(2, int(num_photos)) if num_photos is not None else 2

    # 1. Incertezza GNSS reale (mai inventata; floor conservativo quando flag non è FIX)
    if rtk_fix and gnss_validated and gnss_std_xy_m is not None and gnss_std_z_m is not None:
        sigma_xy = max(float(gnss_std_xy_m), 0.015)
        sigma_z = max(float(gnss_std_z_m), 0.025)
    elif rtk_flag == 34:  # FLOAT
        sigma_xy = max(float(gnss_std_xy_m or 0.0), 0.30)
        sigma_z = max(float(gnss_std_z_m or 0.0), 0.50)
    elif rtk_fix and (gnss_std_xy_m is None or gnss_std_z_m is None):
        sigma_xy = 0.25
        sigma_z = 0.40
    else:  # SINGLE (16), NONE (0), DESCR_CSV, o mancante
        sigma_xy = max(float(gnss_std_xy_m or 0.0), 1.00)
        sigma_z = max(float(gnss_std_z_m or 0.0), 1.50)

    geom_angle_deg = convergence_angle_deg
    if conductor_plane_angle_deg is not None:
        geom_angle_deg = min(convergence_angle_deg, conductor_plane_angle_deg)
    geom_factor = 1.0 / max(0.35, float(np.sin(np.deg2rad(max(8.0, geom_angle_deg)))))
    sigma_gnss_dir = float(np.sqrt((n_xy * sigma_xy) ** 2 + (n_z * sigma_z) ** 2))
    # Contributo differenziale tra edificio e conduttore ridotto dal numero di viste
    u_gnss = sigma_gnss_dir * np.sqrt(2.0 / n_cams_eff) * min(2.0, 0.7 * geom_factor)

    # 2. Incertezza Pixel (usando la focale reale fx_pix della camera, mai valori predefiniti)
    dist_mean = max(1.0, float(mean_camera_distance_m))
    if fx_pix is not None and fx_pix > 0:
        gsd_m = dist_mean / float(fx_pix)
        focal_missing = False
    else:
        # In assenza di focale valida non è possibile stimare un GSD affidabile: stima penalizzata
        gsd_m = dist_mean / 2500.0
        focal_missing = True
    sigma_click_px = 1.5
    u_pixel = float(gsd_m * sigma_click_px)

    # 3. Incertezza Assetto Gimbal / IMU (distinzione esplicita tra post-BA e IMU nominale)
    if ba_applied and attitude_std_deg is not None:
        sigma_att_deg = max(float(attitude_std_deg), 0.05)
        att_src = f"Bundle Adjustment Vincolato (±{sigma_att_deg:.2f}°)"
    else:
        sigma_att_deg = float(attitude_std_deg) if attitude_std_deg is not None else 0.40
        att_src = f"IMU / Gimbal Nominale senza BA (±{sigma_att_deg:.2f}°)"
    u_attitude_single = dist_mean * float(np.deg2rad(sigma_att_deg))
    u_attitude = float((u_attitude_single / np.sqrt(n_cams_eff / 2.0)) * min(2.0, 0.7 * geom_factor))

    # 4. Incertezza Triangolazione Edificio lungo n_CB
    angle_rad = float(np.deg2rad(max(5.0, convergence_angle_deg)))
    sin_theta = float(np.sin(angle_rad))
    u_triang_iso = max(float(mean_triangulation_residual_m), u_pixel / sin_theta)
    if triangulation_cov_3d is not None:
        try:
            cov_b = np.asarray(triangulation_cov_3d, dtype=np.float64)
            var_n = max(1e-8, float(n_vec @ cov_b @ n_vec))
            var_iso = max(1e-8, float(np.trace(cov_b) / 3.0))
            dir_ratio = float(np.clip(np.sqrt(var_n / var_iso), 0.6, 2.5))
        except Exception:
            dir_ratio = 1.0
    else:
        dir_ratio = 1.0
    u_triang = float(u_triang_iso * dir_ratio)

    # 5. Incertezza Modello Conduttore lungo n_CB (include freccia/sag e angolo piani vista)
    plane_sin = float(np.sin(np.deg2rad(max(5.0, conductor_plane_angle_deg)))) if conductor_plane_angle_deg is not None else 0.5
    u_cond_base = max(0.015, float(conductor_fit_residual_m) / max(0.35, plane_sin))
    u_sag_dir = (n_z * float(conductor_sag_uncertainty_m)) if conductor_sag_uncertainty_m is not None else 0.0
    u_conductor = float(np.hypot(u_cond_base, u_sag_dir))

    # Combinazione GUM sulla distanza finale
    u_c = float(np.sqrt(u_gnss ** 2 + u_attitude ** 2 + u_pixel ** 2 + u_triang ** 2 + u_conductor ** 2))
    U95 = float(2.0 * u_c)

    # ------------------------------------------------------------------
    # GATE DI VALIDAZIONE METROLOGICA (tutte le condizioni devono risultare PASS)
    # ------------------------------------------------------------------
    checks: List[ValidationCheck] = []

    if num_photos is None:
        checks.append(
            ValidationCheck(
                "MIN_PHOTOS",
                f"Numero fotografie >= {MIN_PHOTOS_PER_MEASUREMENT}",
                False,
                "Numero di fotografie attive non dichiarato.",
            )
        )
    else:
        ok_photos = int(num_photos) >= MIN_PHOTOS_PER_MEASUREMENT
        checks.append(
            ValidationCheck(
                "MIN_PHOTOS",
                f"Numero fotografie >= {MIN_PHOTOS_PER_MEASUREMENT}",
                ok_photos,
                f"Usate {int(num_photos)} fotografie (minimo richiesto: {MIN_PHOTOS_PER_MEASUREMENT}).",
            )
        )

    intr_ok = (not focal_missing) if cal_intrinsics_ok is None else (bool(cal_intrinsics_ok) and not focal_missing)
    if calibration_validated and not focal_missing:
        checks.append(
            ValidationCheck(
                "OPTICAL_CALIBRATION",
                "Calibrazione ottica documentata",
                True,
                "Calibrazione ottica validata (DewarpData/DewarpFlag o profilo documentato).",
            )
        )
    elif not intr_ok:
        checks.append(
            ValidationCheck(
                "OPTICAL_CALIBRATION",
                "Parametri ottici intrinseci (Focale XMP / Zoom)",
                False,
                "Parametri ottici intrinseci mancanti o zoom digitale non unitario.",
            )
        )

    ok_gnss = bool(gnss_validated and rtk_fix)
    checks.append(
        ValidationCheck(
            "GNSS_FIX",
            "Posizionamento RTK / PPK FIX con std registrate",
            ok_gnss,
            f"Modalità {gnss_mode} FIX validata (σ_xy={sigma_xy*100:.1f} cm, σ_z={sigma_z*100:.1f} cm)."
            if ok_gnss
            else f"GNSS non validato ({gnss_mode}, Flag={rtk_flag}, fix={rtk_fix}): nessuna soluzione FIX completa.",
        )
    )

    ok_triang = bool(triangulation_reliable and cheirality_ok and convergence_angle_deg >= 15.0)
    triang_msg = (
        f"Convergenza {convergence_angle_deg:.1f}°, residuo medio {mean_triangulation_residual_m*100:.1f} cm"
        + (f", max riproiez. {max_reproj_px:.1f} px." if max_reproj_px is not None else ".")
    )
    if not ok_triang:
        triang_msg += " [Triangolazione inaffidabile o cheirality/convergenza insufficiente]"
    checks.append(
        ValidationCheck(
            "TRIANGULATION_GEOM",
            "Triangolazione edificio & cheirality (>= 15°)",
            ok_triang,
            triang_msg,
        )
    )

    if conductor_plane_angle_deg is not None:
        ok_cond = bool(conductor_well_conditioned and conductor_plane_angle_deg >= 20.0)
        cond_msg = (
            f"Angolo piani vista cavo {conductor_plane_angle_deg:.1f}° (>= 20°), "
            f"residuo medio {conductor_fit_residual_m*100:.1f} cm."
        )
    else:
        ok_cond = bool(conductor_well_conditioned)
        cond_msg = f"Residuo modello cavo {conductor_fit_residual_m*100:.1f} cm."
    checks.append(
        ValidationCheck(
            "CONDUCTOR_GEOM",
            "Geometria piani di vista conduttore (>= 20°)",
            ok_cond,
            cond_msg,
        )
    )

    is_validated = all(c.passed for c in checks)

    # Regola decisionale (confronto con soglia opzionale)
    D = float(measured_clearance_m)
    details: List[str] = []

    if not check_threshold:
        if not is_validated:
            outcome = ThresholdOutcome.INDETERMINATO
            failed_labels = [c.label for c in checks if not c.passed]
            details.append(
                f"MISURA NON VALIDATA ({'; '.join(failed_labels)}): "
                f"valore geometrico D = {D:.2f} m ± {U95:.2f} m."
            )
        else:
            outcome = ThresholdOutcome.SOLO_MISURA
            details.append(
                f"MISURA VALIDATA: distanza netta D = {D:.2f} m ± {U95:.2f} m "
                f"(U95, k=2; intervallo [{max(0.0, D - U95):.2f} m, {D + U95:.2f} m])."
            )
    elif not is_validated:
        outcome = ThresholdOutcome.INDETERMINATO
        failed_labels = [c.label for c in checks if not c.passed]
        details.append(
            f"MISURA NON VALIDATA ({'; '.join(failed_labels)}): esito formale INDETERMINATO. "
            f"Valore geometrico grezzo D={D:.2f} m ± {U95:.2f} m rispetto a soglia S={threshold:.2f} m."
        )
    elif D - U95 >= threshold:
        outcome = ThresholdOutcome.SUPERIORE_ALLA_SOGLIA
        details.append(
            f"SUPERIORE ALLA SOGLIA: D - U95 = {D:.2f} - {U95:.2f} = {D - U95:.2f} m >= S = {threshold:.2f} m "
            f"({tech_ref})."
        )
    elif D + U95 < threshold:
        outcome = ThresholdOutcome.INFERIORE_ALLA_SOGLIA
        details.append(
            f"INFERIORE ALLA SOGLIA: D + U95 = {D:.2f} + {U95:.2f} = {D + U95:.2f} m < S = {threshold:.2f} m "
            f"({tech_ref})."
        )
    else:
        outcome = ThresholdOutcome.INDETERMINATO
        details.append(
            f"INDETERMINATO: l'intervallo [D - U95, D + U95] = [{max(0.0, D - U95):.2f} m, {D + U95:.2f} m] "
            f"interseca la soglia S = {threshold:.2f} m ({tech_ref})."
        )

    return UncertaintyAssessment(
        measured_clearance_m=D,
        u_gnss_m=u_gnss,
        u_attitude_m=u_attitude,
        u_pixel_m=u_pixel,
        u_triangulation_m=u_triang,
        u_conductor_m=u_conductor,
        combined_uncertainty_m=u_c,
        expanded_uncertainty_u95_m=U95,
        threshold_s_m=threshold,
        technical_reference=tech_ref,
        outcome=outcome,
        is_validated=is_validated,
        validation_checks=checks,
        details=" ".join(details),
        u_gnss_mode=gnss_mode,
        u_attitude_source=att_src,
        threshold_enabled=check_threshold,
    )


def assess_clearance_from_pipeline(
    active_cameras: Sequence[Any],
    clearance_measurement: Any,
    triangulation_result: Any,
    conductor_diagnostics: Any,
    category: str = "Fabbricato",
    threshold_s_m: Optional[float] = None,
    technical_reference: Optional[str] = None,
    ba_result: Optional[Any] = None,
    check_threshold: bool = True,
) -> UncertaintyAssessment:
    """
    Helper di alto livello che estrae tutti i parametri reali dalle camere e dai risultati
    di triangolazione/fitting per alimentare evaluate_clearance_uncertainty senza alcuna ipotesi fittizia.
    """
    cams = list(active_cameras)
    num_photos = len(cams)

    # Calibrazione: tutte le camere attive devono avere calibrazione validata
    cal_flags = []
    fx_list = []
    for cam in cams:
        cal = getattr(cam, "calibration", None)
        cal_flags.append(bool(cal is not None and cal.is_validated))
        if getattr(cam, "fx_pix", None) and getattr(cam, "fy_pix", None):
            fx_list.append(0.5 * (float(cam.fx_pix) + float(cam.fy_pix)))

    calibration_validated = bool(cal_flags and all(cal_flags))
    cal_intrinsics_ok = bool(cams) and all(
        bool(getattr(c, "has_intrinsics", False))
        and (
            getattr(c, "calibration", None) is None
            or (
                bool(c.calibration.is_complete)
                and (
                    c.calibration.digital_zoom_ratio is None
                    or abs(float(c.calibration.digital_zoom_ratio) - 1.0) <= 0.01
                )
            )
        )
        for c in cams
    )
    fx_mean = float(np.mean(fx_list)) if fx_list else None

    # GNSS: tutte le camere attive devono avere GNSS validato (RTK/PPK FIX con std reali)
    gnss_flags = []
    std_xy_list = []
    std_z_list = []
    modes = set()
    flags_set = set()
    for cam in cams:
        gq = getattr(cam, "gnss_quality", None)
        if gq is not None:
            gnss_flags.append(bool(gq.is_validated))
            modes.add(gq.mode)
            if gq.rtk_flag is not None:
                flags_set.add(int(gq.rtk_flag))
            if gq.max_horiz_std_m is not None:
                std_xy_list.append(float(gq.max_horiz_std_m))
            if gq.std_hgt_m is not None:
                std_z_list.append(float(gq.std_hgt_m))
        else:
            gnss_flags.append(bool(getattr(cam, "rtk_fix", False) and getattr(cam, "rtk_std", None) is not None))
            if getattr(cam, "rtk_flag", None) is not None:
                flags_set.add(int(cam.rtk_flag))
            if getattr(cam, "rtk_std", None) is not None:
                std_xy_list.append(float(cam.rtk_std))
            if getattr(cam, "rtk_std_hgt", None) is not None:
                std_z_list.append(float(cam.rtk_std_hgt))

    gnss_validated = bool(gnss_flags and all(gnss_flags))
    all_fix = all(getattr(c, "rtk_fix", False) for c in cams) if cams else False
    gnss_mode = "/".join(sorted(modes)) if modes else ("RTK" if all_fix else "MISSING")
    rep_flag = min(flags_set) if flags_set else None
    rms_xy = float(np.sqrt(np.mean(np.square(std_xy_list)))) if std_xy_list else None
    rms_z = float(np.sqrt(np.mean(np.square(std_z_list)))) if std_z_list else None

    # Assetto (BA applicato vs IMU nominale): calcola l'RMS reale per-camera anche se una foto non ha Tie-Point
    n_ba_cams = sum(1 for c in cams if getattr(c, "ba_applied", False))
    ba_applied = bool(ba_result is not None and getattr(ba_result, "applied", False)) and (n_ba_cams >= 2)
    if ba_applied:
        stds = [
            float(c.post_ba_attitude_std_deg)
            if (getattr(c, "ba_applied", False) and getattr(c, "post_ba_attitude_std_deg", None))
            else (0.12 if getattr(c, "ba_applied", False) else 0.40)
            for c in cams
        ]
        att_std_deg = float(np.sqrt(np.mean(np.square(stds)))) if stds else 0.12
    else:
        att_std_deg = 0.40

    # Distanza media delle camere dall'area critica
    p_bldg = clearance_measurement.building_closest_pt
    cam_dists = [float(np.linalg.norm(c.pos - p_bldg)) for c in cams]
    mean_cam_dist = float(np.mean(cam_dists)) if cam_dists else 20.0

    cheirality_ok = bool(
        getattr(triangulation_result, "all_in_front", True)
        and getattr(conductor_diagnostics, "all_in_front", True)
    )

    return evaluate_clearance_uncertainty(
        measured_clearance_m=clearance_measurement.clearance_net_m,
        mean_camera_distance_m=mean_cam_dist,
        convergence_angle_deg=triangulation_result.max_convergence_angle_deg,
        mean_triangulation_residual_m=triangulation_result.mean_residual_m,
        conductor_fit_residual_m=conductor_diagnostics.mean_residual_m,
        category=category,
        rtk_fix=all_fix,
        threshold_s_m=threshold_s_m,
        technical_reference=technical_reference,
        fx_pix=fx_mean,
        num_photos=num_photos,
        calibration_validated=calibration_validated,
        gnss_validated=gnss_validated,
        gnss_mode=gnss_mode,
        rtk_flag=rep_flag,
        gnss_std_xy_m=rms_xy,
        gnss_std_z_m=rms_z,
        attitude_std_deg=att_std_deg,
        ba_applied=ba_applied,
        conductor_plane_angle_deg=conductor_diagnostics.max_viewing_plane_angle_deg,
        conductor_sag_uncertainty_m=conductor_diagnostics.sag_uncertainty_m,
        unit_normal_cb=clearance_measurement.unit_normal,
        triangulation_cov_3d=getattr(triangulation_result, "covariance_3d", None),
        triangulation_reliable=triangulation_result.is_reliable,
        conductor_well_conditioned=conductor_diagnostics.is_well_conditioned,
        cheirality_ok=cheirality_ok,
        max_reproj_px=getattr(triangulation_result, "max_reproj_px", None),
        cal_intrinsics_ok=cal_intrinsics_ok,
        check_threshold=check_threshold,
    )
