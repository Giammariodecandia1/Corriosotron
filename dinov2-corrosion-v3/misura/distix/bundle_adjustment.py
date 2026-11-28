"""
DISTIX - Constrained Bundle Adjustment
Raffinamento deterministico dell'orientamento delle fotocamere (Yaw, Pitch, Roll)
mediante vincoli sui centri ottici RTK/PPK e tie point statici manuali.
Preserva integralmente i parametri intrinseci e di distorsione (Calibration / DewarpData)
durante l'ottimizzazione e fornisce diagnostica completa su convergenza, RMS e covarianza.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .camera_model import CameraModel
from .triangulation import triangulate_point_rays


class TiePointObservation:
    def __init__(self, point_id: str, camera_name: str, u_pix: float, v_pix: float):
        self.point_id = str(point_id)
        self.camera_name = camera_name
        self.u_pix = float(u_pix)
        self.v_pix = float(v_pix)


@dataclass
class BundleAdjustmentResult:
    applied: bool
    converged: bool
    num_tie_points: int
    num_observations: int
    rms_before_px: Optional[float]
    rms_after_px: Optional[float]
    max_correction_deg: float
    post_ba_attitude_std_deg: Dict[str, float] = field(default_factory=dict)
    corrections: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    excluded_observations: List[str] = field(default_factory=list)
    conductor_aligned_cameras: List[str] = field(default_factory=list)
    tie_point_counts: Dict[str, int] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applied": self.applied,
            "converged": self.converged,
            "num_tie_points": self.num_tie_points,
            "num_observations": self.num_observations,
            "rms_before_px": round(self.rms_before_px, 3) if self.rms_before_px is not None else None,
            "rms_after_px": round(self.rms_after_px, 3) if self.rms_after_px is not None else None,
            "max_correction_deg": round(self.max_correction_deg, 4),
            "post_ba_attitude_std_deg": {k: round(v, 4) for k, v in self.post_ba_attitude_std_deg.items()},
            "corrections": {
                k: [round(x, 4) for x in v] for k, v in self.corrections.items()
            },
            "excluded_observations": list(self.excluded_observations),
            "conductor_aligned_cameras": list(self.conductor_aligned_cameras),
            "tie_point_counts": dict(self.tie_point_counts),
            "notes": self.notes,
        }


def _evaluate_reprojection_rms(
    cams: Dict[str, CameraModel],
    valid_points: Dict[str, List[TiePointObservation]],
) -> Tuple[Optional[float], np.ndarray]:
    """Calcola il vettore residui in pixel e il relativo RMS sui tie point validi."""
    pix_res: List[float] = []
    for _, obs_list in valid_points.items():
        rays = []
        for obs in obs_list:
            c, d = cams[obs.camera_name].pixel_to_ray(obs.u_pix, obs.v_pix)
            rays.append((c, d))
        t_res = triangulate_point_rays(rays)
        P_3d = t_res.point_3d
        for obs in obs_list:
            proj = cams[obs.camera_name].project_point(P_3d)
            if proj is not None:
                u_p, v_p, _ = proj
                pix_res.append(float(u_p - obs.u_pix))
                pix_res.append(float(v_p - obs.v_pix))
            else:
                pix_res.append(500.0)
                pix_res.append(500.0)
    if not pix_res:
        return None, np.array([], dtype=np.float64)
    arr = np.asarray(pix_res, dtype=np.float64)
    # RMS radiale 2D medio per osservazione: sqrt(mean(du^2 + dv^2))
    pair_sq = arr[0::2] ** 2 + arr[1::2] ** 2
    rms = float(np.sqrt(np.mean(pair_sq)))
    return rms, arr


def _apply_world_rotation(cam: CameraModel, tilt_e_deg: float, tilt_n_deg: float, az_z_deg: float):
    """Applica al clone una rotazione incrementale espressa nel sistema mondo ENU (dTiltE, dTiltN, dAzZ)."""
    rot_vec = np.deg2rad(np.array([tilt_e_deg, tilt_n_deg, az_z_deg], dtype=np.float64))
    R_delta_w = Rotation.from_rotvec(rot_vec).as_matrix()
    R_new = R_delta_w @ cam.rotation_matrix
    cam.right = R_new[:, 0] / np.linalg.norm(R_new[:, 0])
    cam.up = -R_new[:, 1] / np.linalg.norm(R_new[:, 1])
    cam.front = R_new[:, 2] / np.linalg.norm(R_new[:, 2])


def _solve_ba_core(
    base_clones: Dict[str, CameraModel],
    active_cam_names: List[str],
    valid_points: Dict[str, List[TiePointObservation]],
    max_angular_correction_deg: float,
):
    """Esegue il solver a due stadi (soft_l1 + cauchy) su un set di valid_points senza modificare le camere reali."""
    rms_before, _ = _evaluate_reprojection_rms(base_clones, valid_points)

    num_cams = len(active_cam_names)
    init_params = np.zeros(num_cams * 3, dtype=np.float64)

    w_tilt = 2.5
    w_az = 0.35
    prior_weights = np.tile(np.array([w_tilt, w_tilt, w_az], dtype=np.float64), num_cams)

    max_ang = float(max_angular_correction_deg)
    max_tilt = min(2.5, max_ang)
    per_cam_bound = np.array([max_tilt, max_tilt, max_ang], dtype=np.float64)
    bounds_high = np.tile(per_cam_bound, num_cams)
    bounds_low = -bounds_high

    working_clones = {cn: base_clones[cn].clone() for cn in active_cam_names}

    def _update_working_clones(params: np.ndarray):
        for idx, cname in enumerate(active_cam_names):
            tx, ty, tz = params[idx * 3 : (idx + 1) * 3]
            c_work = working_clones[cname]
            c_base = base_clones[cname]
            rot_vec = np.deg2rad(np.array([tx, ty, tz], dtype=np.float64))
            R_delta_w = Rotation.from_rotvec(rot_vec).as_matrix()
            R_new = R_delta_w @ c_base.rotation_matrix
            c_work.right = R_new[:, 0] / np.linalg.norm(R_new[:, 0])
            c_work.up = -R_new[:, 1] / np.linalg.norm(R_new[:, 1])
            c_work.front = R_new[:, 2] / np.linalg.norm(R_new[:, 2])

    def residuals(params: np.ndarray) -> np.ndarray:
        _update_working_clones(params)
        _, pix_arr = _evaluate_reprojection_rms(working_clones, valid_points)
        prior_arr = prior_weights * params
        return np.concatenate([pix_arr, prior_arr])

    stage2_init = init_params
    best_opt = None
    best_rms = rms_before if rms_before is not None else float("inf")

    if rms_before is not None and rms_before > 8.0:
        opt_coarse = least_squares(
            residuals,
            init_params,
            bounds=(bounds_low, bounds_high),
            loss="soft_l1",
            f_scale=max(10.0, float(rms_before)),
            max_nfev=300,
        )
        _update_working_clones(opt_coarse.x)
        rms_s1, _ = _evaluate_reprojection_rms(working_clones, valid_points)
        if opt_coarse.success and rms_s1 is not None and rms_s1 < best_rms:
            stage2_init = opt_coarse.x
            best_opt = opt_coarse
            best_rms = rms_s1

    opt_fine = least_squares(
        residuals,
        stage2_init,
        bounds=(bounds_low, bounds_high),
        loss="cauchy",
        f_scale=2.5,
        max_nfev=300,
    )
    _update_working_clones(opt_fine.x)
    rms_fine, _ = _evaluate_reprojection_rms(working_clones, valid_points)
    if opt_fine.success and rms_fine is not None and (best_opt is None or rms_fine <= best_rms + 0.05):
        opt_res = opt_fine
    elif best_opt is not None:
        opt_res = best_opt
    else:
        opt_res = opt_fine

    opt_clones: Dict[str, CameraModel] = {}
    corrections: Dict[str, Tuple[float, float, float]] = {}
    max_corr = 0.0
    for idx, cname in enumerate(active_cam_names):
        tx, ty, tz = [float(x) for x in opt_res.x[idx * 3 : (idx + 1) * 3]]
        c_opt = base_clones[cname].clone()
        _apply_world_rotation(c_opt, tx, ty, tz)
        opt_clones[cname] = c_opt

        R_nom = base_clones[cname].rotation_matrix
        R_delta_w = Rotation.from_rotvec(np.deg2rad(np.array([tx, ty, tz], dtype=np.float64))).as_matrix()
        R_delta_cam = R_nom.T @ R_delta_w @ R_nom
        d_p, d_y, d_r = [float(a) for a in Rotation.from_matrix(R_delta_cam).as_euler("xyz", degrees=True)]
        corrections[cname] = (d_y, d_p, d_r)
        max_corr = max(max_corr, abs(tx), abs(ty), abs(tz))

    rms_after, pix_after = _evaluate_reprojection_rms(opt_clones, valid_points)
    pair_errs = (
        np.sqrt(pix_after[0::2] ** 2 + pix_after[1::2] ** 2)
        if len(pix_after) >= 2
        else np.array([], dtype=np.float64)
    )
    max_obs_err_px = float(np.max(pair_errs)) if len(pair_errs) > 0 else 0.0
    return (
        opt_res,
        opt_clones,
        rms_before,
        rms_after,
        pix_after,
        max_obs_err_px,
        corrections,
        max_corr,
        prior_weights,
    )


def run_constrained_bundle_adjustment_with_diagnostics(
    cameras: Dict[str, CameraModel],
    observations: List[TiePointObservation],
    max_angular_correction_deg: float = 6.0,
) -> BundleAdjustmentResult:
    """
    Raffina gli orientamenti delle camere mantenendo i centri ottici C vincolati alle coordinate RTK/PPK.
    Opera nello spazio rotazionale mondo ENU separando i micro-tilt gravitazionali (Est/Nord, prior rigido)
    dal bias di azimuth/bussola (+Z, prior morbido fino a ±max_angular_correction_deg), con convergenza
    a due stadi (soft_l1 coarse-to-fine + cauchy) e filtro automatico di eventuali clic anomali (outlier).
    """
    # Raggruppa osservazioni per tie point garantendo una sola osservazione per camera
    points_obs: Dict[str, Dict[str, TiePointObservation]] = {}
    for obs in observations:
        cam = cameras.get(obs.camera_name)
        if cam is None or not cam.has_intrinsics:
            continue
        points_obs.setdefault(obs.point_id, {})[obs.camera_name] = obs

    valid_points: Dict[str, List[TiePointObservation]] = {
        pid: list(cam_map.values())
        for pid, cam_map in points_obs.items()
        if len(cam_map) >= 2
    }

    zero_corr = {cname: (0.0, 0.0, 0.0) for cname in cameras}
    total_obs = sum(len(v) for v in valid_points.values())

    if len(valid_points) < 2:
        return BundleAdjustmentResult(
            applied=False,
            converged=False,
            num_tie_points=len(valid_points),
            num_observations=total_obs,
            rms_before_px=None,
            rms_after_px=None,
            max_correction_deg=0.0,
            corrections=zero_corr,
            notes="Tie-point multi-vista insufficienti (< 2 punti visti su almeno 2 foto): mantenuto assetto nominale.",
        )

    # Individua le camere effettivamente coinvolte nei tie point validi
    active_cam_names = sorted(
        {obs.camera_name for obs_list in valid_points.values() for obs in obs_list}
    )
    if len(active_cam_names) < 2:
        return BundleAdjustmentResult(
            applied=False,
            converged=False,
            num_tie_points=len(valid_points),
            num_observations=total_obs,
            rms_before_px=None,
            rms_after_px=None,
            max_correction_deg=0.0,
            corrections=zero_corr,
            notes="Meno di 2 camere condividono tie-point validi.",
        )

    base_clones = {cname: cameras[cname].clone() for cname in active_cam_names}
    (
        opt_res,
        opt_clones,
        rms_before,
        rms_after,
        pix_after,
        max_obs_err_px,
        cam_corrections,
        max_corr,
        prior_weights,
    ) = _solve_ba_core(base_clones, active_cam_names, valid_points, max_angular_correction_deg)

    excluded_outliers: List[str] = []

    # Se il residuo post-BA è anomalo (es. > 6 px RMS o singolo punto > 14 px), cerca eventuali clic errati (outlier)
    if rms_after is not None and (rms_after > 6.0 or max_obs_err_px > 14.0):
        # Passo A: pre-filtro geometrico rapido sui TP con >= 3 viste se più TP condividono lo stesso clic errato
        pre_filtered = {pid: list(obs_l) for pid, obs_l in valid_points.items()}
        pre_excl: List[str] = []
        for pid, obs_list in list(pre_filtered.items()):
            if len(obs_list) < 3:
                continue
            rays_full = [base_clones[o.camera_name].pixel_to_ray(o.u_pix, o.v_pix) for o in obs_list]
            t_full = triangulate_point_rays(rays_full)
            best_sub_res = None
            best_sub_idx = None
            for idx_o in range(len(obs_list)):
                sub_rays = rays_full[:idx_o] + rays_full[idx_o + 1 :]
                t_sub = triangulate_point_rays(sub_rays)
                if best_sub_res is None or t_sub.max_residual_m < best_sub_res.max_residual_m:
                    best_sub_res = t_sub
                    best_sub_idx = idx_o
            if (
                best_sub_res is not None
                and best_sub_idx is not None
                and t_full.max_residual_m > 1.10
                and best_sub_res.max_residual_m < 0.35
                and t_full.max_residual_m > 3.5 * max(best_sub_res.max_residual_m, 0.02)
            ):
                bad_obs = obs_list[best_sub_idx]
                pre_filtered[pid] = obs_list[:best_sub_idx] + obs_list[best_sub_idx + 1 :]
                pre_excl.append(f"{pid}@{bad_obs.camera_name}")

        if pre_excl:
            cand_cams = sorted({o.camera_name for ol in pre_filtered.values() for o in ol})
            if len(cand_cams) >= 2:
                cand_res = _solve_ba_core(
                    {cn: base_clones[cn] for cn in cand_cams},
                    cand_cams,
                    pre_filtered,
                    max_angular_correction_deg,
                )
                if cand_res[3] is not None and cand_res[3] < rms_after * 0.65:
                    valid_points = pre_filtered
                    active_cam_names = cand_cams
                    excluded_outliers.extend(pre_excl)
                    (
                        opt_res,
                        opt_clones,
                        rms_before,
                        rms_after,
                        pix_after,
                        max_obs_err_px,
                        cam_corrections,
                        max_corr,
                        prior_weights,
                    ) = cand_res

        # Passo B: ricerca leave-one-out fino a 2 iterazioni per eventuali outlier residui
        for _ in range(2):
            if rms_after is None or (rms_after <= 6.0 and max_obs_err_px <= 14.0):
                break
            best_cand_tuple = None
            best_cand_vp = None
            best_cand_cams = None
            best_cand_label = None

            for pid, obs_list in valid_points.items():
                if len(obs_list) >= 3:
                    for idx_o, bad_obs in enumerate(obs_list):
                        cand_vp = dict(valid_points)
                        cand_vp[pid] = obs_list[:idx_o] + obs_list[idx_o + 1 :]
                        cand_cams = sorted({o.camera_name for ol in cand_vp.values() for o in ol})
                        if len(cand_cams) < 2:
                            continue
                        c_res = _solve_ba_core(
                            {cn: base_clones[cn] for cn in cand_cams},
                            cand_cams,
                            cand_vp,
                            max_angular_correction_deg,
                        )
                        c_rms = c_res[3]
                        if c_rms is not None and (best_cand_tuple is None or c_rms < best_cand_tuple[3]):
                            best_cand_tuple = c_res
                            best_cand_vp = cand_vp
                            best_cand_cams = cand_cams
                            best_cand_label = f"{pid}@{bad_obs.camera_name}"
                elif len(obs_list) == 2 and len(valid_points) >= 3:
                    cand_vp = {k: v for k, v in valid_points.items() if k != pid}
                    cand_cams = sorted({o.camera_name for ol in cand_vp.values() for o in ol})
                    if len(cand_cams) < 2:
                        continue
                    c_res = _solve_ba_core(
                        {cn: base_clones[cn] for cn in cand_cams},
                        cand_cams,
                        cand_vp,
                        max_angular_correction_deg,
                    )
                    c_rms = c_res[3]
                    if c_rms is not None and (best_cand_tuple is None or c_rms < best_cand_tuple[3]):
                        best_cand_tuple = c_res
                        best_cand_vp = cand_vp
                        best_cand_cams = cand_cams
                        best_cand_label = f"{pid} (2 viste incoerenti)"

            if (
                best_cand_tuple is not None
                and best_cand_tuple[3] is not None
                and best_cand_tuple[3] < rms_after * 0.65
                and best_cand_vp is not None
                and best_cand_cams is not None
                and best_cand_label is not None
            ):
                valid_points = best_cand_vp
                active_cam_names = best_cand_cams
                excluded_outliers.append(best_cand_label)
                (
                    opt_res,
                    opt_clones,
                    rms_before,
                    rms_after,
                    pix_after,
                    max_obs_err_px,
                    cam_corrections,
                    max_corr,
                    prior_weights,
                ) = best_cand_tuple
            else:
                break

    total_obs = sum(len(v) for v in valid_points.values())
    num_cams = len(active_cam_names)
    corrections = dict(zero_corr)
    for cname, corr_val in cam_corrections.items():
        corrections[cname] = corr_val

    # Condizione di accettazione: convergenza solver, miglioramento/tenuta RMS e gate qualità assoluta (<= 25 px)
    improved = (
        rms_before is not None
        and rms_after is not None
        and (rms_after <= rms_before + 0.05)
        and (rms_after <= 25.0)
    )
    if not opt_res.success or not improved:
        return BundleAdjustmentResult(
            applied=False,
            converged=bool(opt_res.success),
            num_tie_points=len(valid_points),
            num_observations=total_obs,
            rms_before_px=rms_before,
            rms_after_px=rms_after,
            max_correction_deg=0.0,
            corrections=zero_corr,
            excluded_observations=excluded_outliers,
            notes=(
                f"Bundle Adjustment scartato per incoerenza dei Tie-Point (converged={opt_res.success}, "
                f"RMS prima={rms_before:.2f} px, dopo={rms_after:.2f} px): mantenuto assetto nominale."
                if rms_before is not None and rms_after is not None
                else "Bundle Adjustment non convergente."
            ),
        )

    # Stima incertezza angolare residua post-BA tramite covarianza approssimata dello Jacobiano
    post_std_map: Dict[str, float] = {}
    try:
        n_pix = len(pix_after)
        J_pix = opt_res.jac[:n_pix, :]
        dof = max(1, n_pix - num_cams * 3)
        sigma2_px = float(np.sum(pix_after ** 2) / dof)
        sigma2_px = max(sigma2_px, 0.25)  # floor almeno 0.5 px per realismo metrologico
        H = J_pix.T @ J_pix + np.diag(prior_weights ** 2)
        cov = sigma2_px * np.linalg.pinv(H)
        for idx, cname in enumerate(active_cam_names):
            diag = np.diag(cov[idx * 3 : (idx + 1) * 3, idx * 3 : (idx + 1) * 3])
            ang_std = float(np.sqrt(np.max(np.maximum(diag, 0.0))))
            post_std_map[cname] = float(np.clip(ang_std, 0.05, 0.50))
    except Exception:
        for cname in active_cam_names:
            post_std_map[cname] = 0.12

    # Applica le correzioni alle camere reali solo dopo tutte le verifiche
    for cname in active_cam_names:
        d_y, d_p, d_r = corrections[cname]
        cameras[cname].set_rotation_from_delta(d_y, d_p, d_r)
        cameras[cname].ba_applied = True
        cameras[cname].post_ba_attitude_std_deg = post_std_map.get(cname)

    visible_excl = [e for e in excluded_outliers if not e.startswith("__BLDG_")]
    excl_note = (
        f" [Escluso automaticamente clic anomalo: {', '.join(visible_excl)}]"
        if visible_excl
        else (
            " [Escluso automaticamente 1 clic anomalo edificio dal vincolo BA]"
            if excluded_outliers
            else ""
        )
    )
    tp_counts: Dict[str, int] = {cname: 0 for cname in cameras}
    for obs_list in valid_points.values():
        for obs in obs_list:
            if obs.camera_name in tp_counts:
                tp_counts[obs.camera_name] += 1

    return BundleAdjustmentResult(
        applied=True,
        converged=True,
        num_tie_points=len(valid_points),
        num_observations=total_obs,
        rms_before_px=rms_before,
        rms_after_px=rms_after,
        max_correction_deg=max_corr,
        post_ba_attitude_std_deg=post_std_map,
        corrections=corrections,
        excluded_observations=excluded_outliers,
        tie_point_counts=tp_counts,
        notes=(
            f"Bundle Adjustment applicato su {len(valid_points)} tie-point ({len(active_cam_names)} foto): "
            f"RMS {rms_before:.2f} -> {rms_after:.2f} px (max corr {max_corr:.2f}°).{excl_note}"
        ),
    )


def refine_unadjusted_cameras_with_conductor(
    cameras: Dict[str, CameraModel],
    conductor_obs: List[Tuple[CameraModel, List[Tuple[float, float]]]],
    reference_conductor: Any,
    ba_result: Optional[BundleAdjustmentResult] = None,
    max_tilt_deg: float = 2.5,
    max_azimuth_deg: float = 5.0,
    anchor_points_3d: Optional[Dict[str, List[Tuple[np.ndarray, Tuple[float, float]]]]] = None,
    target_camera_names: Optional[Set[str]] = None,
) -> List[str]:
    """
    Per le camere attive prive di vincolo BA completo (0 Tie-Point come scatti orizzontali con solo
    cavo, oppure 1 solo punto condiviso che lascia libero il roll attorno al raggio ottico), allinea
    l'orientamento sul conduttore 3D già ricostruito dalle camere pienamente calibrate (>= 2 TP),
    mantenendo vincolati eventuali punti ancora 3D condivisi (es. edificio).
    """
    if reference_conductor is None:
        return []

    curve_3d = reference_conductor.sample_points(400)
    aligned_names: List[str] = []

    for cam, pts_2d in conductor_obs:
        if target_camera_names is not None:
            if cam.name not in target_camera_names:
                continue
        elif getattr(cam, "ba_applied", False):
            continue
        if not cam.has_intrinsics or len(pts_2d) < 2:
            continue

        base_clone = cam.clone()
        obs_uv = np.asarray(pts_2d, dtype=np.float64)
        cam_anchors = (anchor_points_3d or {}).get(cam.name) or []

        def _curve_pix_residuals(c_test: CameraModel) -> np.ndarray:
            proj_uv = []
            for p3 in curve_3d:
                pr = c_test.project_point(p3)
                if pr is not None and pr[2] > 0.3:
                    proj_uv.append((pr[0], pr[1]))
            if len(proj_uv) < 2:
                return np.full(len(obs_uv) * 2, 500.0, dtype=np.float64)
            arr_c = np.asarray(proj_uv, dtype=np.float64)
            seg_a = arr_c[:-1]
            seg_b = arr_c[1:]
            ab = seg_b - seg_a
            ab_sq = np.sum(ab * ab, axis=1)
            res_list: List[float] = []
            for u_o, v_o in obs_uv:
                p_pt = np.array([u_o, v_o], dtype=np.float64)
                ap = p_pt[None, :] - seg_a
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_seg = np.where(ab_sq > 1e-12, np.sum(ap * ab, axis=1) / ab_sq, 0.0)
                t_seg = np.clip(t_seg, 0.0, 1.0)
                proj_seg = seg_a + t_seg[:, None] * ab
                diffs = proj_seg - p_pt[None, :]
                dist2 = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
                idx_min = int(np.argmin(dist2))
                res_list.append(float(diffs[idx_min, 0]))
                res_list.append(float(diffs[idx_min, 1]))
            return np.asarray(res_list, dtype=np.float64)

        def _anchor_pix_residuals(c_test: CameraModel) -> np.ndarray:
            if not cam_anchors:
                return np.array([], dtype=np.float64)
            out_a: List[float] = []
            for p3_a, (u_a, v_a) in cam_anchors:
                pr = c_test.project_point(p3_a)
                if pr is not None and pr[2] > 0.3:
                    out_a.append(float(pr[0] - u_a))
                    out_a.append(float(pr[1] - v_a))
                else:
                    out_a.extend([500.0, 500.0])
            return np.asarray(out_a, dtype=np.float64)

        pix_init = _curve_pix_residuals(base_clone)
        rms_init = float(np.sqrt(np.mean(pix_init[0::2] ** 2 + pix_init[1::2] ** 2)))
        if rms_init <= 1.0 and not cam_anchors:
            cam.ba_applied = True
            cam.post_ba_attitude_std_deg = 0.20
            aligned_names.append(cam.name)
            continue

        prior_w = np.array([2.5, 2.5, 0.45], dtype=np.float64)
        b_high = np.array([float(max_tilt_deg), float(max_tilt_deg), float(max_azimuth_deg)], dtype=np.float64)
        anchor_weight = max(2.5, float(np.sqrt(len(obs_uv))))

        c_try = base_clone.clone()

        def _update_c_try(tx: float, ty: float, tz: float):
            rot_vec = np.deg2rad(np.array([tx, ty, tz], dtype=np.float64))
            R_delta_w = Rotation.from_rotvec(rot_vec).as_matrix()
            R_new = R_delta_w @ base_clone.rotation_matrix
            c_try.right = R_new[:, 0] / np.linalg.norm(R_new[:, 0])
            c_try.up = -R_new[:, 1] / np.linalg.norm(R_new[:, 1])
            c_try.front = R_new[:, 2] / np.linalg.norm(R_new[:, 2])

        def _res_fn(p: np.ndarray) -> np.ndarray:
            _update_c_try(float(p[0]), float(p[1]), float(p[2]))
            parts = [_curve_pix_residuals(c_try)]
            if cam_anchors:
                parts.append(_anchor_pix_residuals(c_try) * anchor_weight)
            parts.append(prior_w * p)
            return np.concatenate(parts)

        opt = least_squares(
            _res_fn,
            np.zeros(3, dtype=np.float64),
            bounds=(-b_high, b_high),
            loss="soft_l1",
            f_scale=2.5,
            max_nfev=150,
        )
        tx, ty, tz = [float(x) for x in opt.x]
        c_final = base_clone.clone()
        _apply_world_rotation(c_final, tx, ty, tz)
        pix_final = _curve_pix_residuals(c_final)
        rms_final = float(np.sqrt(np.mean(pix_final[0::2] ** 2 + pix_final[1::2] ** 2)))

        anchors_ok = True
        if cam_anchors:
            a_final = _anchor_pix_residuals(c_final)
            a_errs = np.sqrt(a_final[0::2] ** 2 + a_final[1::2] ** 2)
            anchors_ok = bool(np.max(a_errs) <= 10.0)

        if opt.success and anchors_ok and rms_final < rms_init and rms_final <= 8.0:
            R_nom = base_clone.rotation_matrix
            R_delta_w = Rotation.from_rotvec(np.deg2rad(np.array([tx, ty, tz], dtype=np.float64))).as_matrix()
            R_delta_cam = R_nom.T @ R_delta_w @ R_nom
            d_p, d_y, d_r = [float(a) for a in Rotation.from_matrix(R_delta_cam).as_euler("xyz", degrees=True)]

            cam.set_rotation_from_delta(d_y, d_p, d_r)
            cam.ba_applied = True
            new_std = 0.16 if cam_anchors else 0.22
            cam.post_ba_attitude_std_deg = new_std
            aligned_names.append(cam.name)

            if ba_result is not None:
                prev_corr = ba_result.corrections.get(cam.name, (0.0, 0.0, 0.0))
                ba_result.corrections[cam.name] = (
                    prev_corr[0] + d_y,
                    prev_corr[1] + d_p,
                    prev_corr[2] + d_r,
                )
                ba_result.post_ba_attitude_std_deg[cam.name] = new_std
                ba_result.max_correction_deg = max(ba_result.max_correction_deg, abs(tx), abs(ty), abs(tz))
                if cam.name not in ba_result.conductor_aligned_cameras:
                    ba_result.conductor_aligned_cameras.append(cam.name)

    if aligned_names and ba_result is not None:
        ba_result.notes += (
            f" Allineamento assistito su conduttore 3D per {len(aligned_names)} foto senza vincolo TP completo "
            f"({', '.join(aligned_names)})."
        )
    return aligned_names


def run_constrained_bundle_adjustment(
    cameras: Dict[str, CameraModel],
    observations: List[TiePointObservation],
    max_angular_correction_deg: float = 6.0,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Wrapper compatibile con l'API precedente: restituisce {camera_name: (d_yaw_deg, d_pitch_deg, d_roll_deg)}.
    """
    res = run_constrained_bundle_adjustment_with_diagnostics(
        cameras=cameras,
        observations=observations,
        max_angular_correction_deg=max_angular_correction_deg,
    )
    return res.corrections
