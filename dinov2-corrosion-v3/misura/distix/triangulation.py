"""
DISTIX - Triangulation Engine
Triangolazione 3D di punti mediante intersezione ai minimi quadrati di fasci di raggi multi-vista.
Include:
- Controllo cheirality (punto rigorosamente davanti a tutte le fotocamere)
- Controllo residui 3D e riproiezione 2D in pixel dentro il verdetto di affidabilità
- Calcolo matrice di covarianza 3D anisotropa (per propagazione direzionale GUM)
- Calcolo guide epipolari tra viste (supporta anche distorsione attiva DewarpFlag=0)
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class TriangulationResult:
    def __init__(
        self,
        point_3d: np.ndarray,
        residuals_m: np.ndarray,
        reprojection_errors_px: np.ndarray,
        max_convergence_angle_deg: float,
        condition_number: float,
        is_reliable: bool,
        notes: str = "",
        depths_m: Optional[np.ndarray] = None,
        all_in_front: bool = True,
        covariance_3d: Optional[np.ndarray] = None,
        excluded_cameras: Optional[List[str]] = None,
    ):
        self.point_3d = np.asarray(point_3d, dtype=np.float64)
        self.residuals_m = np.asarray(residuals_m, dtype=np.float64)
        self.mean_residual_m = float(np.mean(self.residuals_m)) if len(self.residuals_m) > 0 else 0.0
        self.max_residual_m = float(np.max(self.residuals_m)) if len(self.residuals_m) > 0 else 0.0
        self.reprojection_errors_px = np.asarray(reprojection_errors_px, dtype=np.float64)
        self.mean_reproj_px = float(np.mean(self.reprojection_errors_px)) if len(self.reprojection_errors_px) > 0 else 0.0
        self.max_reproj_px = float(np.max(self.reprojection_errors_px)) if len(self.reprojection_errors_px) > 0 else 0.0
        self.max_convergence_angle_deg = float(max_convergence_angle_deg)
        self.condition_number = float(condition_number)
        self.depths_m = np.asarray(depths_m, dtype=np.float64) if depths_m is not None else np.array([], dtype=np.float64)
        self.all_in_front = bool(all_in_front)
        self.covariance_3d = covariance_3d
        self.is_reliable = bool(is_reliable)
        self.notes = notes
        self.excluded_cameras: List[str] = list(excluded_cameras or [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_3d": self.point_3d.tolist(),
            "mean_residual_m": round(self.mean_residual_m, 4),
            "max_residual_m": round(self.max_residual_m, 4),
            "mean_reproj_px": round(self.mean_reproj_px, 2),
            "max_reproj_px": round(self.max_reproj_px, 2),
            "convergence_angle_deg": round(self.max_convergence_angle_deg, 2),
            "condition_number": round(self.condition_number, 2),
            "all_in_front": self.all_in_front,
            "is_reliable": self.is_reliable,
            "excluded_cameras": list(self.excluded_cameras),
            "notes": self.notes,
        }


def triangulate_point_rays(
    rays: List[Tuple[np.ndarray, np.ndarray]],  # Lista di (C_i, d_i)
    min_convergence_deg: float = 15.0,
    max_acceptable_residual_m: float = 0.25,
    min_depth_m: float = 0.2,
) -> TriangulationResult:
    """
    Triangola un punto 3D da N >= 2 raggi ottici nello spazio.
    Trova P* = (sum(I - d_i d_i^T))^-1 * sum((I - d_i d_i^T) C_i)
    e verifica convergenza, condizionamento, cheirality e residui.
    """
    if len(rays) < 2:
        raise ValueError("Sono necessari almeno 2 raggi per triangolare un punto 3D.")

    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    dirs: List[np.ndarray] = []
    centers: List[np.ndarray] = []

    for C_i, d_i in rays:
        C = np.asarray(C_i, dtype=np.float64)
        d = np.asarray(d_i, dtype=np.float64)
        norm_d = np.linalg.norm(d)
        if norm_d < 1e-12:
            raise ValueError("Direzione di raggio ottico nulla.")
        d = d / norm_d

        P_ortho = np.eye(3) - np.outer(d, d)
        A += P_ortho
        b += P_ortho @ C

        dirs.append(d)
        centers.append(C)

    eigenvals = np.linalg.eigvalsh(A)
    cond = float(eigenvals[-1] / max(eigenvals[0], 1e-12))

    try:
        P_star = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        P_star = np.linalg.pinv(A) @ b

    # Residui 3D e cheirality
    residuals_m = []
    depths_m = []
    for C, d in zip(centers, dirs):
        delta = P_star - C
        depth = float(np.dot(delta, d))
        depths_m.append(depth)
        ortho_dist = float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ delta))
        residuals_m.append(ortho_dist)

    residuals_arr = np.asarray(residuals_m, dtype=np.float64)
    depths_arr = np.asarray(depths_m, dtype=np.float64)
    all_in_front = bool(np.all(depths_arr > min_depth_m))

    # Angolo massimo di convergenza
    max_angle_deg = 0.0
    num_rays = len(dirs)
    for i in range(num_rays):
        for j in range(i + 1, num_rays):
            cos_theta = float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0))
            angle = float(np.rad2deg(np.arccos(cos_theta)))
            if angle > max_angle_deg:
                max_angle_deg = angle

    is_reliable = True
    notes: List[str] = []
    if not all_in_front:
        is_reliable = False
        notes.append(
            f"Cheirality violata: punto dietro o sul centro ottico di una camera (depth min={np.min(depths_arr):.2f} m)."
        )
    if max_angle_deg < min_convergence_deg:
        is_reliable = False
        notes.append(f"Angolo di convergenza scarso ({max_angle_deg:.1f}° < {min_convergence_deg}°).")
    if cond > 100.0:
        is_reliable = False
        notes.append(f"Geometria mal condizionata (cond={cond:.1f}).")
    if float(np.max(residuals_arr)) > max_acceptable_residual_m:
        is_reliable = False
        notes.append(
            f"Residuo massimo elevato ({np.max(residuals_arr)*100:.1f} cm > {max_acceptable_residual_m*100:.1f} cm)."
        )

    if not notes:
        notes.append("Triangolazione geometricamente consistente.")

    # Covarianza di prima approssimazione se chiamata su raggi nudi (aggiornata in triangulate_from_cameras)
    try:
        cov_raw = np.linalg.pinv(A) * max((float(np.mean(residuals_arr)) ** 2), 1e-4)
    except Exception:
        cov_raw = np.eye(3, dtype=np.float64) * 1e-2

    return TriangulationResult(
        point_3d=P_star,
        residuals_m=residuals_arr,
        reprojection_errors_px=np.array([], dtype=np.float64),
        max_convergence_angle_deg=max_angle_deg,
        condition_number=cond,
        is_reliable=is_reliable,
        notes="; ".join(notes),
        depths_m=depths_arr,
        all_in_front=all_in_front,
        covariance_3d=cov_raw,
    )


def triangulate_from_cameras(
    observations: List[Tuple[Any, Tuple[float, float]]],  # [(camera_obj, (u_pix, v_pix)), ...]
    min_convergence_deg: float = 15.0,
    max_acceptable_residual_m: float = 0.25,
    max_acceptable_reproj_px: float = 10.0,
    sigma_click_px: float = 1.5,
    allow_consensus_filter: bool = True,
) -> TriangulationResult:
    """
    Data una lista di coppie (CameraModel, (u_pix, v_pix)), genera i raggi 3D,
    calcola il punto 3D, gli errori di riproiezione in pixel, la cheirality
    e la matrice di covarianza 3D pesata sulle reali camere.
    Se N >= 3 viste e una singola vista presenta un clic errato (outlier) mentre
    le restanti >= 2 viste convergono con alta affidabilità, filtra automaticamente
    la vista anomala per consenso geometrico.
    """
    rays = []
    for cam, (u, v) in observations:
        C, d = cam.pixel_to_ray(u, v)
        rays.append((C, d))

    res = triangulate_point_rays(
        rays,
        min_convergence_deg=min_convergence_deg,
        max_acceptable_residual_m=max_acceptable_residual_m,
    )

    reproj_errors = []
    cam_depths = []
    for cam, (u_obs, v_obs) in observations:
        proj = cam.project_point(res.point_3d)
        if proj is not None:
            u_p, v_p, depth = proj
            err_px = float(np.sqrt((u_p - u_obs) ** 2 + (v_p - v_obs) ** 2))
            reproj_errors.append(err_px)
            cam_depths.append(depth)
        else:
            reproj_errors.append(999.0)
            res.all_in_front = False

    res.reprojection_errors_px = np.asarray(reproj_errors, dtype=np.float64)
    res.mean_reproj_px = float(np.mean(reproj_errors)) if reproj_errors else 0.0
    res.max_reproj_px = float(np.max(reproj_errors)) if reproj_errors else 0.0

    # Controllo coerenza riproiezione e cheirality dentro is_reliable
    extra_notes: List[str] = []
    if not res.all_in_front:
        res.is_reliable = False
        extra_notes.append("Punto non davanti a tutte le camere nella proiezione ottica.")
    if res.max_reproj_px > max_acceptable_reproj_px:
        res.is_reliable = False
        extra_notes.append(
            f"Errore riproiezione elevato ({res.max_reproj_px:.1f} px > {max_acceptable_reproj_px:.1f} px): "
            "possibile corrispondenza errata o errore pose."
        )
    if extra_notes:
        base_note = "" if res.notes == "Triangolazione geometricamente consistente." else (res.notes + "; ")
        res.notes = base_note + "; ".join(extra_notes)

    # Filtro automatico outlier per consenso multi-vista (quando N >= 3 foto osservano il punto)
    if allow_consensus_filter and len(observations) >= 3 and not res.is_reliable:
        best_sub: Optional[TriangulationResult] = None
        best_excl_idx: Optional[int] = None
        for idx_excl in range(len(observations)):
            sub_obs = observations[:idx_excl] + observations[idx_excl + 1 :]
            sub_res = triangulate_from_cameras(
                sub_obs,
                min_convergence_deg=min_convergence_deg,
                max_acceptable_residual_m=max_acceptable_residual_m,
                max_acceptable_reproj_px=max_acceptable_reproj_px,
                sigma_click_px=sigma_click_px,
                allow_consensus_filter=False,
            )
            if (
                sub_res.is_reliable
                and sub_res.all_in_front
                and sub_res.max_residual_m <= min(max_acceptable_residual_m, max(0.05, res.max_residual_m * 0.35))
            ):
                if best_sub is None or (sub_res.max_residual_m, sub_res.max_reproj_px) < (
                    best_sub.max_residual_m,
                    best_sub.max_reproj_px,
                ):
                    best_sub = sub_res
                    best_excl_idx = idx_excl

        if best_sub is not None and best_excl_idx is not None:
            excl_cam, (u_ex, v_ex) = observations[best_excl_idx]
            excl_name = str(getattr(excl_cam, "name", f"cam_{best_excl_idx + 1}"))
            proj_ex = excl_cam.project_point(best_sub.point_3d)
            if proj_ex is not None:
                err_ex_px = float(np.hypot(proj_ex[0] - u_ex, proj_ex[1] - v_ex))
            else:
                err_ex_px = float(res.max_reproj_px)
            best_sub.excluded_cameras = [excl_name]
            best_sub.notes = (
                f"Escluso automaticamente 1 clic anomalo su {excl_name} (scarto {err_ex_px:.1f} px); "
                f"triangolazione validata per consenso su {len(observations) - 1} viste "
                f"(res max {best_sub.max_residual_m * 100:.1f} cm, {best_sub.max_reproj_px:.1f} px)."
            )
            return best_sub

    # Calcolo covarianza 3D anisotropa pesata (propagando click, assetto reale/BA e GNSS std)
    try:
        A_weighted = np.zeros((3, 3), dtype=np.float64)
        chi2_terms = []
        for (cam, _), (_, d), res_m in zip(observations, rays, res.residuals_m):
            dist = float(np.linalg.norm(res.point_3d - cam.pos))
            f_pix = float(np.mean([cam.fx_pix, cam.fy_pix])) if cam.fx_pix and cam.fy_pix else 3500.0
            sigma_ang_pix = sigma_click_px / f_pix

            # Incertezza angolare camera: post-BA se eseguito, altrimenti IMU nominale conservativo (0.4 deg)
            if getattr(cam, "ba_applied", False) and getattr(cam, "post_ba_attitude_std_deg", None):
                sigma_att_rad = np.deg2rad(float(cam.post_ba_attitude_std_deg))
            else:
                sigma_att_rad = np.deg2rad(0.40)

            gnss_q = getattr(cam, "gnss_quality", None)
            if gnss_q is not None and gnss_q.max_std_m is not None:
                sigma_gnss = float(gnss_q.max_std_m)
            elif getattr(cam, "rtk_std", None) is not None:
                sigma_gnss = float(cam.rtk_std)
            else:
                sigma_gnss = 0.05 if getattr(cam, "rtk_fix", False) else 0.50

            sigma_perp2 = (dist * sigma_ang_pix) ** 2 + (dist * sigma_att_rad) ** 2 + sigma_gnss ** 2
            sigma_perp2 = max(sigma_perp2, 1e-6)
            w = 1.0 / sigma_perp2
            d_u = d / np.linalg.norm(d)
            A_weighted += w * (np.eye(3) - np.outer(d_u, d_u))
            chi2_terms.append((res_m ** 2) * w)

        cov_prior = np.linalg.pinv(A_weighted)
        # Se i residui osservati eccedono la stima a priori, scala con fattore di consistenza
        dof = max(1, 2 * len(observations) - 3)
        s0_sq = max(1.0, float(np.sum(chi2_terms) / dof))
        res.covariance_3d = cov_prior * s0_sq
    except Exception:
        pass

    return res


def compute_epipolar_curve_points(
    cam_src: Any,
    uv_src: Tuple[float, float],
    cam_dst: Any,
    depth_min_m: float = 3.0,
    depth_max_m: float = 150.0,
    num_steps: int = 24,
) -> List[Tuple[float, float]]:
    """
    Calcola la curva/retta epipolare sul sensore di cam_dst corrispondente al raggio di (u, v) su cam_src.
    Supporta anche distorsioni ottiche attive (DewarpFlag=0).
    """
    if not cam_src.has_intrinsics or not cam_dst.has_intrinsics:
        return []

    C_src, d_src = cam_src.pixel_to_ray(uv_src[0], uv_src[1])
    depths = np.geomspace(max(0.5, depth_min_m), max(depth_min_m + 1.0, depth_max_m), num_steps)
    pts_dst: List[Tuple[float, float]] = []

    for depth in depths:
        P_world = C_src + float(depth) * d_src
        proj = cam_dst.project_point(P_world)
        if proj is not None:
            u_d, v_d, _ = proj
            if cam_dst.is_pixel_in_bounds(u_d, v_d, margin=50.0):
                pts_dst.append((float(u_d), float(v_d)))

    return pts_dst
