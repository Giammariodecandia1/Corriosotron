"""
DISTIX - Core Camera Model
Modello fotogrammetrico di camera prospettica calibrata (Pinhole + Brown-Conrady opzionale).
Nessun parametro intrinseco o GNSS predefinito viene assunto silenziosamente.
Supporta:
- Distorsione radiale/tangenziale DJI (DewarpFlag=0 con k1,k2,p1,p2,k3)
- Immagini già rettificate in camera (DewarpFlag=1, zero doppia correzione)
- Tracciabilità posa nominale pre-Bundle Adjustment e clonazione sicura
"""

from typing import Tuple, Optional, Dict, Any
import copy
import numpy as np


def _apply_brown_conrady_distortion(
    x_n: float,
    y_n: float,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
    k3: float,
) -> Tuple[float, float]:
    """
    Applica il modello diretto di distorsione Brown-Conrady (OpenCV / DJI DewarpData).
    Per punti molto fuori dal cerchio d'immagine del sensore (r^2 > 1.25), limita il
    polinomio di Taylor al bordo del cono ottico evitando l'inversione di segno di k3*r^6
    che ripiegherebbe falsamente punti fuori campo dentro il frame.
    """
    r2 = x_n * x_n + y_n * y_n
    r2_eff = min(r2, 1.25)
    r4_eff = r2_eff * r2_eff
    r6_eff = r4_eff * r2_eff
    radial = max(0.25, 1.0 + k1 * r2_eff + k2 * r4_eff + k3 * r6_eff)
    x_eff = x_n if r2 <= 1.25 else x_n * np.sqrt(1.25 / r2)
    y_eff = y_n if r2 <= 1.25 else y_n * np.sqrt(1.25 / r2)
    x_d = x_n * radial + 2.0 * p1 * x_eff * y_eff + p2 * (r2_eff + 2.0 * x_eff * x_eff)
    y_d = y_n * radial + p1 * (r2_eff + 2.0 * y_eff * y_eff) + 2.0 * p2 * x_eff * y_eff
    return float(x_d), float(y_d)


def _invert_brown_conrady_distortion(
    x_d: float,
    y_d: float,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
    k3: float,
    max_iter: int = 20,
    tol: float = 1e-12,
) -> Tuple[float, float]:
    """
    Inverte il modello Brown-Conrady mediante Newton-Raphson analitico con Jacobiano esatto 2x2.
    Dato il punto distorto normalizzato (x_d, y_d), trova il punto ideale (x_n, y_n).
    """
    x_n = float(x_d)
    y_n = float(y_d)

    for _ in range(max_iter):
        r2 = x_n * x_n + y_n * y_n
        r4 = r2 * r2
        r6 = r4 * r2
        dr = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        ddr_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4

        fx = x_n * dr + 2.0 * p1 * x_n * y_n + p2 * (r2 + 2.0 * x_n * x_n) - x_d
        fy = y_n * dr + p1 * (r2 + 2.0 * y_n * y_n) + 2.0 * p2 * x_n * y_n - y_d

        if fx * fx + fy * fy < tol * tol:
            break

        d_dr_dx = 2.0 * x_n * ddr_dr2
        d_dr_dy = 2.0 * y_n * ddr_dr2

        j00 = dr + x_n * d_dr_dx + 2.0 * p1 * y_n + 6.0 * p2 * x_n
        j01 = x_n * d_dr_dy + 2.0 * p1 * x_n + 2.0 * p2 * y_n
        j10 = y_n * d_dr_dx + 2.0 * p1 * x_n + 2.0 * p2 * y_n
        j11 = dr + y_n * d_dr_dy + 6.0 * p1 * y_n + 2.0 * p2 * x_n

        det = j00 * j11 - j01 * j10
        if abs(det) < 1e-15:
            break

        dx = (j11 * fx - j01 * fy) / det
        dy = (-j10 * fx + j00 * fy) / det

        x_n -= dx
        y_n -= dy

        if dx * dx + dy * dy < tol * tol:
            break

    return float(x_n), float(y_n)


class CameraModel:
    def __init__(
        self,
        name: str,
        pos: np.ndarray,  # [X, Y, Z] coordinate metriche mondo (es. ENU locale)
        front: np.ndarray,  # Vettore direzione asse ottico normalizzato [fx, fy, fz]
        up: np.ndarray,     # Vettore Up normalizzato [ux, uy, uz]
        hfov_deg: Optional[float] = None,
        vfov_deg: Optional[float] = None,
        focal_length_mm: Optional[float] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        fx_pix: Optional[float] = None,
        fy_pix: Optional[float] = None,
        cx_pix: Optional[float] = None,
        cy_pix: Optional[float] = None,
        calibration: Optional[Any] = None,
        rtk_fix: bool = False,
        rtk_flag: Optional[int] = None,
        rtk_std: Optional[float] = None,
        rtk_std_hgt: Optional[float] = None,
        gnss_quality: Optional[Any] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.pos = np.asarray(pos, dtype=np.float64).copy()
        if self.pos.shape != (3,) or not np.all(np.isfinite(self.pos)):
            raise ValueError(f"Posizione 3D non valida per la camera {name}: {pos}")

        self.timestamp = float(timestamp) if timestamp is not None else None
        self.gnss_quality = gnss_quality
        self.calibration = calibration
        self.metadata: Dict[str, Any] = copy.deepcopy(metadata) if metadata else {}

        # Stato GNSS effettivo (mai forzato a True)
        if gnss_quality is not None:
            self.rtk_fix = bool(gnss_quality.is_fix)
            self.rtk_flag = gnss_quality.rtk_flag
            self.rtk_std = gnss_quality.max_horiz_std_m if rtk_std is None else float(rtk_std)
            self.rtk_std_hgt = gnss_quality.std_hgt_m if rtk_std_hgt is None else float(rtk_std_hgt)
        else:
            self.rtk_fix = bool(rtk_fix)
            self.rtk_flag = int(rtk_flag) if rtk_flag is not None else None
            self.rtk_std = float(rtk_std) if rtk_std is not None else None
            self.rtk_std_hgt = float(rtk_std_hgt) if rtk_std_hgt is not None else None

        # Normalizzazione e ortonormalizzazione rigorosa vettori di orientamento
        f = np.asarray(front, dtype=np.float64)
        u = np.asarray(up, dtype=np.float64)
        if not (np.all(np.isfinite(f)) and np.all(np.isfinite(u))):
            raise ValueError(f"Vettori front/up contengono NaN/Inf per la camera {name}")
        norm_f = np.linalg.norm(f)
        norm_u = np.linalg.norm(u)
        if norm_f < 1e-9 or norm_u < 1e-9:
            raise ValueError(f"Vettori front/up nulli per la camera {name}")
        self.front = f / norm_f
        self.up = u / norm_u

        # Vettore Right orizzontale della camera: r = f x u
        r = np.cross(self.front, self.up)
        norm_r = np.linalg.norm(r)
        if norm_r < 1e-9:
            raise ValueError(f"I vettori Front e Up sono collineari per la camera {name}!")
        self.right = r / norm_r

        # Ricalcola Up per garantire rigorosa ortogonalità: u = r x f
        self.up = np.cross(self.right, self.front)
        self.up = self.up / np.linalg.norm(self.up)

        # Conservazione posa nominale pre-BA
        self.nominal_pos = self.pos.copy()
        self.nominal_front = self.front.copy()
        self.nominal_up = self.up.copy()
        self.nominal_right = self.right.copy()
        self.ba_applied: bool = False
        self.post_ba_attitude_std_deg: Optional[float] = None

        # Risoluzione parametri intrinseci da calibration o argomenti espliciti (nessun default inventato)
        if calibration is not None:
            w = int(calibration.image_width) if calibration.image_width else image_width
            h = int(calibration.image_height) if calibration.image_height else image_height
            fx_c = calibration.fx_pix if fx_pix is None else fx_pix
            fy_c = calibration.fy_pix if fy_pix is None else fy_pix
            cx_c = calibration.cx_pix if cx_pix is None else cx_pix
            cy_c = calibration.cy_pix if cy_pix is None else cy_pix
            hfov_c = calibration.hfov_deg if hfov_deg is None else hfov_deg
            vfov_c = calibration.vfov_deg if vfov_deg is None else vfov_deg
            fmm_c = calibration.focal_length_mm if focal_length_mm is None else focal_length_mm
        else:
            w, h = image_width, image_height
            fx_c, fy_c, cx_c, cy_c = fx_pix, fy_pix, cx_pix, cy_pix
            hfov_c, vfov_c, fmm_c = hfov_deg, vfov_deg, focal_length_mm

        self.focal_length_mm = float(fmm_c) if fmm_c is not None else None
        self.image_width: Optional[int] = int(w) if w is not None else None
        self.image_height: Optional[int] = int(h) if h is not None else None

        self.fx_pix: Optional[float] = None
        self.fy_pix: Optional[float] = None
        self.cx_pix: Optional[float] = None
        self.cy_pix: Optional[float] = None
        self.hfov_deg: Optional[float] = float(hfov_c) if hfov_c is not None else None
        self.vfov_deg: Optional[float] = float(vfov_c) if vfov_c is not None else None
        self.tan_half_hfov: Optional[float] = None
        self.tan_half_vfov: Optional[float] = None
        self.has_intrinsics: bool = False

        if (
            fx_c is not None and fy_c is not None
            and self.image_width and self.image_height
        ):
            self.fx_pix = float(fx_c)
            self.fy_pix = float(fy_c)
            self.cx_pix = float(cx_c) if cx_c is not None else (self.image_width / 2.0)
            self.cy_pix = float(cy_c) if cy_c is not None else (self.image_height / 2.0)
            if self.fx_pix <= 0 or self.fy_pix <= 0:
                raise ValueError(f"Focale in pixel non valida per {name}: ({self.fx_pix}, {self.fy_pix})")
            self.tan_half_hfov = (self.image_width / 2.0) / self.fx_pix
            self.tan_half_vfov = (self.image_height / 2.0) / self.fy_pix
            self.hfov_deg = float(2.0 * np.rad2deg(np.arctan(self.tan_half_hfov)))
            self.vfov_deg = float(2.0 * np.rad2deg(np.arctan(self.tan_half_vfov)))
            self.has_intrinsics = True
        elif (
            hfov_c is not None and vfov_c is not None
            and self.image_width and self.image_height
        ):
            self.hfov_deg = float(hfov_c)
            self.vfov_deg = float(vfov_c)
            if not (0.0 < self.hfov_deg < 179.0 and 0.0 < self.vfov_deg < 179.0):
                raise ValueError(f"FOV non valido per {name}: HFOV={self.hfov_deg}, VFOV={self.vfov_deg}")
            self.tan_half_hfov = float(np.tan(np.deg2rad(self.hfov_deg / 2.0)))
            self.tan_half_vfov = float(np.tan(np.deg2rad(self.vfov_deg / 2.0)))
            self.fx_pix = (self.image_width / 2.0) / self.tan_half_hfov
            self.fy_pix = (self.image_height / 2.0) / self.tan_half_vfov
            self.cx_pix = float(cx_c) if cx_c is not None else (self.image_width / 2.0)
            self.cy_pix = float(cy_c) if cy_c is not None else (self.image_height / 2.0)
            self.has_intrinsics = True
        elif calibration is not None and self.image_width and self.image_height:
            # Consente il caricamento per ispezione e segnalazione diagnostica, ma blocca il calcolo raggi
            self.has_intrinsics = False
        else:
            raise ValueError(
                f"Camera {name}: parametri ottici intrinseci o dimensioni immagine mancanti. "
                "Distix vieta l'uso di valori predefiniti presunti."
            )

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Matrice di rotazione da coordinate camera (Right, Down, Front)
        a coordinate mondo (X, Y, Z).
        R_c2w = [Right, -Up, Front]
        """
        down = -self.up
        return np.column_stack((self.right, down, self.front))

    def clone(self) -> "CameraModel":
        """Crea una copia indipendente e completa della camera preservando calibrazione e GNSS (fast shallow copy dei descrittori immutabili)."""
        cam = CameraModel(
            name=self.name,
            pos=self.pos.copy(),
            front=self.front.copy(),
            up=self.up.copy(),
            hfov_deg=self.hfov_deg,
            vfov_deg=self.vfov_deg,
            focal_length_mm=self.focal_length_mm,
            image_width=self.image_width,
            image_height=self.image_height,
            fx_pix=self.fx_pix,
            fy_pix=self.fy_pix,
            cx_pix=self.cx_pix,
            cy_pix=self.cy_pix,
            calibration=self.calibration,
            rtk_fix=self.rtk_fix,
            rtk_flag=self.rtk_flag,
            rtk_std=self.rtk_std,
            rtk_std_hgt=self.rtk_std_hgt,
            gnss_quality=self.gnss_quality,
            timestamp=self.timestamp,
            metadata=self.metadata,
        )
        cam.right = self.right.copy()
        cam.nominal_pos = self.nominal_pos.copy()
        cam.nominal_front = self.nominal_front.copy()
        cam.nominal_up = self.nominal_up.copy()
        cam.nominal_right = self.nominal_right.copy()
        cam.ba_applied = self.ba_applied
        cam.post_ba_attitude_std_deg = self.post_ba_attitude_std_deg
        return cam

    def reset_to_nominal_pose(self):
        """Ripristina posizione e orientamento nominali dai metadati originali."""
        self.pos = self.nominal_pos.copy()
        self.front = self.nominal_front.copy()
        self.up = self.nominal_up.copy()
        self.right = self.nominal_right.copy()
        self.ba_applied = False
        self.post_ba_attitude_std_deg = None

    def update_calibration(self, calibration: Any):
        """Aggiorna la calibrazione ottica (es. dopo registrazione/rimozione profilo) mantenendo posa e GNSS."""
        self.calibration = calibration
        if calibration is None:
            return
        if (
            getattr(calibration, "fx_pix", None) is not None
            and getattr(calibration, "fy_pix", None) is not None
            and self.image_width
            and self.image_height
        ):
            self.fx_pix = float(calibration.fx_pix)
            self.fy_pix = float(calibration.fy_pix)
            self.cx_pix = float(calibration.cx_pix) if calibration.cx_pix is not None else (self.image_width / 2.0)
            self.cy_pix = float(calibration.cy_pix) if calibration.cy_pix is not None else (self.image_height / 2.0)
            self.tan_half_hfov = (self.image_width / 2.0) / self.fx_pix
            self.tan_half_vfov = (self.image_height / 2.0) / self.fy_pix
            self.hfov_deg = float(2.0 * np.rad2deg(np.arctan(self.tan_half_hfov)))
            self.vfov_deg = float(2.0 * np.rad2deg(np.arctan(self.tan_half_vfov)))
            self.has_intrinsics = True

    def set_rotation_from_delta(self, d_yaw_deg: float, d_pitch_deg: float, d_roll_deg: float):
        """
        Applica una piccola rotazione correttiva locale (es. da bundle adjustment)
        attorno agli assi della camera (right=pitch, down=yaw, front=roll).
        """
        from scipy.spatial.transform import Rotation as R
        r_delta = R.from_euler('xyz', [d_pitch_deg, d_yaw_deg, d_roll_deg], degrees=True)
        R_curr = self.rotation_matrix
        R_new = R_curr @ r_delta.as_matrix()

        self.right = R_new[:, 0] / np.linalg.norm(R_new[:, 0])
        self.up = -R_new[:, 1] / np.linalg.norm(R_new[:, 1])
        self.front = R_new[:, 2] / np.linalg.norm(R_new[:, 2])

    def _distortion_params_if_active(self) -> Optional[Tuple[float, float, float, float, float]]:
        cal = self.calibration
        if cal is not None and getattr(cal, "distortion_active", False):
            return (
                float(cal.k1),
                float(cal.k2),
                float(cal.p1),
                float(cal.p2),
                float(cal.k3),
            )
        return None

    def pixel_to_ray(self, u_pix: float, v_pix: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dato un pixel (u_pix, v_pix) nell'immagine, restituisce il raggio ottico 3D
        nello spazio mondo:
        - origin: centro ottico della camera C [X, Y, Z]
        - direction: vettore unitario d [dx, dy, dz]
        Se DewarpFlag=0 e la distorsione è attiva, rimuove la distorsione Brown-Conrady.
        Se DewarpFlag=1, applica il modello pinhole puro evitando doppie correzioni.
        """
        if not self.has_intrinsics or self.fx_pix is None or self.fy_pix is None:
            raise ValueError(
                f"Camera {self.name}: parametri ottici intrinseci mancanti, "
                "impossibile calcolare il raggio ottico 3D."
            )

        x_d = (float(u_pix) - self.cx_pix) / self.fx_pix
        y_d = (float(v_pix) - self.cy_pix) / self.fy_pix

        dist_params = self._distortion_params_if_active()
        if dist_params is not None:
            x_c, y_c = _invert_brown_conrady_distortion(x_d, y_d, *dist_params)
        else:
            x_c, y_c = x_d, y_d

        # Direzione nel sistema camera (Right, Down, Front) -> mondo
        ray_world = x_c * self.right - y_c * self.up + 1.0 * self.front
        ray_dir = ray_world / np.linalg.norm(ray_world)
        return self.pos.copy(), ray_dir

    def project_point(self, point_world: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Proietta un punto 3D mondo sul piano immagine della fotocamera.
        Applica la distorsione Brown-Conrady solo se distortion_active == True (DewarpFlag=0).
        Restituisce (u_pix, v_pix, depth) oppure None se il punto è dietro la camera.
        """
        if not self.has_intrinsics or self.fx_pix is None or self.fy_pix is None:
            raise ValueError(
                f"Camera {self.name}: parametri ottici intrinseci mancanti, "
                "impossibile proiettare il punto 3D."
            )

        pt = np.asarray(point_world, dtype=np.float64)
        delta = pt - self.pos

        depth = float(np.dot(delta, self.front))
        if depth <= 1e-4:
            return None  # Punto dietro o complanare al centro ottico

        x_n = float(np.dot(delta, self.right) / depth)
        y_n = float(-np.dot(delta, self.up) / depth)  # y_n positivo verso il basso

        dist_params = self._distortion_params_if_active()
        if dist_params is not None:
            x_img, y_img = _apply_brown_conrady_distortion(x_n, y_n, *dist_params)
        else:
            x_img, y_img = x_n, y_n

        u_pix = (x_img * self.fx_pix) + self.cx_pix
        v_pix = (y_img * self.fy_pix) + self.cy_pix

        return float(u_pix), float(v_pix), depth

    def is_pixel_in_bounds(self, u_pix: float, v_pix: float, margin: float = 0.0) -> bool:
        """Verifica se il pixel ricade nei limiti dell'immagine."""
        if not self.image_width or not self.image_height:
            return False
        return (
            -margin <= u_pix < self.image_width + margin
            and -margin <= v_pix < self.image_height + margin
        )

    def compute_projection_plane(
        self, p1_pix: Tuple[float, float], p2_pix: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dati due pixel che definiscono una direzione sul piano immagine,
        determina il piano 3D nello spazio mondo passante per il centro ottico C e i due raggi.
        Equazione del piano: n_plane . (P - C) = 0
        Restituisce: (C, n_plane) normalizzato.
        """
        _, d1 = self.pixel_to_ray(p1_pix[0], p1_pix[1])
        _, d2 = self.pixel_to_ray(p2_pix[0], p2_pix[1])

        n_plane = np.cross(d1, d2)
        norm_n = np.linalg.norm(n_plane)
        if norm_n < 1e-9:
            raise ValueError("I due punti immagine sono coincidenti o non definiscono una direzione valida.")
        n_plane = n_plane / norm_n
        return self.pos.copy(), n_plane
