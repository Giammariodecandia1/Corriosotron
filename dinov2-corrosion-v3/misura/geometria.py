"""Geometria comune: raggi vettorizzati, piani e proiettore pixel → punto 3D sul piano del sostegno."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .distix.calibration_manager import OpticalCalibration
from .distix.camera_model import CameraModel
from .distix.metadata_reader import dji_euler_to_vectors


@dataclass
class Piano:
    p0: np.ndarray
    n: np.ndarray

    def __post_init__(self):
        self.p0 = np.asarray(self.p0, dtype=np.float64)
        n = np.asarray(self.n, dtype=np.float64)
        self.n = n / np.linalg.norm(n)


def _senza_distorsione(xd, yd, k1, k2, p1, p2, k3, iterazioni=20):
    """Inversa Brown-Conrady vettorizzata: stesso Newton con Jacobiano esatto di DISTIX (camera_model)."""
    x, y = xd.copy(), yd.copy()
    for _ in range(iterazioni):
        r2 = x * x + y * y
        r4 = r2 * r2
        dr = 1 + k1 * r2 + k2 * r4 + k3 * r4 * r2
        ddr = k1 + 2 * k2 * r2 + 3 * k3 * r4
        fx = x * dr + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) - xd
        fy = y * dr + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y - yd
        j00 = dr + x * 2 * x * ddr + 2 * p1 * y + 6 * p2 * x
        j01 = x * 2 * y * ddr + 2 * p1 * x + 2 * p2 * y
        j10 = y * 2 * x * ddr + 2 * p1 * x + 2 * p2 * y
        j11 = dr + y * 2 * y * ddr + 6 * p1 * y + 2 * p2 * x
        det = j00 * j11 - j01 * j10
        ok = np.abs(det) > 1e-15
        det = np.where(ok, det, 1.0)
        dx = np.where(ok, (j11 * fx - j01 * fy) / det, 0.0)
        dy = np.where(ok, (-j10 * fx + j00 * fy) / det, 0.0)
        x, y = x - dx, y - dy
        if max(np.max(np.abs(dx), initial=0.0), np.max(np.abs(dy), initial=0.0)) < 1e-12:
            break
    return x, y


def raggi(cam: CameraModel, u, v) -> np.ndarray:
    """Direzioni unitarie (…, 3) dei pixel (u, v), stessa convenzione di DISTIX ``pixel_to_ray``,
    compresa la distorsione Brown-Conrady quando la calibrazione la rende attiva."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    x = (u - cam.cx_pix) / cam.fx_pix
    y = (v - cam.cy_pix) / cam.fy_pix
    dist = cam._distortion_params_if_active()
    if dist is not None:
        x, y = _senza_distorsione(np.atleast_1d(x), np.atleast_1d(y), *dist)
        x, y = x.reshape(u.shape), y.reshape(u.shape)
    d = x[..., None] * cam.right - y[..., None] * cam.up + cam.front
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def interseca_piano(C, dirs, p0, n) -> np.ndarray:
    """Punti 3D dove i raggi C + t·d incontrano il piano; NaN se paralleli o dietro la camera."""
    C = np.asarray(C, dtype=np.float64)
    dirs = np.asarray(dirs, dtype=np.float64)
    den = dirs @ n
    num = float((np.asarray(p0) - C) @ n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(den) > 1e-9, num / den, np.nan)
    t = np.where(t > 0, t, np.nan)
    return C + t[..., None] * dirs


def camera(nome: str, pos, pitch: float, yaw: float, roll: float, k: dict, w: int, h: int,
           scala_f: float = 1.0, **extra) -> CameraModel:
    """CameraModel DISTIX da angoli gimbal DJI e intrinseci (fx scalato di ``scala_f`` per il Monte Carlo)."""
    front, up = dji_euler_to_vectors(pitch, yaw, roll or 0.0)
    if "calibration" not in extra and (k.get("calibrata") or k.get("dist")):
        dist = tuple(k.get("dist") or (0.0,) * 5)
        extra["calibration"] = OpticalCalibration(
            int(w), int(h), k["fx"] * scala_f, k["fy"] * scala_f, k["cx"], k["cy"], *dist,
            distortion_active=any(dist), source=k.get("fonte", ""),
            is_complete=bool(k.get("calibrata")), is_validated=bool(k.get("calibrata")))
    return CameraModel(nome, np.asarray(pos, dtype=np.float64), front, up, image_width=int(w), image_height=int(h),
                       fx_pix=k["fx"] * scala_f, fy_pix=k["fy"] * scala_f, cx_pix=k["cx"], cy_pix=k["cy"], **extra)


@dataclass
class Proiettore:
    """Pixel della foto principale → punti 3D sul piano della faccia del sostegno (dalla sessione multi-foto).

    ``campioni`` sono le 200 varianti Monte Carlo (camera, piano) da cui si ricava l'U95 (intervallo al 95 %).
    """
    cam: CameraModel
    piano: Piano
    etichetta: str
    campioni: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def punti(self, u, v) -> np.ndarray:
        return interseca_piano(self.cam.pos, raggi(self.cam, u, v), self.piano.p0, self.piano.n)

    def punti_campione(self, k: int, u, v) -> np.ndarray:
        cam, piano = self.campioni[k]
        return interseca_piano(cam.pos, raggi(cam, u, v), piano.p0, piano.n)
