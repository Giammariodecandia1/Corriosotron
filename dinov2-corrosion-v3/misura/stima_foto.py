"""Stima da una foto sola: il sostegno è un piano verticale per la sua base, rivolto verso la camera.

La distanza viene dal telemetro laser se valido e puntato sul sostegno, altrimenti dal raggio del
piede del sostegno intersecato con il suolo alla quota di decollo (RelativeAltitude).
Ogni pixel ha la sua scala: il raggio di ogni pixel si interseca con il piano.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .drone import FotoMeta, Profilo, intrinseci, misurabile
from .geometria import Piano, Proiettore, camera, interseca_piano, raggi, ruota

N_CAMPIONI = 200
SIGMA_SUOLO_M = 1.5        # ±3 m al 95 % fra quota di decollo e base del sostegno
SIGMA_ROT_PIANO_DEG = 10.0  # faccia ruotata rispetto alla vista
SIGMA_TILT_PIANO_DEG = 5.0  # rastremazione del traliccio
MAX_DISTANZA_M = 300.0   # oltre, la base è quasi all'orizzonte: la stima non è affidabile
MAX_TENTATIVI = 10 * N_CAMPIONI
SUGGERIMENTO = "apri la sessione multi-foto oppure clicca la base del sostegno"
Z = np.array([0.0, 0.0, 1.0])


@dataclass
class NessunaStima:
    motivo: str
    suggerimento: str = SUGGERIMENTO


def etichetta_rtk(meta: FotoMeta) -> str:
    stato = {50: "RTK FIX", 34: "RTK FLOAT", 16: "GNSS singolo", 0: "senza RTK"}.get(meta.rtk_flag, "RTK sconosciuto")
    if meta.rtk_std_h is not None:
        return f"{stato} · ±{max(1, round(meta.rtk_std_h * 100))} cm"
    return stato


SIGMA_LASER_MIN_M = 0.2
SIGMA_LASER_REL = 0.01
TOLLERANZA_FACCIA_M = 3.0  # il laser colpisce la faccia a metà altezza, la geometria la base: rastremazione


def _sigma_laser(d: float) -> float:
    return max(SIGMA_LASER_MIN_M, SIGMA_LASER_REL * d)


def _laser_valido(meta: FotoMeta, profilo: Profilo, box) -> bool:
    if meta.lrf_m is None or box is None:
        return False
    if meta.lrf_status and meta.lrf_status.strip().lower() != "normal":
        return False
    lo, hi = profilo.lrf_range_m
    if not lo <= meta.lrf_m <= hi:
        return False
    cx, cy = meta.width / 2, meta.height / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _piano_verticale(cam, p0) -> Optional[Piano]:
    n = cam.pos - p0
    n[2] = 0.0
    if not np.all(np.isfinite(n)) or np.linalg.norm(n) < 1e-6:
        return None
    return Piano(p0, n)


def _base(cam, meta: FotoMeta, laser: bool, piede, quota_suolo: float, d_laser: float = 0.0):
    """Punto 3D da cui passa il piano verticale; None se non calcolabile."""
    if laser:
        d = raggi(cam, meta.width / 2, meta.height / 2)
        return cam.pos + (meta.lrf_m + d_laser) * d
    d = raggi(cam, *piede)
    if d[2] > -0.02:
        return None
    p = interseca_piano(cam.pos, d, np.array([0.0, 0.0, quota_suolo]), Z)
    return None if np.isnan(p).any() else p


def costruisci(meta: FotoMeta, profilo: Profilo, supporto_box, base_px, seed: int = 0):
    """Proiettore della stima da una foto. Il laser vince solo se valido, puntato sul sostegno e coerente con la
    geometria (quando la base è visibile): un traliccio è quasi tutto vuoto e il raggio può passare oltre."""
    ok, motivo = misurabile(meta, profilo)
    if not ok:
        return NessunaStima(motivo)
    box = [float(x) for x in supporto_box] if supporto_box is not None else None
    if _laser_valido(meta, profilo, box):
        geo = _costruisci(meta, profilo, box, base_px, seed, laser=False)
        las = _costruisci(meta, profilo, box, base_px, seed, laser=True)
        if isinstance(geo, Proiettore) and isinstance(las, Proiettore):
            dg, dl = geo.info["distanza_m"], las.info["distanza_m"]
            sg = float(np.std([np.hypot(*(pl.p0 - c.pos)[:2]) for c, pl in geo.campioni]))
            sl = float(np.std([np.hypot(*(pl.p0 - c.pos)[:2]) for c, pl in las.campioni]))
            if abs(dl - dg) > 2 * float(np.hypot(sg, sl)) + TOLLERANZA_FACCIA_M:
                geo.info["avviso"] = (f"il telemetro laser ({dl:.1f} m) non concorda con la geometria ({dg:.1f} m): "
                                      "probabilmente non ha colpito il sostegno, si usa la geometria")
                return geo
        return las if isinstance(las, Proiettore) else geo
    return _costruisci(meta, profilo, box, base_px, seed, laser=False)


def _costruisci(meta: FotoMeta, profilo: Profilo, box, base_px, seed: int, laser: bool):
    k = intrinseci(meta, profilo)
    w, h = meta.width, meta.height
    piede = None
    if not laser:
        if base_px is not None:
            piede = (float(base_px[0]), float(base_px[1]))
        elif box is not None:
            if box[3] >= 0.99 * h:
                return NessunaStima("base del sostegno non visibile: il riquadro tocca il bordo basso della foto")
            piede = ((box[0] + box[2]) / 2, box[3])
        else:
            return NessunaStima("base del sostegno non indicata: nessun riquadro del sostegno")
        if meta.rel_alt is None or meta.rel_alt <= 0:
            return NessunaStima("quota di volo non valida nei metadati")

    def variante(dp=0.0, dy=0.0, dr=0.0, sf=1.0, dz=0.0, rot=0.0, tilt=0.0, dl=0.0):
        cam = camera("principale", np.zeros(3), meta.gimbal_pitch + dp, meta.gimbal_yaw + dy,
                     (meta.gimbal_roll or 0.0) + dr, k, w, h, scala_f=sf)
        b = _base(cam, meta, laser, piede, -meta.rel_alt + dz if not laser else 0.0, dl)
        if b is None:
            return None
        piano = _piano_verticale(cam, b)
        if piano is None:
            return None
        if rot or tilt:
            n = ruota(piano.n, Z, rot)
            n = ruota(n, np.cross(n, Z), tilt)
            piano = Piano(piano.p0, n)
        return cam, piano

    nominale = variante()
    if nominale is None:
        return NessunaStima("la base del sostegno è sopra l'orizzonte della camera: non si può ricavare la distanza")
    cam, piano = nominale
    if np.hypot(*(piano.p0 - cam.pos)[:2]) > MAX_DISTANZA_M:
        return NessunaStima(f"distanza dal sostegno non plausibile (oltre {MAX_DISTANZA_M:.0f} m): base troppo vicina all'orizzonte")
    rng = np.random.default_rng(seed)
    sg, sf = profilo.sigma_gimbal_deg, profilo.sigma_focal_rel
    campioni = []
    for _ in range(MAX_TENTATIVI):
        if len(campioni) == N_CAMPIONI:
            break
        c = variante(*rng.normal(0, sg, 3), 1 + rng.normal(0, sf), rng.normal(0, SIGMA_SUOLO_M),
                     rng.normal(0, SIGMA_ROT_PIANO_DEG), rng.normal(0, SIGMA_TILT_PIANO_DEG),
                     rng.normal(0, _sigma_laser(meta.lrf_m)) if laser else 0.0)
        if c is not None:
            campioni.append(c)
    if len(campioni) < N_CAMPIONI:
        return NessunaStima("geometria instabile: la base del sostegno è troppo vicina all'orizzonte della camera")
    uc, vc = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) if box else (w / 2, h / 2)
    p = interseca_piano(cam.pos, raggi(cam, np.array([uc, uc + 1, uc]), np.array([vc, vc, vc + 1])), piano.p0, piano.n)
    scala = 1000 * float(np.sqrt(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))))
    info = {"drone": profilo.nome, "rtk": etichetta_rtk(meta),
            "distanza_m": round(float(np.hypot(*(piano.p0 - cam.pos)[:2])), 2),
            "fonte_distanza": "laser" if laser else "geometria",
            "scala_mm_px_centro": round(scala, 2) if np.isfinite(scala) else None,
            "profilo_generico": profilo.generico, "fonte_focale": k["fonte"]}
    return Proiettore(cam=cam, piano=piano, etichetta="STIMA", campioni=campioni, info=info)
