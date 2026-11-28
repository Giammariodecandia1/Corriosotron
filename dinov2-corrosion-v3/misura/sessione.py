"""Sessione di misura con più foto e tie point, come in DISTIX.

L'operatore aggiunge foto, clicca i punti e preme «Calcola». Il resto avviene qui senza parametri:
bundle adjustment vincolato DISTIX → triangolazione → piano della faccia → controlli → proiettore.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from . import drone
from .distix.bundle_adjustment import TiePointObservation, run_constrained_bundle_adjustment_with_diagnostics
from .distix.coordinate_system import CoordinateManager
from .distix.triangulation import compute_epipolar_curve_points, triangulate_from_cameras
from .geometria import Piano, Proiettore, camera
from .stima_foto import etichetta_rtk

ROOT = Path(__file__).resolve().parents[1]
CARTELLA = ROOT / "misure"
MIN_TIE = 4
N_CAMPIONI = 200
SIGMA_PUNTO_M = 0.02          # se la triangolazione non dà covarianza
SIGMA_ASSETTO_DEG = 0.12      # valore DISTIX quando il BA non stima l'incertezza residua
MAX_REPROJ_PX = 10.0          # soglia di default di triangulate_from_cameras (DISTIX)
RICHIESTI = ("T1", "T2", "T3", "T4")


@dataclass
class Sessione:
    sha_principale: str
    foto: list = field(default_factory=list)   # [{"sha", "nome", "w", "h", "misurabile", "motivo", ...}]
    tie: dict = field(default_factory=dict)    # pid -> {sha: [u, v]}
    cartella: Path = CARTELLA

    def _file(self) -> Path:
        return Path(self.cartella) / f"{self.sha_principale}.json"

    def salva(self) -> None:
        Path(self.cartella).mkdir(parents=True, exist_ok=True)
        data = {"sha_principale": self.sha_principale, "foto": self.foto,
                "tie": {p: {s: [float(c[0]), float(c[1])] for s, c in o.items()} for p, o in self.tie.items()}}
        self._file().write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def carica(cls, sha: str, cartella: Path = CARTELLA) -> Optional["Sessione"]:
        path = Path(cartella) / f"{sha}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(sha_principale=data["sha_principale"], foto=data.get("foto", []),
                   tie={p: {s: tuple(c) for s, c in o.items()} for p, o in data.get("tie", {}).items()},
                   cartella=Path(cartella))

    @classmethod
    def apri(cls, meta_bytes: bytes, analysed_size, nome: str, sha: str, cartella: Path = CARTELLA) -> "Sessione":
        """Sessione della foto principale (creata se non esiste). ``analysed_size`` = immagine analizzata."""
        s = cls.carica(sha, cartella)
        if s is None:
            s = cls(sha_principale=sha, foto=[], tie={}, cartella=Path(cartella))
            s._registra(meta_bytes, nome, sha, analysed_size)
            s.salva()
        return s

    def _registra(self, data: bytes, nome: str, sha: str, size) -> dict:
        (Path(self.cartella) / "foto").mkdir(parents=True, exist_ok=True)
        (Path(self.cartella) / "foto" / f"{sha}.jpg").write_bytes(data)
        meta = drone.leggi_foto(data, size)
        profilo = drone.scegli_profilo(meta)
        ok, motivo = drone.misurabile(meta, profilo)
        info = {"sha": sha, "nome": nome, "w": int(size[0]), "h": int(size[1]), "misurabile": ok, "motivo": motivo,
                "drone": profilo.nome, "rtk": etichetta_rtk(meta), "yaw": meta.gimbal_yaw}
        self.foto.append(info)
        return info

    def aggiungi_foto(self, data: bytes, nome: str) -> dict:
        sha = hashlib.sha256(data).hexdigest()
        presente = next((f for f in self.foto if f["sha"] == sha), None)
        if presente is not None:
            return {**presente, "motivo": "foto già presente nella sessione"}
        try:
            with Image.open(io.BytesIO(data)) as im:
                size = im.size
        except OSError:
            return {"sha": sha, "nome": nome, "misurabile": False, "motivo": "il file non è una foto leggibile"}
        info = self._registra(data, nome, sha, size)
        self.salva()
        return info

    def imposta_punto(self, pid: str, sha: str, u, v) -> None:
        obs = self.tie.setdefault(pid, {})
        if u is None or v is None:
            obs.pop(sha, None)
            if not obs:
                self.tie.pop(pid, None)
        else:
            obs[sha] = (float(u), float(v))
        self.salva()


@dataclass
class EsitoSessione:
    completa: bool
    etichetta: str
    controlli: list
    tie_info: dict
    proiettore: Optional[Proiettore]
    ba: dict
    messaggi: list

    def to_dict(self) -> dict:
        return {"completa": self.completa, "etichetta": self.etichetta, "controlli": self.controlli,
                "tie_info": self.tie_info, "ba": self.ba, "messaggi": self.messaggi,
                "info": self.proiettore.info if self.proiettore else None}


def camere(sessione: Sessione) -> dict:
    """CameraModel DISTIX per ogni foto misurabile, in ENU locale con origine nella foto principale."""
    out, metas = {}, {}
    for f in sessione.foto:
        if not f.get("misurabile"):
            continue
        data = (Path(sessione.cartella) / "foto" / f"{f['sha']}.jpg").read_bytes()
        meta = drone.leggi_foto(data, (f["w"], f["h"]))
        if meta.abs_alt is None:
            continue
        metas[f["sha"]] = (meta, drone.scegli_profilo(meta))
    if sessione.sha_principale not in metas:
        return out
    m0 = metas[sessione.sha_principale][0]
    cm = CoordinateManager()
    cm.set_local_enu_origin(m0.lon, m0.lat, m0.abs_alt)
    for sha, (meta, profilo) in metas.items():
        pos = cm.wgs84_to_local(meta.lon, meta.lat, meta.abs_alt)
        k = drone.intrinseci(meta, profilo)
        out[sha] = camera(sha, pos, meta.gimbal_pitch, meta.gimbal_yaw, meta.gimbal_roll, k,
                          meta.width, meta.height, rtk_fix=meta.rtk_flag == 50, rtk_flag=meta.rtk_flag,
                          rtk_std=meta.rtk_std_h, rtk_std_hgt=meta.rtk_std_v)
        out[sha].sigma_gimbal_deg = profilo.sigma_gimbal_deg
    return out


def _piano(punti: np.ndarray, verso: np.ndarray) -> Piano:
    c = punti.mean(axis=0)
    n = np.linalg.svd(punti - c)[2][-1]
    if (verso - c) @ n < 0:
        n = -n
    return Piano(c, n)


def calcola(sessione: Sessione, camere_fn: Callable = camere, seed: int = 0) -> EsitoSessione:
    messaggi = [f"{f['nome']}: {f['motivo']} (esclusa)" for f in sessione.foto if f.get("misurabile") is False]
    cams = camere_fn(sessione)
    principale = sessione.sha_principale
    if principale not in cams:
        return EsitoSessione(False, "INCOMPLETA", [], {}, None, {},
                             messaggi + ["la foto principale non è misurabile: servono i suoi metadati di posizione e gimbal"])
    oss = {p: {s: uv for s, uv in o.items() if s in cams} for p, o in sessione.tie.items()}
    validi = {p: o for p, o in oss.items() if len(o) >= 2}
    tie_info = {p: {"n_foto": len(o), "reproj_px": None} for p, o in oss.items()}
    if len(validi) < MIN_TIE:
        for p in sorted(set(RICHIESTI) | set(oss)):
            n = len(oss.get(p, {}))
            if n == 0:
                messaggi.append(f"{p} non è ancora stato segnato")
            elif n == 1:
                messaggi.append(f"{p} manca in una seconda foto")
        messaggi.append(f"servono almeno {MIN_TIE} tie point visti in almeno 2 foto (ora {len(validi)})")
        return EsitoSessione(False, "INCOMPLETA", [], tie_info, None, {}, messaggi)

    osservazioni = [TiePointObservation(p, s, uv[0], uv[1]) for p, o in validi.items() for s, uv in o.items()]
    ba = run_constrained_bundle_adjustment_with_diagnostics(cams, osservazioni)
    tri = {p: triangulate_from_cameras([(cams[s], tuple(uv)) for s, uv in o.items()]) for p, o in validi.items()}
    for p, r in tri.items():
        tie_info[p]["reproj_px"] = round(float(r.max_reproj_px), 2)
    cam0 = cams[principale]
    punti = np.array([r.point_3d for r in tri.values()])
    piano = _piano(punti, cam0.pos)

    usate = sorted({s for o in validi.values() for s in o})
    conv = min(r.max_convergence_angle_deg for r in tri.values())
    controlli = [
        {"id": "rtk", "testo": "RTK FIX su tutte le foto", "ok": all(cams[s].rtk_fix for s in usate)},
        {"id": "convergenza", "testo": f"Convergenza ≥ 15° ({conv:.0f}°)", "ok": conv >= 15.0},
        # solo coerenza dei clic: la convergenza ha il suo controllo (is_reliable di DISTIX le unisce)
        {"id": "riproiezione", "testo": "Riproiezione entro soglia DISTIX",
         "ok": all(r.all_in_front and r.max_reproj_px <= MAX_REPROJ_PX for r in tri.values())},
        {"id": "tie", "testo": f"Almeno {MIN_TIE} tie point in ≥ 2 foto", "ok": True},
        {"id": "calibrazione", "testo": "Camera calibrata",
         "ok": all(getattr(cams[s].calibration, "is_validated", False) for s in usate)},
    ]
    spiega = {"rtk": "almeno una foto non è in RTK FIX: la posizione del drone non è abbastanza precisa",
              "convergenza": "le foto sono troppo allineate: scattane una più di lato (almeno 15° di differenza)",
              "riproiezione": "alcuni punti non tornano fra le foto: controlla di aver cliccato lo stesso punto fisico",
              "calibrazione": "la camera non ha un profilo di calibrazione: la misura resta NON VALIDATA"}
    messaggi += [spiega[c["id"]] for c in controlli if not c["ok"] and c["id"] in spiega]
    for p, r in tri.items():
        if not (r.all_in_front and r.max_reproj_px <= MAX_REPROJ_PX) and r.notes:
            messaggi.append(f"{p}: {r.notes}")
    etichetta = "VALIDATA" if all(c["ok"] for c in controlli) else "NON VALIDATA"

    rng = np.random.default_rng(seed)
    sigma_ass = (getattr(ba, "post_ba_attitude_std_deg", {}) or {}).get(principale) or SIGMA_ASSETTO_DEG
    campioni = []
    for _ in range(N_CAMPIONI):
        pk = np.array([rng.multivariate_normal(r.point_3d, r.covariance_3d) if r.covariance_3d is not None
                       else r.point_3d + rng.normal(0, SIGMA_PUNTO_M, 3) for r in tri.values()])
        ck = cam0.clone()
        ck.set_rotation_from_delta(*rng.normal(0, sigma_ass, 3))
        campioni.append((ck, _piano(pk, cam0.pos)))
    info = {"tie_point": len(validi), "foto_usate": len(usate), "convergenza_deg": round(conv, 1),
            "fonte_distanza": "tie point", "ba": ba.notes}
    pr = Proiettore(cam=cam0, piano=piano, etichetta=etichetta, campioni=campioni, info=info)
    return EsitoSessione(True, etichetta, controlli, tie_info, pr, ba.to_dict(), messaggi)


def suggerisci(sessione: Sessione, pid: str, sha_dst: str, camere_fn: Callable = camere) -> list:
    """Curva epipolare nella foto ``sha_dst`` per il punto ``pid`` (dalla foto principale se possibile)."""
    oss = sessione.tie.get(pid, {})
    cams = camere_fn(sessione)
    if sha_dst not in cams:
        return []
    sorgenti = [sessione.sha_principale] + [s for s in oss if s != sessione.sha_principale]
    src = next((s for s in sorgenti if s in oss and s in cams and s != sha_dst), None)
    if src is None:
        return []
    return [list(map(float, p)) for p in compute_epipolar_curve_points(cams[src], tuple(oss[src]), cams[sha_dst])]
