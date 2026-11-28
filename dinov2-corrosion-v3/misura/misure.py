"""Misure in m e m² sopra il Proiettore della sessione multi-foto.

U95 = 2σ sui campioni Monte Carlo del proiettore (GUM Supplemento 1). Per linee e aree tracciate a mano
ogni campione sposta anche i vertici di N(0, 1,5 px), l'incertezza di clic di DISTIX.

Le aree di classe si calcolano a blocchi (al massimo ~256×256 blocchi): dentro un blocco l'area di una
cella è costante, pari all'area 3D del quadrilatero del blocco divisa per le sue celle. Il quadrilatero
proiettato su un piano è piano, quindi 0,5·|d1 × d2| è la sua area esatta.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

import anomaly_clusters

from .geometria import Proiettore

SIGMA_CLIC_PX = 1.5
FATTORE_SVILUPPO = 3.0  # angolare a L di ala a: ~4a di acciaio per metro, in foto se ne vede ~1,2–1,4a
U95_FATTORE = 1.0
CLASSI = ((2, "Ruggine media"), (3, "Ruggine intensa"), (4, "Zone grigie"))
RUGGINE = (2, 3)
MAX_BLOCCHI = 256


@dataclass
class Misura:
    id: str
    tipo: str
    nome: str
    valore: float
    u95: float
    unita: str
    etichetta: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["valore"] = round(float(self.valore), 6)
        d["u95"] = round(float(self.u95), 6)
        return d


def _u95(valori) -> float:
    """Metà dell'intervallo di copertura al 95 % (percentili 2,5–97,5, GUM S1 §7.7): pochi campioni Monte Carlo
    assurdi (gimbal estremo, raggio quasi radente) non lo gonfiano come farebbe 2σ."""
    valori = np.asarray([v for v in valori if np.isfinite(v)], dtype=np.float64)
    if len(valori) < 2:
        return 0.0
    basso, alto = np.percentile(valori, [2.5, 97.5])
    return float((alto - basso) / 2)


def _proiezioni(pr: Proiettore):
    """La funzione nominale seguita da quelle dei campioni: f(u, v) -> punti 3D."""
    yield pr.punti
    for k in range(len(pr.campioni)):
        yield lambda u, v, k=k: pr.punti_campione(k, u, v)


class _Blocchi:
    def __init__(self, mask: np.ndarray, roi_box):
        h, w = mask.shape
        self.B = max(1, math.ceil(max(w, h) / MAX_BLOCCHI))
        xs = np.append(np.arange(0, w, self.B), w)
        ys = np.append(np.arange(0, h, self.B), h)
        self.nbx, self.nby = len(xs) - 1, len(ys) - 1
        x0, y0, x1, y1 = (float(v) for v in roi_box)
        self.sx, self.sy, self.x0, self.y0 = (x1 - x0) / w, (y1 - y0) / h, x0, y0
        self.U, self.V = np.meshgrid(x0 + xs * self.sx, y0 + ys * self.sy)
        self.celle = np.outer(np.diff(ys), np.diff(xs)).astype(np.float64)
        jj, ii = np.indices(mask.shape)
        self.indice = (jj // self.B) * self.nbx + (ii // self.B)
        self.mask = mask

    def conta(self, selezione: np.ndarray) -> np.ndarray:
        return np.bincount(self.indice[selezione], minlength=self.nbx * self.nby).reshape(self.nby, self.nbx)

    def area_cella(self, f) -> np.ndarray:
        P = f(self.U, self.V)
        d1 = P[1:, 1:] - P[:-1, :-1]
        d2 = P[:-1, 1:] - P[1:, :-1]
        return 0.5 * np.linalg.norm(np.cross(d1, d2), axis=-1) / self.celle

    def px(self, i, j):
        """Coordinate di griglia (bordi delle celle) → pixel originali."""
        return self.x0 + np.asarray(i, dtype=np.float64) * self.sx, self.y0 + np.asarray(j, dtype=np.float64) * self.sy


def _area(conteggi, aree) -> float:
    """Area delle celle proiettabili; NaN se qualche cella con pixel non si proietta (campione da scartare)."""
    if np.any(conteggi[~np.isfinite(aree)]):
        return float("nan")
    return float(np.sum(conteggi * aree))


def _area_parziale(conteggi, aree) -> tuple[float, float]:
    """Area nominale delle sole celle proiettabili e % di pixel che non si proiettano (sopra l'orizzonte o
    dietro la camera): quella parte non è misurata."""
    ok = np.isfinite(aree)
    tot = float(conteggi.sum())
    persi = float(conteggi[~ok].sum())
    return float(np.sum(conteggi[ok] * aree[ok])), (100.0 * persi / tot if tot else 0.0)


def _pixel_anomalia(rust: np.ndarray, gruppi: Optional[np.ndarray], a: dict) -> np.ndarray:
    """Pixel di ruggine della sola anomalia ``a``: il gruppo (stessa fusione di anomaly_clusters) con lo stesso
    riquadro e lo stesso numero di pixel. Senza gruppi: tutta la ruggine nel riquadro."""
    gx0, gy0, gx1, gy1 = a["bbox_px"]
    sel = np.zeros(rust.shape, bool)
    if gruppi is not None:
        dentro = gruppi[gy0:gy1, gx0:gx1][rust[gy0:gy1, gx0:gx1]]
        for lab in np.unique(dentro):
            ys, xs = np.nonzero(rust & (gruppi == lab))
            if ([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] == list(a["bbox_px"])
                    and len(ys) == a.get("rust_pixels", len(ys))):
                sel[ys, xs] = True
                return sel
    sel[gy0:gy1, gx0:gx1] = rust[gy0:gy1, gx0:gx1]
    return sel


def misure_automatiche(proiettore: Proiettore, class_mask: np.ndarray, roi_box, clusters,
                       fattore: float = FATTORE_SVILUPPO, u95_fattore: float = U95_FATTORE,
                       merge_distance_px: Optional[int] = None) -> list[Misura]:
    bl = _Blocchi(np.asarray(class_mask), roi_box)
    conteggi = {cid: bl.conta(bl.mask == cid) for cid, _ in CLASSI}
    rust = np.isin(bl.mask, RUGGINE)
    gruppi = None
    if clusters and merge_distance_px:
        gruppi = anomaly_clusters.label(anomaly_clusters.dilate(rust, max(1, int(merge_distance_px) // 2)))
    anomalie = []
    for a in clusters or []:
        gx0, gy0, gx1, gy1 = a["bbox_px"]
        sel = _pixel_anomalia(rust, gruppi, a)
        xc, yc = (gx0 + gx1) / 2, (gy0 + gy1) / 2
        u, v = bl.px([gx0, gx1, xc, xc], [yc, yc, gy0, gy1])
        anomalie.append((a, bl.conta(sel), u, v))
    serie = {cid: [] for cid, _ in CLASSI}
    serie_an = [([], [], []) for _ in anomalie]
    persi = {}       # % di pixel non proiettabili nella proiezione nominale
    persi_an = []
    for k, f in enumerate(_proiezioni(proiettore)):
        aree = bl.area_cella(f)
        for cid, _ in CLASSI:
            if k == 0:
                area, persi[cid] = _area_parziale(conteggi[cid], aree)
            else:
                area = _area(conteggi[cid], aree)
            serie[cid].append(area)
        for (a, cnt, u, v), (s_area, s_oriz, s_vert) in zip(anomalie, serie_an):
            p = f(u, v)
            if k == 0:
                area, pct = _area_parziale(cnt, aree)
                persi_an.append(pct)
            else:
                area = _area(cnt, aree)
            s_area.append(area)
            s_oriz.append(float(np.linalg.norm(p[1] - p[0])))
            s_vert.append(float(np.linalg.norm(p[3] - p[2])))
    out = []
    for cid, nome in CLASSI:
        valore, u95 = serie[cid][0], _u95(serie[cid][1:])
        reale = valore * fattore
        rel = math.hypot(u95 / valore, u95_fattore / fattore) if valore > 0 else 0.0
        extra = {"reale_m2": reale, "u95_reale_m2": reale * rel, "fattore": fattore,
                 "u95_fattore": u95_fattore, "etichetta_reale": "STIMA"}
        if persi[cid] > 0:
            extra["non_misurabile_pct"] = round(persi[cid], 1)
        out.append(Misura(f"classe-{cid}", "classe", nome, valore, u95, "m²", proiettore.etichetta, extra))
    for (a, *_), (s_area, s_oriz, s_vert), pct in zip(anomalie, serie_an, persi_an):
        lato = [(_finito(s_oriz[0]), _u95(s_oriz[1:])), (_finito(s_vert[0]), _u95(s_vert[1:]))]
        (lung, u_lung), (larg, u_larg) = sorted(lato, key=lambda t: -t[0])
        extra = {"lunghezza_m": lung, "larghezza_m": larg, "u95_lunghezza_m": u_lung, "u95_larghezza_m": u_larg}
        if pct > 0 or not (np.isfinite(s_oriz[0]) and np.isfinite(s_vert[0])):
            extra["non_misurabile_pct"] = round(max(pct, 0.1), 1)
        out.append(Misura(f"anomalia-{a['id']}", "anomalia", f"Anomalia {a['id']}", s_area[0], _u95(s_area[1:]), "m²",
                          proiettore.etichetta, extra))
    return out


def _finito(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def _lunghezza(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _area_poligono(p: np.ndarray) -> float:
    return float(0.5 * np.linalg.norm(np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)))


ERRORE_MANUALE = ("un punto non cade sulla faccia del sostegno (sopra l'orizzonte o dietro la camera): "
                  "ridisegna la misura sul sostegno")


def _manuale(proiettore: Proiettore, punti_px, calcolo, seed: int) -> tuple[float, float]:
    uv = np.asarray(punti_px, dtype=np.float64)
    valore = calcolo(proiettore.punti(uv[:, 0], uv[:, 1]))
    if not np.isfinite(valore):
        return float("nan"), 0.0
    rng = np.random.default_rng(seed)
    serie = []
    for k in range(len(proiettore.campioni)):
        j = uv + rng.normal(0, SIGMA_CLIC_PX, uv.shape)
        serie.append(calcolo(proiettore.punti_campione(k, j[:, 0], j[:, 1])))
    return valore, _u95(serie)


def _misura_manuale(tipo, unita, calcolo, proiettore, punti_px, nome, seed) -> Misura:
    valore, u95 = _manuale(proiettore, punti_px, calcolo, seed)
    extra = {"punti_px": [list(map(float, p)) for p in punti_px]}
    if not np.isfinite(valore):  # la misura resta in elenco (si può cancellare) con il motivo, senza numeri NaN
        valore, extra["errore"] = 0.0, ERRORE_MANUALE
    return Misura(f"{tipo}-{nome}", tipo, nome, valore, u95, unita, proiettore.etichetta, extra)


def misura_linea(proiettore: Proiettore, punti_px, nome: str, seed: int = 0) -> Misura:
    return _misura_manuale("linea", "m", _lunghezza, proiettore, punti_px, nome, seed)


def misura_area(proiettore: Proiettore, punti_px, nome: str, seed: int = 0) -> Misura:
    return _misura_manuale("area", "m²", _area_poligono, proiettore, punti_px, nome, seed)
