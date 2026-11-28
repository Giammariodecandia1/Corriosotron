"""Endpoint /api/misura/*: sessione multi-foto con tie point e PDF con le misure.

Le misure in metri si fanno solo con la sessione (almeno 2 foto dello stesso sostegno da direzioni diverse):
la stima da una foto sola supponeva la faccia frontale e sbagliava del 20–30 % sulle facce oblique.

Le analisi vivono in memoria (``analisi``: result_id → dati dell'analisi, le ultime 5). Se un'analisi non
c'è più, si risponde 404 e si chiede di ripeterla: niente risultati ricostruiti a metà.
"""
from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from . import misure, sessione

MAX_FOTO = 40 * 1024 * 1024
FATTORE = Field(misure.FATTORE_SVILUPPO, gt=0, le=10)


def _sha(v: str) -> str:
    if not isinstance(v, str) or not v.isalnum() or len(v) > 128:
        raise ValueError("identificativo foto non valido")
    return v


class Forma(BaseModel):
    nome: str
    punti: list[list[float]]


class RichiestaSessione(BaseModel):
    result_id: str


class Punto(BaseModel):
    sha_principale: str
    pid: str = Field(pattern=r"^T\d{1,3}$")
    sha_foto: str
    u: Optional[float] = None
    v: Optional[float] = None
    _v = field_validator("sha_principale", "sha_foto")(_sha)


class Suggerimento(BaseModel):
    sha_principale: str
    pid: str
    sha_foto: str
    _v = field_validator("sha_principale", "sha_foto")(_sha)


class Calcolo(BaseModel):
    sha_principale: str
    result_id: str
    linee: list[Forma] = Field(default_factory=list)
    aree: list[Forma] = Field(default_factory=list)
    fattore: float = FATTORE
    _v = field_validator("sha_principale")(_sha)


class RichiestaPdf(BaseModel):
    result_id: str
    misure: list[dict]
    info: Optional[dict] = None
    controlli: Optional[list[dict]] = None


def _tutte(pr, e, linee, aree, fattore):
    cl = e.get("clusters")
    clusters = cl.get("clusters") if isinstance(cl, dict) else cl
    merge = cl.get("merge_distance_px") if isinstance(cl, dict) else None
    out = misure.misure_automatiche(pr, e["pred"], e["roi_box"], clusters, fattore=fattore, merge_distance_px=merge)
    out += [misure.misura_linea(pr, f.punti, f.nome) for f in linee if len(f.punti) >= 2]
    out += [misure.misura_area(pr, f.punti, f.nome) for f in aree if len(f.punti) >= 3]
    return [m.to_dict() for m in out]


def crea_router(analisi, cartella: Path = sessione.CARTELLA) -> APIRouter:
    r = APIRouter(prefix="/api/misura")

    def voce(result_id):
        e = analisi.get(result_id)
        if e is None:
            raise HTTPException(404, "Analisi non più in memoria: ripeti l'analisi e poi riapri Misura")
        return e

    def sess(sha) -> sessione.Sessione:
        try:
            _sha(sha)
        except ValueError:
            raise HTTPException(422, "Identificativo della sessione non valido")
        s = sessione.Sessione.carica(sha, cartella)
        if s is None:
            raise HTTPException(404, "Sessione di misura non trovata: riaprila dal pannello Misura")
        return s

    @r.post("/sessione/apri")
    async def apri(req: RichiestaSessione):
        e = voce(req.result_id)
        s = await asyncio.to_thread(sessione.Sessione.apri, e["meta_bytes"], e["analysed_size"], e["photo_name"],
                                    e["sha"], cartella)
        return {"sessione": {"sha_principale": s.sha_principale, "foto": s.foto, "tie": s.tie}}

    @r.post("/sessione/foto")
    async def foto(sha_principale: str = Form(...), file: UploadFile = File(...)):
        s = sess(sha_principale)
        raw = await file.read(MAX_FOTO + 1)
        if len(raw) > MAX_FOTO:
            raise HTTPException(413, "Foto oltre 40 MB")
        return await asyncio.to_thread(s.aggiungi_foto, raw, file.filename or "foto.jpg")

    @r.get("/sessione/foto/{sha}")
    def leggi_foto(sha: str):
        path = Path(cartella) / "foto" / f"{sha}.jpg"
        if not sha.isalnum() or not path.exists():
            raise HTTPException(404, "Foto non trovata")
        return FileResponse(path, media_type="image/jpeg")

    @r.post("/sessione/punto")
    def punto(req: Punto):
        s = sess(req.sha_principale)
        if req.sha_foto not in {f["sha"] for f in s.foto}:
            raise HTTPException(422, "La foto non fa parte della sessione")
        s.imposta_punto(req.pid, req.sha_foto, req.u, req.v)
        return {"tie": s.tie}

    @r.post("/sessione/suggerisci")
    async def suggerisci(req: Suggerimento):
        s = sess(req.sha_principale)
        return {"curva": await asyncio.to_thread(sessione.suggerisci, s, req.pid, req.sha_foto)}

    @r.post("/sessione/calcola")
    async def calcola(req: Calcolo):
        e = voce(req.result_id)
        s = sess(req.sha_principale)
        if s.sha_principale != e["sha"]:
            raise HTTPException(409, "La sessione appartiene a un'altra foto: riapri la sessione da questa analisi")
        esito = await asyncio.to_thread(sessione.calcola, s)
        out = esito.to_dict()
        out["misure"] = []
        if esito.proiettore is not None:
            out["misure"] = await asyncio.to_thread(_tutte, esito.proiettore, e, req.linee, req.aree, req.fattore)
        return out

    @r.post("/pdf")
    async def pdf(req: RichiestaPdf):
        import report_pdf
        e = voce(req.result_id)
        rep = e["report"]

        def build():
            return report_pdf.build_pdf(e["photo_name"], e["crop"], e["blend"], rep, rep.get("operator_notes", ""),
                                        clusters=e.get("clusters"), shot_map=e.get("map_info"), misure=req.misure,
                                        misure_info=req.info, misure_controlli=req.controlli)
        data = await asyncio.to_thread(build)
        return {"report_pdf_base64": base64.b64encode(data).decode("ascii")}

    return r
