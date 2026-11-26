"""Eagle Projects DINOv2 CPU corrosion tool v2: adds object recognition (YOLO + SAM) to
exclude objects from the analysis or analyse only them. Research demo, not a structural assessment."""
import os, io, json, base64, threading, time, hashlib, uuid
import csv, asyncio
import subprocess, tempfile, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, Response
from feature_sampling import TILED_MODE, MAX_SAMPLES, tiled_features, class_scores, merge_samples
from PIL import ImageFilter
import color_evidence as ce
import report_pdf
import photo_meta
import anomaly_clusters
import object_detection as od

ROOT=Path(__file__).resolve().parent
MODEL_ID="facebook/dinov2-small"
LABELS=("background","healthy","medium","intense")
OBJECT_EXCLUDED_RGBA=(84,126,255,120)      # excluded by a YOLO model (automatic)
MANUAL_EXCLUDED_RGBA=(40,190,255,135)      # excluded by hand (drawn by the operator): same blue family, lighter/cyan
OUTSIDE_FOCUS_RGBA=(5,10,18,150)
COLORS={"healthy":(33,175,143,75),"medium":(249,202,61,155),"intense":(238,62,88,165),"uncertain":(220,225,235,135)}
PROFILE=ROOT/"local_profile.json"
PROFILE_DIR=ROOT/"local_profiles"
LOCK=threading.Lock()
PROFILE_LOCK=threading.RLock()
MODEL=None
PROCESSOR=None
CHECKPOINT_REVISION=None
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
app=FastAPI(title="Rustron / Corrosion Intelligence",docs_url=None,redoc_url=None)

@app.exception_handler(Exception)
async def unexpected_error(request,exc):
    """Unexpected errors: full traceback to errori_server.log, a readable JSON message to the page
    (instead of a plain 'Internal Server Error' the page cannot show)."""
    import traceback
    from fastapi.responses import JSONResponse
    text=f"{type(exc).__name__}: {exc}"
    if isinstance(exc,MemoryError) or "bad alloc" in text.lower() or "not enough memory" in text.lower():
        hint="memoria RAM esaurita (altri programmi pesanti aperti?): chiudi o metti in pausa qualcosa e riprova"
    else:hint="errore interno"
    try:
        with open(ROOT/"errori_server.log","a",encoding="utf-8") as log:
            log.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} {request.method} {request.url.path}\n{traceback.format_exc()}")
    except OSError:pass
    return JSONResponse(status_code=500,content={"detail":f"{hint} · {text[:300]} · dettagli in errori_server.log nella cartella della v2"})

def make_foreground_mask(img):
    """Run the project's isolated BiRefNet CPU environment on an in-memory photo."""
    worker=ROOT/"segmenta_immagine.py"
    python=ROOT/"model_cache"/"seg_env"/"Scripts"/"python.exe"
    if not worker.is_file() or not python.is_file():
        raise HTTPException(503,"Rimozione sfondo locale non installata: eseguire installa_windows.bat aggiornato")
    try:
        with tempfile.TemporaryDirectory(prefix="dinov2-mask-") as temp:
            source=Path(temp)/"photo.png"
            target=Path(temp)/"mask.png"
            img.save(source,format="PNG")
            env=os.environ.copy()
            env["HF_HOME"]=str(ROOT/"model_cache")
            env["U2NET_HOME"]=str(ROOT/"model_cache"/"rembg")
            env["V2_SERVER_PID"]=str(os.getpid())  # the worker exits if this server stops
            env.pop("OMP_NUM_THREADS",None)  # ultralytics sets it to 1 on import: BiRefNet would run on a single core
            run=subprocess.run([str(python),str(worker),str(source),str(target)],
                               cwd=ROOT,capture_output=True,text=True,timeout=240,env=env,
                               creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            if run.returncode!=0 or not target.is_file():
                detail=(run.stderr or run.stdout or "errore senza dettagli")[-1200:]
                raise HTTPException(500,"Rimozione sfondo fallita: "+detail)
            return target.read_bytes()
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504,"Rimozione sfondo oltre il limite di 4 minuti") from exc
    except OSError as exc:
        raise HTTPException(500,"Impossibile avviare BiRefNet locale") from exc

FINE_MASK_DIR=ROOT/"model_cache"/"fine_masks"
FINE_MASK_LOCK=threading.Lock()
FINE_MASK_TIMEOUT=int(os.getenv("V2_FINE_MASK_TIMEOUT","5400"))  # seconds; the CPU is often shared with training jobs

FINE_STATE={"key":None,"proc":None,"background":False,"done":0,"total":0,"computed":0,"skipped":0,"state":"idle","note":None}
FINE_STATE_LOCK=threading.Lock()
PREPARE_WANTED={"key":None}

def needed_mask(objects,box):
    """ROI pixels whose metal mask the analysis will actually use: inside the 'Analizza solo qui'
    objects (the whole ROI when there are none) and outside excluded objects. None = whole ROI."""
    if objects is None:return None
    instance,rules,_=objects
    exclude,focus,_=od.object_masks(instance,rules,box,(box[2]-box[0],box[3]-box[1]))
    need=(focus if focus is not None else np.ones(exclude.shape,bool))&~exclude
    return None if need.all() else need

def fine_keys(source_sha256,box,need):
    """(final mask key, tiles key). Tiles depend only on photo + ROI, so they are shared by every
    variant of objects/modes; the final mask also depends on which tiles were skipped."""
    base=hashlib.sha256(f"{source_sha256}:{list(box)}:tiles_v1".encode()).hexdigest()[:32]
    if need is None:return base,base
    h=hashlib.sha256(np.packbits(need).tobytes()+str(need.shape).encode()).hexdigest()[:16]
    return hashlib.sha256(f"{base}:{h}".encode()).hexdigest()[:32],base

def _kill_tree(pid):
    subprocess.call(["taskkill","/PID",str(pid),"/T","/F"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))

def stop_background_mask(except_key=None):
    """Stop a precomputation started for a ROI/objects no longer current (its finished tiles stay on disk)."""
    with FINE_STATE_LOCK:
        proc=FINE_STATE["proc"]
        if proc and FINE_STATE["background"] and FINE_STATE["key"]!=except_key and proc.poll() is None:
            FINE_STATE["state"]="stopped";_kill_tree(proc.pid)

def make_fine_mask(img,box,source_sha256,need=None,background=False):
    """Sharp metal mask for the ROI: BiRefNet on native-resolution tiles, cached on disk.
    need: bool ROI-size array of the pixels the analysis uses; tiles with none of them are skipped.
    Returns (bool array at ROI pixel size, None) or (None, reason)."""
    import re
    worker=ROOT/"segmenta_riquadri.py"
    python=ROOT/"model_cache"/"seg_env"/"Scripts"/"python.exe"
    if not worker.is_file() or not python.is_file():return None,"ambiente BiRefNet locale non installato"
    key,tiles_key=fine_keys(source_sha256,box,need)
    target=FINE_MASK_DIR/(key+".png");tiles=FINE_MASK_DIR/(tiles_key+"_riquadri")
    size=(box[2]-box[0],box[3]-box[1])
    if not background:stop_background_mask(except_key=key)
    with FINE_MASK_LOCK:
        try:
            if not target.is_file():
                if background and PREPARE_WANTED["key"]!=key:return None,"superata da una ROI più recente"
                FINE_MASK_DIR.mkdir(parents=True,exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="dinov2-fine-") as temp:
                    source=Path(temp)/"photo.png";partial=Path(temp)/"mask.png";need_png=Path(temp)/"need.png"
                    img.save(source,format="PNG")
                    args=[str(python),str(worker),str(source),*map(str,box),str(partial)]
                    if need is not None:
                        Image.fromarray(need.astype(np.uint8)*255,"L").save(need_png,format="PNG");args.append(str(need_png))
                    env=os.environ.copy()
                    env["HF_HOME"]=str(ROOT/"model_cache");env["U2NET_HOME"]=str(ROOT/"model_cache"/"rembg")
                    env["V2_SERVER_PID"]=str(os.getpid())  # the worker exits if this server stops
                    env.pop("OMP_NUM_THREADS",None)  # ultralytics sets it to 1 on import: BiRefNet would run on a single core
                    env["V2_TILE_CACHE"]=str(tiles)  # finished tiles survive a timeout, a restart or a ROI change
                    proc=subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,
                                          creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                    with FINE_STATE_LOCK:
                        FINE_STATE.update(key=key,proc=proc,background=background,done=0,total=0,computed=0,skipped=0,state="running",note=None)
                    lines=[]
                    def read():
                        for line in proc.stdout:
                            lines.append(line)
                            m=re.match(r"tile (\d+)/(\d+)( skipped)?",line)
                            if m:
                                with FINE_STATE_LOCK:
                                    if FINE_STATE["proc"] is proc:
                                        FINE_STATE["done"]=int(m.group(1));FINE_STATE["total"]=int(m.group(2))
                                        FINE_STATE["skipped" if m.group(3) else "computed"]+=1
                    reader=threading.Thread(target=read,daemon=True);reader.start()
                    deadline=time.time()+FINE_MASK_TIMEOUT
                    while proc.poll() is None:
                        if time.time()>deadline:
                            _kill_tree(proc.pid);proc.wait()
                            with FINE_STATE_LOCK:FINE_STATE.update(state="error",note="tempo scaduto",proc=None)
                            return None,f"BiRefNet a riquadri oltre il limite di {FINE_MASK_TIMEOUT//60} minuti (CPU occupata da altri programmi?). I riquadri già calcolati sono salvati: ripetendo l'analisi si riparte da lì"
                        time.sleep(0.5)
                    reader.join(5)
                    if proc.returncode!=0 or not partial.is_file():
                        with FINE_STATE_LOCK:
                            stopped=FINE_STATE["state"]=="stopped"
                            FINE_STATE.update(state="stopped" if stopped else "error",proc=None,note=None if stopped else "".join(lines)[-300:])
                        if stopped:return None,"interrotta: ROI o oggetti cambiati (riquadri salvati)"
                        return None,"BiRefNet a riquadri fallito: "+("".join(lines) or "senza dettagli")[-300:]
                    target.write_bytes(partial.read_bytes())
                    with FINE_STATE_LOCK:FINE_STATE.update(state="done",proc=None)
            elif FINE_STATE["key"]!=key:
                with FINE_STATE_LOCK:FINE_STATE.update(key=key,proc=None,background=background,state="done",done=0,total=0,computed=0,skipped=0)
            with Image.open(target) as saved:
                if saved.size!=size:return None,"maschera di dettaglio in cache con dimensioni inattese"
                return np.asarray(saved.convert("L"))>=128,None
        except OSError as exc:return None,f"maschera di dettaglio non disponibile ({exc})"

def _clean_old_tiles(days=3):
    """Tile folders are kept for reuse; drop the ones not touched for a few days."""
    try:
        for d in FINE_MASK_DIR.glob("*_riquadri"):
            if time.time()-d.stat().st_mtime>days*86400:shutil.rmtree(d,ignore_errors=True)
    except OSError:pass
_clean_old_tiles()

@app.post("/api/prepare-mask")
async def prepare_mask(file:UploadFile=File(...),roi:str=Form(...),object_mask:UploadFile|None=File(None),object_rules:str=Form("[]"),object_info:str=Form("{}")):
    """Start the ROI metal mask in the background as soon as ROI and objects are known, so it is
    ready (or nearly) when the operator presses Analizza. Same key the analysis will look for."""
    img,source_sha256=await image_from_upload(file);box=region(img,roi)
    objects=await objects_from_upload(object_mask,object_rules,object_info,img)
    need=needed_mask(objects,box)
    key,_=fine_keys(source_sha256,box,need)
    if (FINE_MASK_DIR/(key+".png")).is_file():
        img.close();return {"key":key,"state":"done"}
    PREPARE_WANTED["key"]=key
    with FINE_STATE_LOCK:same=FINE_STATE["key"]==key and FINE_STATE["state"]=="running"
    if same:img.close()
    else:
        stop_background_mask(except_key=key)
        def job():
            try:make_fine_mask(img,box,source_sha256,need,background=True)
            finally:img.close()
        threading.Thread(target=job,daemon=True).start()
    return {"key":key,"state":"running"}

@app.get("/api/fine-mask-status")
def fine_mask_status(key:str=""):
    with FINE_STATE_LOCK:
        st={k:FINE_STATE[k] for k in ("key","background","done","total","computed","skipped","state","note")}
    if key and st["key"]!=key:
        st={"key":key,"state":"done" if (FINE_MASK_DIR/(key+".png")).is_file() else "waiting","done":0,"total":0,"computed":0,"skipped":0,"background":True,"note":None}
    return st

@app.post("/api/foreground-mask")
async def foreground_mask_endpoint(file:UploadFile=File(...)):
    img,_=await image_from_upload(file)
    try:
        png=await __import__("asyncio").to_thread(make_foreground_mask,img)
    finally:
        img.close()
    return Response(png,media_type="image/png",headers={"Cache-Control":"no-store"})

@app.get("/api/object-models")
def object_models():
    """Folders and the .pt files in each, without loading them (drop-down menu)."""
    return {"folders":od.list_files(),"sam_model":od.SAM_WEIGHTS,"sam_ready":od.sam_path().is_file(),"modes":list(od.MODES)}

@app.get("/api/object-models/info")
async def object_model_info(id:str):
    try:return await asyncio.to_thread(od.model_info,id)
    except KeyError:raise HTTPException(404,"Modello non trovato nelle cartelle configurate")

@app.post("/api/object-folders")
def add_object_folder(folder:str=Form(...)):
    try:od.add_folder(folder)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    return object_models()

@app.post("/api/object-folders/browse")
async def browse_object_folder():
    try:chosen=await asyncio.to_thread(od.browse_folder)
    except Exception as exc:raise HTTPException(500,f"Finestra di selezione non disponibile: {exc}. Incollare il percorso a mano.") from exc
    if chosen:
        try:od.add_folder(chosen)
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    return {**object_models(),"chosen":chosen}

@app.delete("/api/object-folders")
def remove_object_folder(folder:str):
    try:od.remove_folder(folder)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    return object_models()

@app.post("/api/detect-objects")
async def detect_objects_endpoint(file:UploadFile=File(...),models:str=Form("[]"),confidence:float=Form(0.35),use_sam:bool=Form(True),roi:str=Form("")):
    try:
        ids=json.loads(models)
        if not isinstance(ids,list) or not ids or not all(isinstance(x,str) for x in ids) or len(ids)>16:raise ValueError()
    except (ValueError,TypeError):raise HTTPException(422,"Selezionare almeno un modello di riconoscimento")
    img,_=await image_from_upload(file)
    # Only inside the operator's ROI when one is drawn; empty or [0,0,1,1] = whole photo.
    box=region(img,roi) if roi.strip() and roi.replace(" ","")!="[0,0,1,1]" else None
    try:
        detections,png,info=await asyncio.to_thread(od.detect,img,ids,float(confidence),bool(use_sam),box)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    except OSError as exc:raise HTTPException(503,f"Modello SAM non scaricabile: {exc}") from exc
    finally:img.close()
    return {"detections":detections,"instance_mask_base64":base64.b64encode(png).decode("ascii"),"info":info}

async def objects_from_upload(file,rules_text,info_text,img):
    """Instance mask (from /api/detect-objects) + per-detection modes -> (ids, rules, info)."""
    if file is None:return None
    raw=await file.read(18*1024*1024+1)
    if len(raw)>18*1024*1024:raise HTTPException(413,"Maschera oggetti oltre 18 MB")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.size!=img.size:raise HTTPException(422,"La maschera oggetti deve avere le dimensioni della fotografia")
            if source.mode!="L":raise HTTPException(422,"Maschera oggetti non valida")
            instance=np.asarray(source).copy()
    except (OSError,UnidentifiedImageError,ValueError) as exc:
        raise HTTPException(400,"Maschera oggetti PNG non valida") from exc
    try:rules=od.parse_rules(rules_text,int(instance.max()) if instance.size else 0)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    try:
        info=json.loads(info_text or "{}")
        if not isinstance(info,dict):raise ValueError()
    except (ValueError,TypeError):raise HTTPException(422,"Informazioni sul riconoscimento non valide")
    return instance,rules,info

def model():
    global MODEL, PROCESSOR, CHECKPOINT_REVISION
    with LOCK:
        if MODEL is None:
            import torch
            from transformers import AutoModel, AutoImageProcessor
            torch.set_num_threads(max(1,min(8,int(os.getenv("DINO_CPU_THREADS","4")))))
            PROCESSOR=AutoImageProcessor.from_pretrained(MODEL_ID,use_fast=False)
            MODEL=AutoModel.from_pretrained(MODEL_ID,use_safetensors=True).to("cpu").eval()
            CHECKPOINT_REVISION=getattr(MODEL.config,"_commit_hash",None)
    return PROCESSOR,MODEL

async def image_from_upload(file):
    raw=await file.read(18*1024*1024+1)
    if len(raw)>18*1024*1024: raise HTTPException(413,"Massimo 18 MB")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.width*im.height>40_000_000: raise HTTPException(413,"Massimo 40 MP")
            shot_meta=photo_meta.read_shot_meta(im,file.filename or "")  # before EXIF rotation drops it
            image=ImageOps.exif_transpose(im).convert("RGB")
            image.info["shot_meta"]=shot_meta
            return image,hashlib.sha256(raw).hexdigest()
    except (OSError,UnidentifiedImageError,ValueError) as exc:
        raise HTTPException(400,"Immagine non valida: JPG/PNG/WEBP") from exc


async def foreground_from_upload(file,img):
    if file is None:return None,None
    raw=await file.read(18*1024*1024+1)
    if len(raw)>18*1024*1024:raise HTTPException(413,"Maschera oltre 18 MB")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            oriented=ImageOps.exif_transpose(source)
            if oriented.size!=img.size:
                raise HTTPException(422,"La maschera deve avere le stesse dimensioni della fotografia originale")
            if oriented.mode in ("RGBA","LA") or (oriented.mode=="P" and "transparency" in oriented.info):
                mask=oriented.convert("RGBA").getchannel("A")
            elif oriented.mode in ("L","1"):
                mask=oriented.convert("L")
            else:
                raise HTTPException(422,"Usare una maschera grigia PNG o un PNG trasparente")
            return mask,hashlib.sha256(raw).hexdigest()
    except (OSError,UnidentifiedImageError,ValueError) as exc:
        raise HTTPException(400,"Maschera PNG non valida") from exc

def region(img,text):
    try:
        x0,y0,x1,y1=map(float,json.loads(text))
        if not all(np.isfinite([x0,y0,x1,y1])) or not (0<=x0<x1<=1 and 0<=y0<y1<=1): raise ValueError()
        box=(int(x0*img.width),int(y0*img.height),int(x1*img.width),int(y1*img.height))
        if box[2]-box[0]<42 or box[3]-box[1]<42: raise ValueError()
        return box
    except (ValueError,TypeError,OverflowError):
        raise HTTPException(422,"ROI non valida: selezionare almeno 42 px per lato")

def parse_exclusions(text):
    try:
        items=json.loads(text)
        if not isinstance(items,list) or len(items)>80:raise ValueError()
        clean=[]
        for item in items:
            if not isinstance(item,dict):raise ValueError()
            a,b=item["start"],item["end"]
            if not isinstance(a,list) or not isinstance(b,list) or len(a)!=2 or len(b)!=2:raise ValueError()
            vals=[float(a[0]),float(a[1]),float(b[0]),float(b[1]),float(item["radius_px"])]
            if not np.isfinite(vals).all() or not all(0<=v<=1 for v in vals[:4]) or not 2<=vals[4]<=300:raise ValueError()
            clean.append({"start":vals[:2],"end":vals[2:4],"radius_px":vals[4]})
        return clean
    except (ValueError,TypeError,KeyError,OverflowError):
        raise HTTPException(422,"Esclusioni isolatori non valide")

def excluded_pixels(exclusions,geo,w,h):
    mask=np.zeros((h,w),dtype=bool)
    if not exclusions:return mask
    x0,y0,x1,y1=geo["box"]
    sx=w/(x1-x0);sy=h/(y1-y0)
    for item in exclusions:
        ax=(item["start"][0]*geo["source_width"]-x0)*sx
        ay=(item["start"][1]*geo["source_height"]-y0)*sy
        bx=(item["end"][0]*geo["source_width"]-x0)*sx
        by=(item["end"][1]*geo["source_height"]-y0)*sy
        radius=item["radius_px"]*(sx+sy)/2
        left=max(0,int(np.floor(min(ax,bx)-radius)));right=min(w,int(np.ceil(max(ax,bx)+radius+1)))
        top=max(0,int(np.floor(min(ay,by)-radius)));bottom=min(h,int(np.ceil(max(ay,by)+radius+1)))
        if left>=right or top>=bottom:continue
        yy,xx=np.ogrid[top:bottom,left:right]
        dx,dy=bx-ax,by-ay
        t=np.clip(((xx-ax)*dx+(yy-ay)*dy)/max(dx*dx+dy*dy,1e-12),0,1)
        mask[top:bottom,left:right]|=(xx-(ax+t*dx))**2+(yy-(ay+t*dy))**2<=radius*radius
    return mask

def embeddings(img,box,size,feature_mode="standard"):
    import torch
    if size not in (392,518): raise HTTPException(422,"Risoluzione non valida")
    if feature_mode==TILED_MODE:
        try:return tiled_features(img,box,size,embeddings)
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    if feature_mode!="standard":raise HTTPException(422,"Modalità di estrazione non valida")
    crop=img.crop(box)
    ratio=min(size/crop.width,size/crop.height)
    rw,rh=max(1,round(crop.width*ratio)),max(1,round(crop.height*ratio))
    left,top=(size-rw)//2,(size-rh)//2
    square=Image.new("RGB",(size,size),(128,128,128))
    square.paste(crop.resize((rw,rh),Image.Resampling.BICUBIC),(left,top))
    processor,m=model()
    t=time.perf_counter()
    inp=processor(images=square,return_tensors="pt",do_resize=False,do_center_crop=False)
    with LOCK,torch.inference_mode():
        out=m(pixel_values=inp["pixel_values"]).last_hidden_state[0,1:]
    side=size//14
    vec=out.reshape(side,side,-1).float().cpu().numpy()
    vec/=np.maximum(np.linalg.norm(vec,axis=-1,keepdims=True),1e-8)
    return vec,crop,{"left":left,"top":top,"rw":rw,"rh":rh,"side":size,"box":box,
                     "source_width":img.width,"source_height":img.height},round(time.perf_counter()-t,2)

def patch_from_point(p,img,geo,grid):
    try:
        if not isinstance(p,dict): return None
        x,y=float(p["x"]),float(p["y"])
        if not np.isfinite([x,y]).all() or not(0<=x<=1 and 0<=y<=1): return None
        px,py=x*img.width-geo["box"][0],y*img.height-geo["box"][1]
        bw=geo["box"][2]-geo["box"][0];bh=geo["box"][3]-geo["box"][1]
        if not(0<=px<=bw and 0<=py<=bh):return None
        px=min(px,np.nextafter(float(bw),-np.inf));py=min(py,np.nextafter(float(bh),-np.inf))
        if geo.get("feature_mode")==TILED_MODE:
            gw,gh=geo["grid_dimensions"]
            return min(gh-1,int(py/bh*gh)),min(gw-1,int(px/bw*gw))
        sx=geo["left"]+px*geo["rw"]/bw
        sy=geo["top"]+py*geo["rh"]/bh
        return min(grid-1,int(sy/geo["side"]*grid)),min(grid-1,int(sx/geo["side"]*grid))
    except (KeyError,ValueError,TypeError):return None

def train(vec,img,geo,points,require_complete=True,source_sha256=None,color_source=None):
    """color_source: (ROI pixels HxWx3 uint8, sharp metal mask HxW bool). When given,
    every marker also stores the colour of the metal under the click."""
    patches={c:{} for c in LABELS if c!="background"}; assigned={}; colors={}
    if not isinstance(points,list) or len(points)>160: raise HTTPException(422,"Massimo 160 punti")
    for point_index,p in enumerate(points, start=1):
        if not isinstance(p,dict) or p.get("label") not in LABELS[1:]:
            raise HTTPException(422,"Punto non valido: ogni punto deve avere una classe riconosciuta")
        loc=patch_from_point(p,img,geo,vec.shape[0])
        if loc is None: raise HTTPException(422,"Punto non valido o fuori dalla ROI: correggere le coordinate")
        previous=assigned.get(loc)
        if previous is not None and previous[0]!=p["label"]:
            previous_index,previous_label=previous[1],previous[0]
            raise HTTPException(422,
                f"Conflitto tra marker #{previous_index} ({previous_label}) e #{point_index} ({p['label']}) "
                f"nella patch {loc[1]},{loc[0]}. I due marker sono evidenziati: clicca uno dei due per rimuoverlo.")
        assigned[loc]=(p["label"],point_index)
        if loc not in patches[p["label"]]:
            sample=vec[loc].copy()
            if sample.ndim!=1 or not np.isfinite(sample).all() or float(np.linalg.norm(sample))<=1e-12:
                raise HTTPException(422,"Vettore DINOv2 non valido per un punto annotato")
            patches[p["label"]][loc]=sample
            if color_source is not None:
                lab=ce.sample_color(color_source[0],color_source[1],
                                    float(p["x"])*img.width-geo["box"][0],float(p["y"])*img.height-geo["box"][1])
                colors[loc]=None if lab is None else [round(float(v),3) for v in lab]
    groups={k:list(v.values()) for k,v in patches.items()}
    if require_complete and (not groups["healthy"] or not (groups["medium"] or groups["intense"])):
        raise HTTPException(422,"Occorrono metallo sano e almeno una classe di ruggine presente nella foto; non inventare esempi di classi assenti")
    if not any(groups.values()):raise HTTPException(422,"Aggiungere almeno un punto di calibrazione")
    prototypes={}
    sums={}
    for label,v in groups.items():
        if v:
            mean=np.sum(v,axis=0);norm=float(np.linalg.norm(mean))
            if not np.isfinite(mean).all() or norm<=1e-12: raise HTTPException(422,f"Campioni incompatibili o vettore nullo per {label}")
            sums[label]=mean.tolist();mean/=norm;prototypes[label]=mean.tolist()
    extraction={"mode":geo.get("feature_mode","standard"),"resolution":geo["side"]}
    source_sha256=source_sha256 or hashlib.sha256(img.tobytes()).hexdigest()
    samples=[]
    for label,items in patches.items():
        for loc,vector in items.items():
            provenance={"source_sha256":source_sha256,"roi":list(geo["box"]),"extraction":extraction,"patch_yx":list(loc)}
            identity=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
            samples.append({**provenance,"sample_id":identity,"label":label,"embedding":vector.tolist(),
                            "color_lab":colors.get(loc)})
    return {"schema_version":2,"markers_off_metal":sum(1 for v in colors.values() if v is None),"profile_id":str(uuid.uuid4()),"model":MODEL_ID,
            "checkpoint_revision":CHECKPOINT_REVISION,"preprocessing":preprocessing_config(),
            "created_at_utc":datetime.now(timezone.utc).isoformat(),
            "extraction":extraction,"classifier":"balanced_top3_cosine_v1","samples":samples,
            "prototypes":prototypes,"prototype_sums":sums,"class_scheme":"metal_healthy_medium_intense_v1",
            "sample_counts":{k:len(v) for k,v in groups.items()}}

def preprocessing_config():
    return {"processor_type":"BitImageProcessor","use_fast":False,"do_resize":False,"do_center_crop":False,
            "do_convert_rgb":True,"do_rescale":True,"rescale_factor":1/255,
            "image_mean":[0.485,0.456,0.406],"image_std":[0.229,0.224,0.225],
            "input_sizes":[392,518],"letterbox_fill_rgb":[128,128,128],"resize_interpolation":"PIL.BICUBIC",
            "patch_size":14,"embedding_dim":384,"embedding_normalization":"L2 per patch"}

def profile_file(profile_id):
    if profile_id=="legacy":return PROFILE
    try:return PROFILE_DIR/(str(uuid.UUID(str(profile_id)))+".json")
    except (ValueError,TypeError,AttributeError):raise HTTPException(404,"Profilo non trovato")

def validated_profile(profile_id="legacy"):
    try:
        path=profile_file(profile_id)
        p=json.loads(path.read_text("utf-8"))
        if not isinstance(p,dict) or p.get("schema_version") not in (1,2):
            raise ValueError("Profilo precedente o schema non supportato: eliminarlo esplicitamente e calibrare di nuovo")
        if p.get("model")!=MODEL_ID: raise ValueError("Checkpoint incompatibile")
        model()
        if not CHECKPOINT_REVISION or p.get("checkpoint_revision")!=CHECKPOINT_REVISION:
            raise ValueError("Revisione del checkpoint diversa o non identificabile")
        if p.get("preprocessing")!=preprocessing_config(): raise ValueError("Preprocessing incompatibile")
        prototypes=p.get("prototypes")
        if not isinstance(prototypes,dict) or "healthy" not in prototypes or not ({"medium","intense"}&prototypes.keys()) or "background" in prototypes:
            raise ValueError("Il profilo richiede metallo sano e almeno una classe di ruggine")
        if p.get("class_scheme")!="metal_healthy_medium_intense_v1":
            raise ValueError("Schema classi precedente: cancellare i profili locali e ricreare la calibrazione")
        for k,v in prototypes.items():
            if k not in LABELS: raise ValueError("Classe non riconosciuta nel profilo")
            a=np.asarray(v,dtype=np.float32)
            if a.shape!=(384,) or not np.isfinite(a).all() or float(np.linalg.norm(a))<=1e-12:
                raise ValueError(f"Vettore incompatibile o nullo per la classe {k}")
        if p["schema_version"]==2:
            cfg=p.get("extraction",{})
            if cfg.get("mode") not in ("standard",TILED_MODE) or cfg.get("resolution") not in (392,518):
                raise ValueError("Configurazione di estrazione incompatibile")
            if p.get("classifier")!="balanced_top3_cosine_v1" or p.get("class_scheme")!="metal_healthy_medium_intense_v1":raise ValueError("Schema classi o classificatore non supportato")
            samples=p.get("samples")
            if not isinstance(samples,list) or not 1<=len(samples)<=MAX_SAMPLES:raise ValueError("Campioni mancanti o troppi")
            ids=set();counts={k:0 for k in LABELS[1:]}
            for sample in samples:
                a=np.asarray(sample.get("embedding"),np.float32)
                if sample.get("label") not in prototypes or a.shape!=(384,) or not np.isfinite(a).all() or np.linalg.norm(a)<1e-12:
                    raise ValueError("Campione non valido")
                if not sample.get("sample_id") or sample["sample_id"] in ids:raise ValueError("ID campione duplicato o mancante")
                if sample.get("extraction")!=cfg:raise ValueError("Scale dei campioni incompatibili")
                if sample.get("color_lab") is not None:
                    lab=np.asarray(sample["color_lab"],np.float32)
                    if lab.shape!=(3,) or not np.isfinite(lab).all():raise ValueError("Colore del campione non valido")
                ids.add(sample["sample_id"]);counts[sample["label"]]+=1
            if counts!=p.get("sample_counts") or any(counts[k]==0 for k in prototypes):raise ValueError("Conteggi campioni incoerenti")
        if profile_id!="legacy" and p.get("profile_id")!=str(profile_id):raise ValueError("ID profilo non corrispondente")
        p.setdefault("profile_name","Profilo esistente")
        return p
    except (OSError,ValueError,KeyError,TypeError,OverflowError) as exc:
        raise HTTPException(409,f"Profilo non valido o incompatibile: {exc}. Eliminare esplicitamente il profilo precedente e calibrare di nuovo.")

def classify(vec,crop,geo,p,seconds,source_sha256=None,foreground_mask=None,foreground_sha256=None,exclusions=None,profile_persisted=True,metal=None,metal_note=None,markers_off_metal=0,photo_name="",operator_notes="",shot_meta=None,cluster_anomalies=False,objects=None):
    """objects: (instance ids, rules, info) from objects_from_upload, or None.
    metal: sharp ROI metal mask (bool, ROI pixel size). With it, and a profile whose
    samples carry colours, rust is decided per pixel from colour; the DINOv2 patch
    scores are still computed and reported as context, but no longer decide."""
    import torch
    import torch.nn.functional as F
    dino_names,scores=class_scores(vec,p,LABELS[1:])
    bank=ce.color_bank(p.get("samples",[]),LABELS[1:])
    metal_mask=metal
    use_color=metal_mask is not None and ce.has_color_classes(bank)
    names=[k for k in LABELS[1:] if k in bank] if use_color else dino_names
    size=geo["side"]
    if geo.get("feature_mode")==TILED_MODE:
        t=torch.from_numpy(scores)
    else:
        t=F.interpolate(torch.from_numpy(scores)[None],size=(size,size),mode="bilinear",align_corners=False)[0]
        t=t[:,geo["top"]:geo["top"]+geo["rh"],geo["left"]:geo["left"]+geo["rw"]]
    # Colour works per pixel, so keep enough pixels across a thin lattice member.
    scale=min(1.,(2200 if use_color else 1100)/max(crop.size))
    w,h=max(1,round(crop.width*scale)),max(1,round(crop.height*scale))
    t=F.interpolate(t[None],size=(h,w),mode="bilinear",align_corners=False)[0]
    active_pred=t.argmax(0).numpy()
    stable_ids={"background":0,"healthy":1,"medium":2,"intense":3,"uncertain":4}
    dino_pred=np.asarray([stable_ids[k] for k in dino_names],dtype=np.uint8)[active_pred]
    rust_ids=(stable_ids["medium"],stable_ids["intense"])
    support=None
    if metal_mask is not None:
        # One-pixel erosion drops edge pixels that blend metal with whatever is behind it
        # (and with them most bare wires, which are not lattice steel).
        sharp=Image.fromarray(metal_mask.astype(np.uint8)*255).resize((w,h),Image.Resampling.BILINEAR)
        support=np.asarray(sharp.filter(ImageFilter.MinFilter(3)))>=128
    elif foreground_mask is not None:
        support=np.asarray(foreground_mask.crop(geo["box"]).resize((w,h),Image.Resampling.BILINEAR))>=128
    excluded=excluded_pixels(exclusions or [],geo,w,h)
    manual_excluded=excluded.copy()  # drawn by the operator, kept apart for the map colour
    excluded_foreground=int((excluded & support).sum()) if support is not None else int(excluded.sum())
    object_report=None;object_excluded=object_focus=None
    if objects is not None:
        instance,rules,object_info=objects
        object_excluded,object_focus,object_px=od.object_masks(instance,rules,geo["box"],(w,h))
        before=support if support is not None else np.ones((h,w),bool)
        object_report={"used":True,"models":object_info.get("models",[]),"sam_model":object_info.get("sam_model"),
                       "confidence_threshold":object_info.get("confidence_threshold"),"search_area":object_info.get("area"),"search_area_px":object_info.get("area_px"),
                       "excluded_pixels_in_roi":int((object_excluded&before).sum()),
                       "focus_pixels_in_roi":int((object_focus&before).sum()) if object_focus is not None else None,
                       "objects":[{**{k:r.get(k) for k in ("n","model_name","model","class_name","confidence","box_px","mask_source","operator_override")},
                                   "mode":r["mode"],"pixels_in_analysis_grid":object_px.get(n,0)} for n,r in sorted(rules.items())]}
        summary={}
        for item in object_report["objects"]:
            if item["mode"]!="ignore" and item["pixels_in_analysis_grid"]:
                key=f'{item["class_name"]}|{item["mode"]}';summary[key]=summary.get(key,0)+1
        object_report["removed_by_operator"]=sum(1 for o in object_report["objects"] if o["operator_override"]=="removed")
        object_report["summary"]=[{"class_name":k.split("|")[0],"mode":k.split("|")[1],"count_in_roi":v} for k,v in summary.items()]
        if object_focus is not None:support=before&object_focus
        excluded=excluded|object_excluded
    dino_context=None
    if use_color:
        keep=support&~excluded
        pixels=np.asarray(crop.resize((w,h),Image.Resampling.LANCZOS) if scale<1 else crop)
        ids,raw_rust,rust_level=ce.classify_pixels(ce.rgb_to_lab(ce.box_blur3(pixels))[keep],bank)
        pred=np.zeros((h,w),np.uint8);pred[keep]=ids
        raw_pred=np.zeros((h,w),np.uint8);raw_pred[keep]=np.where(raw_rust,rust_level,stable_ids["healthy"])
        ambiguous=pred==stable_ids["uncertain"]
        low_margin_roi=float(ambiguous.mean())
        colour_rust=np.isin(pred,rust_ids)
        if colour_rust.any():
            dino_context={"dino_agrees_on_rust_pct":round(100*float(np.isin(dino_pred,rust_ids)[colour_rust].mean()),1),
                          "note":"Quota dei pixel di ruggine (decisi dal colore) che anche le patch DINOv2 assegnano a ruggine. Solo informativo."}
    else:
        pred=dino_pred.copy()
        if support is not None:pred[~support]=stable_ids["background"]
        pred[excluded]=stable_ids["background"]
        margin=(torch.topk(t,k=2,dim=0).values[0]-torch.topk(t,k=2,dim=0).values[1]).numpy()
        raw_pred=pred.copy()
        rust_score=t[[names.index(k) for k in ("medium","intense") if k in names]].max(0).values.numpy()
        healthy_score=t[names.index("healthy")].numpy()
        uncertain_rust=np.isin(pred,rust_ids) & ((rust_score-healthy_score)<.025)
        pred[uncertain_rust]=stable_ids["uncertain"]
        ambiguous=(margin<.025)&(pred!=0)
        low_margin_roi=float((margin<.025).mean())
    counts={k:int((pred==stable_ids[k]).sum()) for k in (*LABELS,"uncertain")}
    raw_rust_pixels=int(np.isin(raw_pred,rust_ids).sum())
    metal=counts["healthy"]+counts["medium"]+counts["intense"]+counts["uncertain"]
    rust=counts.get("medium",0)+counts.get("intense",0)
    ambiguity_bytes=io.BytesIO();Image.fromarray(ambiguous.astype(np.uint8)*255).save(ambiguity_bytes,format="PNG")
    if use_color:
        warnings=["Classificazione sperimentale: la ruggine è decisa pixel per pixel dal colore del metallo, confrontato con gli esempi marcati, dentro una maschera del metallo a risoluzione nativa. Le zone grigie restano nel denominatore."]
        if counts["uncertain"]:warnings.append("Zone grigie: colore troppo vicino sia agli esempi sani sia a quelli di ruggine, oppure mai marcato (ombre, vernici, isolatori). Marcare esempi in quelle zone le riduce.")
        if markers_off_metal:warnings.append(f"{markers_off_metal} marker non cadono sul metallo della maschera: non contribuiscono al colore. Ingrandire e cliccare al centro del profilato.")
    else:
        warnings=["Classificazione sperimentale: la percentuale di ruggine include solo assegnazioni con margine di almeno 0,025 rispetto al metallo sano; le zone grigie restano nel denominatore."]
        if counts["uncertain"]:warnings.append("Zone grigie da verificare: il modello non separa abbastanza ruggine e metallo sano. Il margine non è una probabilità.")
        if ambiguous.any():warnings.append("Pixel ambigui da verificare: usare la mappa dedicata; il margine non è una probabilità.")
        if (foreground_mask is not None or metal_mask is not None) and not ce.has_color_classes(bank):
            warnings.append("Profilo senza colori dei campioni: analisi con le sole patch DINOv2, meno precisa su profilati sottili. Marcare di nuovo qualche esempio con «Aggiungi esempi al profilo» oppure eseguire aggiorna_profilo_colore.py.")
    if object_report is not None:
        area=object_info.get("area_px")
        if area and not (area[0]<=geo["box"][0] and area[1]<=geo["box"][1] and area[2]>=geo["box"][2] and area[3]>=geo["box"][3]):
            warnings.append("Gli oggetti sono stati cercati in un'area diversa dalla ROI analizzata: rieseguire il rilevamento sulla ROI attuale.")
        if object_focus is not None and not object_report["focus_pixels_in_roi"]:
            warnings.append("Modalità «analizza solo»: nessun oggetto selezionato cade nel metallo della ROI, quindi non c'è nulla da analizzare.")
        elif object_focus is not None:
            warnings.append("Analisi limitata agli oggetti riconosciuti in modalità «analizza solo»: le percentuali si riferiscono solo al loro metallo visibile.")
        if object_report["excluded_pixels_in_roi"]:
            warnings.append("Oggetti riconosciuti automaticamente (YOLO + SAM) ed esclusi dal conteggio: zone azzurre nella mappa. Verificare che il contorno non copra metallo del sostegno.")
    if metal_note:warnings.append("Maschera di dettaglio non disponibile ("+metal_note+"): "+("usata la maschera grossolana dell'intera foto, che include lo sfondo visibile attraverso il traliccio." if foreground_mask is not None else "analisi senza maschera del metallo, lo sfondo nella ROI viene contato. Risultato non affidabile."))
    if p.get("schema_version")==1:warnings.append("Profilo precedente: una sola media per classe, nessun campione distinto recuperabile.")
    if any(p.get("sample_counts",{}).get(k,0)<3 for k in names):warnings.append("Una o più classi hanno meno di tre esempi.")
    if geo.get("feature_mode")!=TILED_MODE:warnings.append("ROI ridotta in un unico riquadro: dettagli sottili possono andare persi.")
    overlay=np.zeros((h,w,4),np.uint8)
    for k in (*names,"uncertain"):
        if k in COLORS:overlay[pred==stable_ids[k]]=COLORS[k]
    if object_focus is not None:overlay[~object_focus]=OUTSIDE_FOCUS_RGBA
    if object_excluded is not None:overlay[object_excluded]=OBJECT_EXCLUDED_RGBA
    if manual_excluded.any():overlay[manual_excluded]=MANUAL_EXCLUDED_RGBA
    blend=Image.alpha_composite(crop.resize((w,h)).convert("RGBA"),Image.fromarray(overlay,"RGBA"))
    b=io.BytesIO();blend.convert("RGB").save(b,format="PNG")
    # Indexed PNG: pixel values remain stable class IDs, while viewers show colors.
    mask=Image.fromarray(pred,mode="P")
    palette=[0,0,0,33,175,143,249,202,61,238,62,88,220,225,235]
    mask.putpalette(palette+[0]*(768-len(palette)))
    mask_bytes=io.BytesIO();mask.save(mask_bytes,format="PNG")
    weighted=counts["medium"]+2*counts["intense"]
    ambiguous_rust=int((ambiguous & ((raw_pred==stable_ids["medium"]) | (raw_pred==stable_ids["intense"]))).sum())
    ambiguous_rust_pct=round(100*ambiguous_rust/raw_rust_pixels,1) if raw_rust_pixels else None
    if ambiguous_rust_pct is not None and ambiguous_rust_pct>50:
        warnings.append("Verifica manuale consigliata: oltre metà dei pixel classificati come ruggine ha una separazione debole fra le classi.")
    missing_classes=[k for k in ("medium","intense") if k not in names]
    if missing_classes:warnings.append("Classi di ruggine senza esempi: "+", ".join(missing_classes)+". Non vengono assegnate finché il profilo non viene arricchito.")
    # Optional numbered anomalies (UI checkbox "Clusterizza anomalie"); off = report unchanged.
    clusters=anomaly_clusters.find_clusters(pred,rust_ids,stable_ids["intense"]) if cluster_anomalies else None
    report={"result_id":str(uuid.uuid4()),"source_image_sha256":source_sha256,
      "analysis_pixels_sha256":hashlib.sha256(crop.tobytes()).hexdigest(),
      "profile_id":p.get("profile_id"),"profile_name":p.get("profile_name","Profilo temporaneo"),
      "profile_schema_version":p.get("schema_version"),
      "profile_persisted":bool(profile_persisted),
      "profile_classifier":"pixel_color_knn_v1" if use_color else p.get("classifier","cosine_centroids_v1"),
      "decision_basis":"pixel_color" if use_color else "dino_patches",
      "color_decision":{"lightness_weight":ce.L_WEIGHT,"min_margin_lab":ce.MARGIN,"max_distance_lab":ce.FAR,
                        "neighbours":ce.K_NEIGHBOURS,"color_samples":{k:int(len(v)) for k,v in bank.items()}} if use_color else None,
      "dino_context":dino_context,"markers_off_metal":int(markers_off_metal),
      "profile_source_images":len({s["source_sha256"] for s in p.get("samples",[])}) or None,
      "checkpoint_revision":p.get("checkpoint_revision"),"profile_sample_counts":p.get("sample_counts",{}),
      "preprocessing":p.get("preprocessing"),"class_legend":{"0":"background/outside_mask","1":"healthy","2":"medium","3":"intense","4":"uncertain_metal"},
      "mask_dimensions_px":[int(w),int(h)],"class_pixel_counts":counts,
      "foreground_mask":{"used":support is not None,"sha256":foreground_sha256,
                         "native_tiles_refined":metal_mask is not None,"refinement_fallback_reason":metal_note,
                         "threshold_alpha_0_to_255":128 if support is not None else None,
                         "foreground_pixels_in_roi":int(support.sum()) if support is not None else None,
                         "foreground_pct_of_roi":round(100*float(support.mean()),2) if support is not None else None},
      "insulator_exclusions":{"geometry":"oriented_capsule","regions":exclusions or [],"map_rgba":list(MANUAL_EXCLUDED_RGBA),"pixels_in_analysis_grid":int(manual_excluded.sum()),
                               "excluded_foreground_pixels_in_roi":excluded_foreground},
      "parameters":{"resolution":int(geo["side"]),"roi_original_pixels":list(geo["box"]),
                     "feature_mode":geo.get("feature_mode","standard"),"tile_count":geo.get("tile_count",1),
                     "feature_grid_dimensions":[vec.shape[1],vec.shape[0]],
                     "source_pixels_per_patch":geo.get("source_pixels_per_patch",[crop.width*14/geo["rw"],crop.height*14/geo["rh"]]),
                     "source_dimensions_px":[geo["source_width"],geo["source_height"]],"patch_size":14,
                     "roi_fraction_of_source":[geo["box"][0]/geo["source_width"],geo["box"][1]/geo["source_height"],
                                               geo["box"][2]/geo["source_width"],geo["box"][3]/geo["source_height"]]},
      "model":MODEL_ID,"device":"CPU","method":("Per-pixel Lab colour k-NN against labelled examples inside a native-resolution BiRefNet metal mask; DINOv2 patch scores reported as context only"
                if use_color else "DINOv2 patch embeddings + "+p.get("classifier","cosine_centroids_v1")+" + foreground mask excludes background"),
      "created_at_utc":datetime.now(timezone.utc).isoformat(),"roi_original_pixels":list(geo["box"]),
      "shot_captured_at":(shot_meta or {}).get("captured_at") or "","shot_coordinates":(shot_meta or {}).get("coordinates") or "","shot_date_source":(shot_meta or {}).get("date_source") or "","operator_notes":operator_notes.strip()[:180],
      "visible_metal_pct_of_roi":round(100*metal/pred.size,2),
      "corrosion_pct_of_visible_metal":round(100*rust/metal,2) if metal else None,
      "relative_visual_intensity_1_to_2":round(weighted/rust,2) if rust and {"medium","intense"}.issubset(names) else None,
      "calibrated_classes":names,"uncalibrated_rust_classes":missing_classes,
      "class_pct_of_visible_metal":{k:round(100*counts.get(k,0)/metal,2) if metal and (k in names or k=="uncertain") else None for k in ("healthy","medium","intense","uncertain")},
      "low_similarity_margin_pct_roi":round(100*low_margin_roi,1),
      "low_similarity_margin_pct_visible_metal":round(100*int(ambiguous.sum())/metal,1) if metal else None,
      "ambiguous_metal_pixels":int(ambiguous.sum()),"ambiguity_threshold":.025,
      "ambiguous_rust_pixels":ambiguous_rust,"ambiguous_rust_pct":ambiguous_rust_pct,
      "raw_rust_pixels_before_abstention":raw_rust_pixels,"uncertain_metal_pixels":counts["uncertain"],
      "manual_review_recommended":bool(ambiguous_rust_pct is not None and ambiguous_rust_pct>50),
      "ambiguous_pixels_by_class":{k:int(((raw_pred==stable_ids[k])&ambiguous).sum()) for k in ("healthy","medium","intense")},
      "quality_warnings":warnings,
      "inference_seconds":seconds,
      "warning":"POC non validata. Percentuali su metallo visibile nella ROI, non superficie 3D. La maschera BiRefNet può includere cavi e isolatori. Intensità visiva relativa, non perdita di spessore."}
    if clusters is not None:report["anomaly_clusters"]=clusters
    report["object_recognition"]=object_report or {"used":False}
    table=io.StringIO();writer=csv.writer(table)
    writer.writerow(["result_id",report["result_id"],"source_image_sha256",report["source_image_sha256"],
                     "profile_id",report["profile_id"],"profile_name",report["profile_name"],
                     "profile_persisted",report["profile_persisted"],
                     "shot_captured_at",report["shot_captured_at"],"shot_coordinates",report["shot_coordinates"],"shot_date_source",report["shot_date_source"],"operator_notes",report["operator_notes"],
                     "classifier",report["profile_classifier"],"feature_mode",geo.get("feature_mode","standard"),
                     "tile_count",geo.get("tile_count",1),"ambiguous_metal_pixels",int(ambiguous.sum()),
                     "ambiguous_rust_pixels",ambiguous_rust,"ambiguous_rust_pct",ambiguous_rust_pct,
                     "manual_review_recommended",report["manual_review_recommended"],
                     "raw_rust_pixels_before_abstention",raw_rust_pixels,"uncertain_metal_pixels",counts["uncertain"],
                     "calibrated_classes",json.dumps(names),"uncalibrated_rust_classes",json.dumps(missing_classes),
                     "mask_width_px",w,"mask_height_px",h,
                     "source_dimensions_px",json.dumps(report["parameters"]["source_dimensions_px"]),
                     "roi_original_pixels",json.dumps(report["roi_original_pixels"]),"resolution",geo["side"],
                     "foreground_mask_sha256",foreground_sha256,"foreground_pixels_in_roi",report["foreground_mask"]["foreground_pixels_in_roi"],
                     "insulator_exclusions",len(exclusions or []),"excluded_foreground_pixels_in_roi",excluded_foreground])
    writer.writerow(["class_id","classe","pixel","percentuale_roi","percentuale_metallo_visibile"])
    for k in ("background","healthy","medium","intense","uncertain"):
        writer.writerow([stable_ids[k],k,counts.get(k,0),round(100*counts.get(k,0)/pred.size,2),
                         round(100*counts.get(k,0)/metal,2) if metal and (k in names or k=="uncertain") else ""])
    if object_report is not None:
        writer.writerow(["oggetto_n","modello","classe","confidenza","modalita","origine_maschera","modifica_operatore","pixel_griglia_analisi","box_x0","box_y0","box_x1","box_y1"])
        for o in object_report["objects"]:
            writer.writerow([o["n"],o["model_name"] or o["model"],o["class_name"],o["confidence"],o["mode"],o["mask_source"],o["operator_override"] or "",o["pixels_in_analysis_grid"],*(o["box_px"] or ["","","",""])])
        writer.writerow(["oggetti_pixel_esclusi_roi",object_report["excluded_pixels_in_roi"],"oggetti_pixel_focus_roi",object_report["focus_pixels_in_roi"],
                         "sam",object_report["sam_model"],"confidenza_minima",object_report["confidence_threshold"]])
    if clusters is not None:
        writer.writerow(["anomaly_id","fascia","rust_pixels","pct_metallo_visibile","pct_intensa_anomalia",
                         "estensione_pct_altezza_traliccio","bbox_x0","bbox_y0","bbox_x1","bbox_y1"])
        for a in clusters["clusters"]:
            writer.writerow([a["id"],a["band"],a["rust_pixels"],a["pct_of_visible_metal"],a["intense_pct_of_anomaly"],
                             a["extent_pct_of_tower_height"],*a["bbox_px"]])
        writer.writerow(["anomalie_sotto_soglia",clusters["below_threshold_count"],
                         "merge_distance_px",clusters["merge_distance_px"]])
    pdf=report_pdf.build_pdf(photo_name,crop.resize((w,h),Image.Resampling.LANCZOS),blend.convert("RGB"),report,
                             report["operator_notes"],clusters=clusters)
    return {"report":report,"report_pdf_base64":base64.b64encode(pdf).decode("ascii"),"overlay_base64":base64.b64encode(b.getvalue()).decode("ascii"),
            "ambiguity_mask_base64":base64.b64encode(ambiguity_bytes.getvalue()).decode("ascii"),
            "class_mask_base64":base64.b64encode(mask_bytes.getvalue()).decode("ascii"),
            "class_id_labels":list(LABELS),"report_csv":table.getvalue()}

@app.get("/")
def index():return FileResponse(ROOT/"static"/"index.html")

@app.get("/api/status")
def status():
    exists=PROFILE.exists()
    valid=False
    if exists:
        try:
            validated_profile()
            valid=True
        except HTTPException:
            pass
    return {"model":MODEL_ID,"device":"cpu","loaded":MODEL is not None,
            "profile_exists":exists,"profile_valid":valid}

@app.get("/api/profiles")
def list_profiles():
    items=[]
    paths=([("legacy",PROFILE)] if PROFILE.exists() else [])
    if PROFILE_DIR.is_dir():
        paths += [(p.stem,p) for p in sorted(PROFILE_DIR.glob("*.json"))]
    for profile_id,path in paths:
        try:
            raw=json.loads(path.read_text("utf-8"));name=raw.get("profile_name","Profilo esistente")
            valid=True
            try:validated_profile(profile_id)
            except HTTPException:valid=False
            items.append({"profile_id":profile_id,"name":name,"valid":valid,
                          "schema_version":raw.get("schema_version"),"extraction":raw.get("extraction",{"mode":"standard"}),
                          "source_images":len({s["source_sha256"] for s in raw.get("samples",[])}) or None,
                          "sample_counts":raw.get("sample_counts",{})})
        except (OSError,ValueError,AttributeError):
            items.append({"profile_id":profile_id,"name":"Profilo illeggibile","valid":False,"sample_counts":{}})
    return {"profiles":items}

@app.post("/api/calibrate")
async def calibrate(file:UploadFile=File(...),seeds:str=Form(...),roi:str=Form("[0,0,1,1]"),resolution:int=Form(392),foreground_mask:UploadFile|None=File(None),exclusions:str=Form("[]"),save_profile:bool=Form(True),profile_name:str=Form(""),feature_mode:str=Form("standard"),operator_notes:str=Form(""),cluster_anomalies:bool=Form(False),object_mask:UploadFile|None=File(None),object_rules:str=Form("[]"),object_info:str=Form("{}")):
    name=profile_name.strip()
    if save_profile and not name:raise HTTPException(422,"Assegnare un nome al profilo")
    if len(name)>64:raise HTTPException(422,"Nome profilo: massimo 64 caratteri")
    if save_profile:
        existing=list_profiles()["profiles"]
        if any(x["name"].casefold()==name.casefold() for x in existing):raise HTTPException(409,"Esiste già un profilo con questo nome")
    img,source_sha256=await image_from_upload(file);box=region(img,roi)
    support,support_sha256=await foreground_from_upload(foreground_mask,img)
    objects=await objects_from_upload(object_mask,object_rules,object_info,img)
    try:points=json.loads(seeds)
    except (ValueError,TypeError):raise HTTPException(422,"Punti non validi")
    vec,crop,geo,sec=await asyncio.to_thread(embeddings,img,box,resolution,feature_mode)
    metal,note=await asyncio.to_thread(make_fine_mask,img,box,source_sha256)
    p=train(vec,img,geo,points,source_sha256=source_sha256,color_source=None if metal is None else (np.asarray(crop),metal))
    off_metal=p.pop("markers_off_metal",0)
    p["profile_name"]=name if save_profile else "Prova dettaglio ROI"
    out=classify(vec,crop,geo,p,sec,source_sha256,support,support_sha256,parse_exclusions(exclusions),profile_persisted=save_profile,
                 metal=metal,metal_note=note,markers_off_metal=off_metal,photo_name=file.filename or "",shot_meta=img.info.get("shot_meta"),
                 operator_notes=operator_notes,cluster_anomalies=cluster_anomalies,objects=objects)
    if save_profile:
        with PROFILE_LOCK:
            PROFILE_DIR.mkdir(parents=True,exist_ok=True)
            if any(json.loads(q.read_text("utf-8")).get("profile_name","").casefold()==name.casefold() for q in PROFILE_DIR.glob("*.json")):
                raise HTTPException(409,"Esiste già un profilo con questo nome")
            target=PROFILE_DIR/(p["profile_id"]+".json");tmp=target.with_suffix(".tmp")
            tmp.write_text(json.dumps(p,ensure_ascii=False),"utf-8");tmp.replace(target)
    return out

@app.post("/api/analyze")
async def analyze(file:UploadFile=File(...),roi:str=Form("[0,0,1,1]"),resolution:int=Form(392),foreground_mask:UploadFile|None=File(None),exclusions:str=Form("[]"),profile_id:str=Form("legacy"),operator_notes:str=Form(""),cluster_anomalies:bool=Form(False),object_mask:UploadFile|None=File(None),object_rules:str=Form("[]"),object_info:str=Form("{}")):
    p=validated_profile(profile_id);img,source_sha256=await image_from_upload(file);box=region(img,roi)
    support,support_sha256=await foreground_from_upload(foreground_mask,img)
    objects=await objects_from_upload(object_mask,object_rules,object_info,img)
    cfg=p.get("extraction",{"mode":"standard","resolution":resolution})
    vec,crop,geo,sec=await asyncio.to_thread(embeddings,img,box,cfg["resolution"],cfg["mode"])
    # v2: the precise metal mask is computed on the ROI only (cached per photo + ROI); the
    # whole-photo mask is optional and no longer produced when the photo is loaded.
    metal,note=await asyncio.to_thread(make_fine_mask,img,box,source_sha256,needed_mask(objects,box))
    if metal is None and support is None:
        # Without any metal mask every ROI pixel (ground, roads, roofs) would be counted as metal:
        # the rust figure would be meaningless, so no result is better than a wrong one.
        raise HTTPException(503,"Analisi non eseguita: la maschera del metallo nella ROI non è stata calcolata ("+str(note)+"). Senza maschera anche strade e terreno verrebbero contati come metallo e la percentuale di ruggine sarebbe sbagliata. Ripeti l'analisi: i riquadri già calcolati sono salvati e non si rifanno.")
    return classify(vec,crop,geo,p,sec,source_sha256,support,support_sha256,parse_exclusions(exclusions),metal=metal,metal_note=note,photo_name=file.filename or "",operator_notes=operator_notes,cluster_anomalies=cluster_anomalies,shot_meta=img.info.get("shot_meta"),objects=objects)

@app.post("/api/profiles/{profile_id}/samples")
async def add_profile_samples(profile_id:str,file:UploadFile=File(...),seeds:str=Form(...),roi:str=Form("[0,0,1,1]"),resolution:int=Form(392),foreground_mask:UploadFile|None=File(None),exclusions:str=Form("[]"),operator_notes:str=Form(""),cluster_anomalies:bool=Form(False),object_mask:UploadFile|None=File(None),object_rules:str=Form("[]"),object_info:str=Form("{}")):
    if profile_id=="legacy":raise HTTPException(409,"Il profilo legacy non si modifica; crea un nuovo profilo nominato")
    with PROFILE_LOCK:
        p=validated_profile(profile_id)
        expected=profile_file(profile_id).read_bytes()
    img,source_sha256=await image_from_upload(file);box=region(img,roi)
    support,support_sha256=await foreground_from_upload(foreground_mask,img)
    objects=await objects_from_upload(object_mask,object_rules,object_info,img)
    try:points=json.loads(seeds)
    except (ValueError,TypeError):raise HTTPException(422,"Punti non validi")
    cfg=p.get("extraction",{"mode":"standard","resolution":resolution})
    vec,crop,geo,sec=await asyncio.to_thread(embeddings,img,box,cfg["resolution"],cfg["mode"])
    metal,note=await asyncio.to_thread(make_fine_mask,img,box,source_sha256)
    addition=train(vec,img,geo,points,require_complete=False,source_sha256=source_sha256,
                   color_source=None if metal is None else (np.asarray(crop),metal))
    if p.get("schema_version")==2:
        try:added=merge_samples(p,addition["samples"])
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc
        if not added:raise HTTPException(409,"Queste patch sono già presenti nel profilo: nessun duplicato aggiunto")
        # Rebuild summaries only for compatibility; classification uses the individual bank.
        p["prototypes"]={};p["prototype_sums"]={};p["sample_counts"]={k:0 for k in LABELS[1:]}
        for label in LABELS[1:]:
            group=[s["embedding"] for s in p["samples"] if s["label"]==label]
            if group:
                total=np.asarray(group,np.float32).sum(axis=0);norm=float(np.linalg.norm(total))
                if norm<=1e-12:raise HTTPException(422,"Campioni incompatibili")
                p["prototype_sums"][label]=total.tolist();p["prototypes"][label]=(total/norm).tolist()
                p["sample_counts"][label]=len(group)
        addition["prototype_sums"]={} # Already merged exactly once above.
    sums=p.get("prototype_sums");approximate=sums is None
    if sums is None:
        sums={k:(np.asarray(v,dtype=np.float32)*max(1,int(p.get("sample_counts",{}).get(k,1)))).tolist()
               for k,v in p["prototypes"].items()}
    counts=p.setdefault("sample_counts",{})
    for label,vector in addition["prototype_sums"].items():
        old=np.asarray(sums.get(label,np.zeros(384)),dtype=np.float64)
        total=old+np.asarray(vector,dtype=np.float64)
        norm=float(np.linalg.norm(total))
        if not np.isfinite(total).all() or norm<=1e-12:raise HTTPException(422,f"Campioni incompatibili per {label}")
        sums[label]=total.tolist();p["prototypes"][label]=(total/norm).tolist()
        counts[label]=int(counts.get(label,0))+int(addition["sample_counts"].get(label,0))
    p["prototype_sums"]=sums;p["profile_merge_approximate"]=bool(p.get("profile_merge_approximate",False) or approximate)
    p["updated_at_utc"]=datetime.now(timezone.utc).isoformat()
    out=classify(vec,crop,geo,p,sec,source_sha256,support,support_sha256,parse_exclusions(exclusions),
                 metal=metal,metal_note=note,markers_off_metal=addition.get("markers_off_metal",0),photo_name=file.filename or "",shot_meta=img.info.get("shot_meta"),
                 operator_notes=operator_notes,cluster_anomalies=cluster_anomalies,objects=objects)
    path=profile_file(profile_id)
    with PROFILE_LOCK:
        if not path.exists() or path.read_bytes()!=expected:
            raise HTTPException(409,"Profilo modificato o eliminato durante l'analisi: riprovare")
        tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(p,ensure_ascii=False),"utf-8");tmp.replace(path)
    return out

@app.delete("/api/profile")
def delete_profile():PROFILE.unlink(missing_ok=True);return {"ok":True}

@app.delete("/api/profiles/{profile_id}")
def delete_named_profile(profile_id:str):
    if profile_id=="legacy":raise HTTPException(409,"Il profilo legacy non può essere eliminato da questo controllo")
    with PROFILE_LOCK:profile_file(profile_id).unlink(missing_ok=True)
    return {"ok":True}

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host=os.getenv("POC_HOST","127.0.0.1"),port=int(os.getenv("POC_PORT","8766")))
