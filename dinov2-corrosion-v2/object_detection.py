"""Generic object recognition for the analysis: any YOLO model + SAM masks.

YOLO models (.pt) are taken from one or more folders: `modelli_oggetti/` inside the tool
plus any folder the operator adds from the UI (saved in `modelli_oggetti/cartelle.json`).
Several models can be active in the same analysis. For each detected object the
operator chooses, per class, what it means for the corrosion analysis:
  - "exclude": the object's pixels are removed from the analysis (e.g. insulators);
  - "focus":   the analysis only looks inside these objects (e.g. a support, later);
  - "ignore":  the object is shown but changes nothing.

Detection-only models give boxes: SAM turns each box into the object's real outline.
Segmentation models (task "segment") already give masks and are used as they are.

A model is identified by its absolute path. Optional sidecar `<model>.json` next to it:
  {"enabled": true, "default_modes": {"Isolatori": "exclude"}, "imgsz": 960,
   "description": "Isolatori V1 (YOLO11s)"}
"""
import io, json, os, threading, time, urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "modelli_oggetti"
FOLDERS_FILE = MODELS_DIR / "cartelle.json"
SAM_DIR = ROOT / "model_cache" / "sam"
# Any ultralytics SAM checkpoint name works. Measured 2026-11-24 on DJI tower photos (CPU):
# sam2.1_b follows insulator outlines closely; mobile_sam is ~3x faster but leaks onto the
# lattice behind them, which would remove sound steel from the analysis.
SAM_WEIGHTS = os.getenv("V2_SAM_WEIGHTS", "sam2.1_b.pt")
SAM_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{}"
MODES = ("exclude", "focus", "ignore")
MAX_OBJECTS = 254  # instance ids 1..254 in an 8-bit PNG
MAX_LOADED = 6     # YOLO models kept in memory, so several can run in one analysis
LOCK = threading.Lock()
_YOLO, _SAM, _INFO = OrderedDict(), {}, {}


def _threads():
    import torch
    torch.set_num_threads(max(1, min(8, int(os.getenv("DINO_CPU_THREADS", "4")))))


def _sidecar(path):
    side = path.with_suffix(".json")
    if not side.is_file():
        return {}
    try:
        data = json.loads(side.read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


# --- folders --------------------------------------------------------------------------
def folders():
    """The tool's own folder first, then the ones added by the operator."""
    extra = []
    try:
        data = json.loads(FOLDERS_FILE.read_text("utf-8"))
        extra = [str(x) for x in data.get("folders", []) if isinstance(x, str)]
    except (OSError, ValueError, AttributeError):
        pass
    out, seen = [MODELS_DIR.resolve()], {str(MODELS_DIR.resolve()).casefold()}
    for item in extra:
        path = Path(item).resolve()
        if str(path).casefold() not in seen:
            seen.add(str(path).casefold())
            out.append(path)
    return out


def _save_extra(paths):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = FOLDERS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"folders": [str(p) for p in paths]}, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(FOLDERS_FILE)


def add_folder(text):
    raw = str(text or "").strip().strip('"')
    path = Path(raw).expanduser()
    if not raw or not path.is_dir():
        raise ValueError("Cartella non trovata: " + raw)
    path = path.resolve()
    if all(str(path).casefold() != str(p).casefold() for p in folders()):
        _save_extra(folders()[1:] + [path])
    return path


def remove_folder(text):
    target = str(Path(str(text)).resolve()).casefold()
    if target == str(MODELS_DIR.resolve()).casefold():
        raise ValueError("La cartella interna dei modelli non si può togliere")
    _save_extra([p for p in folders()[1:] if str(p).casefold() != target])


def browse_folder():
    """Native Windows folder picker, opened by the local server on the operator's desktop."""
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        chosen = filedialog.askdirectory(parent=root, title="Scegli la cartella con i modelli YOLO (.pt)",
                                         mustexist=True)
    finally:
        root.destroy()
    return chosen or None


def resolve_model(model_id):
    """Model id = absolute path of a .pt directly inside one of the configured folders."""
    path = Path(str(model_id)).resolve()
    if path.suffix.lower() != ".pt" or not path.is_file() or all(path.parent != f for f in folders()):
        raise KeyError(model_id)
    return path


def list_files():
    """Every .pt in every folder, without loading them (fast): for the drop-down menu."""
    out = []
    for folder in folders():
        files = sorted(folder.glob("*.pt"), key=lambda p: p.name.casefold()) if folder.is_dir() else []
        out.append({"folder": str(folder), "internal": folder == MODELS_DIR.resolve(), "exists": folder.is_dir(),
                    "models": [{"id": str(p), "name": p.stem, "size_mb": round(p.stat().st_size / 1e6, 1),
                                "enabled_by_default": bool(_sidecar(p).get("enabled", False))} for p in files]})
    return out


# --- models ---------------------------------------------------------------------------
def _load_yolo(model_id):
    path = resolve_model(model_id)
    key = (str(path), path.stat().st_mtime_ns)
    if key in _YOLO:
        _YOLO.move_to_end(key)
    else:
        from ultralytics import YOLO
        _threads()
        _YOLO[key] = YOLO(str(path))
        while len(_YOLO) > MAX_LOADED:
            _YOLO.popitem(last=False)
    return _YOLO[key]


def model_info(model_id):
    """{id, name, folder, task, classes, default_modes, imgsz, description, error}. Loads the model once."""
    path = resolve_model(model_id)
    stamp = path.stat().st_mtime_ns
    cached = _INFO.get(str(path))
    if cached is None or cached[0] != stamp:
        try:
            with LOCK:
                m = _load_yolo(str(path))
                train_args = (getattr(m, "ckpt", None) or {}).get("train_args", {}) or {}
                info = {"task": m.task, "classes": {int(k): str(v) for k, v in m.names.items()},
                        "trained_imgsz": train_args.get("imgsz"), "error": None}
            if info["task"] not in ("detect", "segment", "obb"):
                info["error"] = f"tipo di modello non supportato ({info['task']}): servono rilevamento o segmentazione"
        except Exception as exc:  # unreadable or incompatible model: say why
            info = {"task": None, "classes": {}, "trained_imgsz": None, "error": str(exc)[:200]}
        _INFO[str(path)] = (stamp, info)
    info = _INFO[str(path)][1]
    side = _sidecar(path)
    modes = side.get("default_modes") or {}
    imgsz = side.get("imgsz") or info["trained_imgsz"] or 640
    if isinstance(imgsz, (list, tuple)):
        imgsz = max(imgsz)
    return {"id": str(path), "name": path.stem, "folder": str(path.parent), **info,
            "default_modes": {name: modes.get(name) if modes.get(name) in MODES else "ignore"
                              for name in info["classes"].values()},
            "enabled_by_default": bool(side.get("enabled", False)),
            "imgsz": int(imgsz), "description": side.get("description") or path.stem}


def list_models():
    """Full info for every model in every folder (loads them all: tests and tools only)."""
    return [model_info(m["id"]) for f in list_files() for m in f["models"]]


def sam_path():
    return SAM_DIR / SAM_WEIGHTS


def ensure_sam():
    """Download the SAM checkpoint once (about 40 MB for MobileSAM)."""
    target = sam_path()
    if target.is_file():
        return target
    SAM_DIR.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(SAM_URL.format(SAM_WEIGHTS), timeout=120) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.replace(target)
    return target


def _load_sam():
    path = ensure_sam()
    if path not in _SAM:
        from ultralytics import SAM
        _threads()
        _SAM.clear()
        _SAM[path] = SAM(str(path))
    return _SAM[path]


def _boxes(result):
    """xyxy boxes, class ids and confidences for detect / segment / obb results."""
    holder = result.obb if getattr(result, "obb", None) is not None else result.boxes
    if holder is None or len(holder) == 0:
        return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)
    return (holder.xyxy.cpu().numpy().astype(float), holder.cls.cpu().numpy().astype(int),
            holder.conf.cpu().numpy().astype(float))


def _box_mask(box, h, w):
    m = np.zeros((h, w), bool)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    m[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    return m


def detect(img, model_ids, conf=0.35, use_sam=True, roi_box=None):
    """img: PIL RGB (already EXIF-oriented). Returns (detections, instance PNG bytes, info).

    roi_box: (x0, y0, x1, y1) photo pixels of the operator's ROI, or None for the whole photo.
    With a ROI, YOLO and SAM only see the ROI (objects outside it are not searched, and
    small objects are seen at a higher resolution); the results are put back in photo
    coordinates. The PNG has the photo's size; pixel value n (1..254) = detection n,
    0 = nothing. Larger objects are painted first, so a small object inside a large one
    stays visible.
    """
    if not 0.01 <= conf <= 0.99:
        raise ValueError("Confidenza fuori intervallo (0,01-0,99)")
    available = {}
    for model_id in model_ids:
        try:
            info = model_info(model_id)
        except KeyError:
            raise ValueError("Modello non trovato nelle cartelle configurate: " + str(model_id))
        if info["error"]:
            raise ValueError(f"Modello {info['name']} non utilizzabile: {info['error']}")
        available[info["id"]] = info
    model_ids = list(available)
    full_w, full_h = img.size
    ox, oy = (int(roi_box[0]), int(roi_box[1])) if roi_box else (0, 0)
    if roi_box:
        img = img.crop(tuple(int(v) for v in roi_box))
    w, h = img.size
    t0 = time.perf_counter()
    found = []  # (model_id, class_name, conf, box, mask or None)
    with LOCK:
        for model_id in model_ids:
            meta = available[model_id]
            m = _load_yolo(model_id)
            result = m.predict(img, imgsz=meta["imgsz"], conf=conf, iou=0.5, max_det=MAX_OBJECTS,
                               device="cpu", verbose=False)[0]
            boxes, classes, confs = _boxes(result)
            masks = None
            if getattr(result, "masks", None) is not None and len(boxes):
                data = result.masks.data.cpu().numpy() > 0.5
                masks = [np.asarray(Image.fromarray(x).resize((w, h), Image.Resampling.NEAREST)) for x in data]
            for i in range(len(boxes)):
                found.append([model_id, meta["classes"].get(int(classes[i]), str(classes[i])), float(confs[i]),
                              boxes[i], None if masks is None else masks[i]])
        found.sort(key=lambda f: -f[2])
        found = found[:MAX_OBJECTS]
        t_yolo = time.perf_counter() - t0
        need_sam = [i for i, f in enumerate(found) if f[4] is None]
        sam_used = None
        if need_sam and use_sam:
            sam = _load_sam()
            sam_used = sam_path().name
            res = sam(img, bboxes=[found[i][3].tolist() for i in need_sam], verbose=False)[0]
            data = res.masks.data.cpu().numpy() > 0.5
            for k, i in enumerate(need_sam):
                mask = data[k]
                if mask.shape != (h, w):
                    mask = np.asarray(Image.fromarray(mask).resize((w, h), Image.Resampling.NEAREST))
                # SAM may leak outside the prompt box; the object cannot extend beyond it.
                found[i][4] = mask & _box_mask(_grow(found[i][3], 0.03), h, w)
    t_total = time.perf_counter() - t0
    instance = np.zeros((h, w), np.uint8)
    detections = []
    for n, (model_id, name, c, box, mask) in enumerate(found, start=1):
        source = "sam" if n - 1 in need_sam else "yolo-seg"
        if mask is None or not mask.any():
            mask, source = _box_mask(box, h, w), "box"
        found[n - 1][4] = mask
        detections.append({"n": n, "model": model_id, "model_name": available[model_id]["name"], "class_name": name, "confidence": round(c, 3),
                           "box_px": [int(round(v)) for v in box], "mask_source": source,
                           "pixels": int(mask.sum())})
    for n in sorted(range(1, len(found) + 1), key=lambda n: -detections[n - 1]["pixels"]):
        instance[found[n - 1][4]] = n
    if roi_box:  # back to photo coordinates
        whole = np.zeros((full_h, full_w), np.uint8)
        whole[oy:oy + h, ox:ox + w] = instance
        instance = whole
        for d in detections:
            b = d["box_px"]
            d["box_px"] = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
    buf = io.BytesIO()
    Image.fromarray(instance, mode="L").save(buf, format="PNG", optimize=True)
    info = {"models": [available[m]["name"] for m in model_ids], "model_paths": list(model_ids), "confidence_threshold": conf, "sam_model": sam_used,
            "seconds_detection": round(t_yolo, 2), "seconds_total": round(t_total, 2),
            "area": "roi" if roi_box else "intera foto", "area_px": [ox, oy, ox + w, oy + h]}
    return detections, buf.getvalue(), info


def _grow(box, fraction):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * fraction + 4, (y1 - y0) * fraction + 4
    return [x0 - dx, y0 - dy, x1 + dx, y1 + dy]


def parse_rules(text, n_max):
    """[{"n":1,"mode":"exclude", ...}] -> {n: mode}. Unknown extra keys are kept for the report."""
    try:
        items = json.loads(text or "[]")
        if not isinstance(items, list) or len(items) > MAX_OBJECTS:
            raise ValueError()
        rules = {}
        for item in items:
            n, mode = int(item["n"]), item["mode"]
            if not 1 <= n <= n_max or mode not in MODES:
                raise ValueError()
            rules[n] = item
        return rules
    except (ValueError, TypeError, KeyError):
        raise ValueError("Regole oggetti non valide")


def object_masks(instance, rules, box, size):
    """instance: uint8 HxW (photo size). Returns (exclude, focus) bool masks cropped to
    the ROI `box` and resized to `size` (w, h), and per-detection pixels in that grid."""
    crop = Image.fromarray(instance).crop(box).resize(size, Image.Resampling.NEAREST)
    ids = np.asarray(crop)
    exclude_ids = [n for n, r in rules.items() if r["mode"] == "exclude"]
    focus_ids = [n for n, r in rules.items() if r["mode"] == "focus"]
    exclude = np.isin(ids, exclude_ids) if exclude_ids else np.zeros(ids.shape, bool)
    focus = np.isin(ids, focus_ids) if focus_ids else None
    counts = np.bincount(ids.ravel(), minlength=256)
    return exclude, focus, {n: int(counts[n]) for n in rules}
