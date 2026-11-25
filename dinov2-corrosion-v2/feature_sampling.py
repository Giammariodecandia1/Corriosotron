"""Local DINO feature tiling and exemplar scoring; no alternative visual model."""
import math
import numpy as np
from PIL import Image

TILED_MODE = "tiles_v1"
SOURCE_TILE = 784
MAX_TILES = 64
MAX_SAMPLES = 2048


def tile_plan(width, height, size):
    scale = min(1., size / SOURCE_TILE)
    side = size // 14
    stride = max(1, side * 3 // 4)
    def starts(n):
        if n <= side:
            return [0]
        return sorted(set(list(range(0, n - side + 1, stride)) + [n - side]))
    # Keep CPU work bounded on full-resolution photographs. The report records
    # source_pixels_per_patch so the effective detail remains visible.
    for _ in range(32):
        gw, gh = max(1, math.ceil(width * scale / 14)), max(1, math.ceil(height * scale / 14))
        xs, ys = starts(gw), starts(gh)
        if len(xs) * len(ys) <= MAX_TILES:
            break
        scale *= .85
    else:
        raise ValueError("Impossibile pianificare i riquadri entro il limite CPU")
    return gw, gh, side, xs, ys


def tiled_features(img, box, size, extract):
    """Patch-aligned overlapping windows; edge-weighted fusion, bounded memory."""
    crop = img.crop(box)
    gw, gh, side, xs, ys = tile_plan(*crop.size, size)
    resized = crop.resize((gw * 14, gh * 14), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (max(gw, side)*14, max(gh, side)*14), (128,128,128))
    canvas.paste(resized, (0,0))
    accum = np.zeros((max(gh, side), max(gw, side), 384), np.float32)
    weights = np.zeros(accum.shape[:2], np.float32)
    ramp = np.minimum(np.arange(side)+1, np.arange(side,0,-1)).astype(np.float32)
    window = np.outer(ramp, ramp); window /= window.max()
    seconds = 0.
    for y in ys:
        for x in xs:
            vec, tile, _, elapsed = extract(canvas, (x*14,y*14,(x+side)*14,(y+side)*14), size)
            if vec.shape != (side, side, 384) or not np.isfinite(vec).all():
                raise ValueError("Embedding DINOv2 non valido")
            accum[y:y+side,x:x+side] += vec * window[...,None]
            weights[y:y+side,x:x+side] += window
            seconds += elapsed
            tile.close()
    vec = accum[:gh,:gw] / weights[:gh,:gw,None]
    vec /= np.maximum(np.linalg.norm(vec,axis=-1,keepdims=True), 1e-8)
    canvas.close(); resized.close()
    geo = {"left":0,"top":0,"rw":gw*14,"rh":gh*14,"side":size,"box":box,
           "source_width":img.width,"source_height":img.height,"feature_mode":TILED_MODE,
           "tile_count":len(xs)*len(ys),"grid_dimensions":[gw,gh],
           "source_pixels_per_patch":[crop.width/gw,crop.height/gh]}
    return vec, crop, geo, round(seconds,2)


def class_scores(vec, profile, labels):
    names = [k for k in labels if k in profile["prototypes"]]
    if profile.get("schema_version") != 2:
        proto = np.asarray([profile["prototypes"][k] for k in names], np.float32)
        proto /= np.maximum(np.linalg.norm(proto,axis=1,keepdims=True),1e-8)
        return names, np.einsum("hwc,kc->khw",vec,proto)
    # Equal class weight: mean of up to three closest examples within each class.
    # Chunk query patches to bound allocations as the multi-photo bank grows.
    flat = vec.reshape(-1,vec.shape[-1])
    scores = np.empty((len(names),len(flat)),np.float32)
    for i,label in enumerate(names):
        bank = np.asarray([s["embedding"] for s in profile["samples"] if s["label"]==label],np.float32)
        bank /= np.maximum(np.linalg.norm(bank,axis=1,keepdims=True),1e-8)
        k = min(3,len(bank))
        for start in range(0,len(flat),512):
            similarities = flat[start:start+512] @ bank.T
            scores[i,start:start+512] = np.partition(similarities, -k, axis=1)[:,-k:].mean(axis=1)
    return names, scores.reshape(len(names),*vec.shape[:2])


def merge_samples(profile, additions):
    """Idempotent by image/ROI/extraction/patch; conflicting labels never overwrite."""
    known = {s["sample_id"]:s for s in profile.get("samples",[])}
    added = 0
    for sample in additions:
        previous = known.get(sample["sample_id"])
        if previous and previous["label"] != sample["label"]:
            raise ValueError("Campione già presente con una classe diversa: creare un profilo corretto")
        if not previous:
            known[sample["sample_id"]] = sample
            added += 1
        elif previous.get("color_lab") is None and sample.get("color_lab") is not None:
            # Same patch marked again on the same photo: an older sample gains its colour.
            previous["color_lab"] = sample["color_lab"]
            added += 1
    if len(known)>MAX_SAMPLES:
        raise ValueError(f"Massimo {MAX_SAMPLES} campioni per profilo")
    profile["samples"] = list(known.values())
    return added
