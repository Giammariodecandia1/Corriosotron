"""Native-resolution foreground mask for a ROI: BiRefNet on overlapping square tiles.

BiRefNet resizes every input to about 1024 px. On a whole 12 MP photo a lattice
member a few pixels wide disappears into a blob that also covers the ground seen
through the lattice. Running the same model on ROI tiles keeps the members sharp.
Runs in the isolated CPU-only rembg environment, one model load for all tiles.
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import new_session, remove


def _stop_with_server():
    """Exit if the v2 server that started this worker is gone (closed or restarted), so a
    BiRefNet run never keeps the CPU busy for an analysis nobody is waiting for."""
    import ctypes, os, threading, time
    pid = int(os.environ.get("V2_SERVER_PID", "0") or 0)
    if not pid or os.name != "nt":
        return
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        os._exit(3)

    def watch():
        while kernel32.WaitForSingleObject(handle, 0) != 0:  # 0 = process ended
            time.sleep(2)
        os._exit(3)
    threading.Thread(target=watch, daemon=True).start()


_stop_with_server()

MIN_SIDE, MAX_SIDE, MAX_TILES = 512, 1024, 24


def starts(length, side, overlap):
    if length <= side:
        return [0]
    step = side - overlap
    return sorted(set(list(range(0, length - side, step)) + [length - side]))


def plan(width, height):
    """Square tiles no larger than the model input, so nothing is downscaled."""
    side = int(min(MAX_SIDE, max(MIN_SIDE, min(width, height))))
    for _ in range(16):
        xs, ys = starts(width, side, side // 6), starts(height, side, side // 6)
        if len(xs) * len(ys) <= MAX_TILES:
            return side, xs, ys
        side = int(side * 1.25)  # very large ROI: accept coarser tiles, bounded CPU time
    raise SystemExit("ROI troppo grande per la maschera a riquadri")


def main() -> int:
    if len(sys.argv) not in (7, 8):
        raise SystemExit("Uso: segmenta_riquadri.py FOTO X0 Y0 X1 Y1 MASCHERA_ROI_PNG [PIXEL_UTILI_PNG]")
    source, target = Path(sys.argv[1]), Path(sys.argv[6])
    box = tuple(int(v) for v in sys.argv[2:6])
    # Optional ROI-size mask of the pixels the analysis will use (inside "Analizza solo qui", outside
    # excluded objects). Tiles without any of them are not computed: their result would be discarded.
    need = None
    if len(sys.argv) == 8:
        with Image.open(sys.argv[7]) as m:
            need = np.asarray(m.convert("L")) >= 128
    # No CPU memory arena: it kept ~13 GB reserved on 1024 px tiles and failed with "bad allocation"
    # when other jobs (training, Maskatron) used the RAM.
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    session = new_session("birefnet-general", sess_opts=opts, providers=["CPUExecutionProvider"])
    with Image.open(source) as original:
        crop = original.convert("RGB").crop(box)
    width, height = crop.size
    side, xs, ys = plan(width, height)
    total = np.zeros((height, width), np.float32)
    weight = np.zeros((height, width), np.float32)
    # Finished tiles are kept on disk (V2_TILE_CACHE) so an interrupted run resumes instead of starting over.
    cache = Path(os.environ["V2_TILE_CACHE"]) if os.environ.get("V2_TILE_CACHE") else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    if need is not None and need.shape != (height, width):
        raise SystemExit("La maschera dei pixel utili non ha le dimensioni della ROI")
    margin = 32  # px: a needed pixel just outside a tile still keeps that tile
    done = skipped = 0
    for y in ys:
        for x in xs:
            if need is not None and not need[max(0, y - margin):y + side + margin, max(0, x - margin):x + side + margin].any():
                done += 1; skipped += 1
                print(f"tile {done}/{len(xs) * len(ys)} skipped", flush=True)
                continue
            tile = crop.crop((x, y, min(width, x + side), min(height, y + side)))
            saved = cache / f"tile_{side}_{x}_{y}.png" if cache else None
            alpha = None
            if saved and saved.is_file():
                try:
                    with Image.open(saved) as old:
                        alpha = np.asarray(old.convert("L"), np.float32)
                    if alpha.shape != (tile.height, tile.width):
                        alpha = None
                except OSError:
                    alpha = None
            if alpha is None:
                cutout = remove(tile, session=session)
                alpha = np.asarray(cutout.convert("RGBA").getchannel("A"), np.float32)
                if alpha.shape != (tile.height, tile.width):
                    raise RuntimeError("BiRefNet non ha restituito un riquadro con dimensioni corrette")
                if saved:
                    part = saved.with_suffix(".part")
                    Image.fromarray(alpha.astype(np.uint8), "L").save(part, format="PNG")
                    os.replace(part, saved)
            done += 1
            print(f"tile {done}/{len(xs) * len(ys)}", flush=True)
            # Trust the tile centre more than its borders, where context is missing.
            ry = np.minimum(np.arange(tile.height) + 1, np.arange(tile.height, 0, -1))
            rx = np.minimum(np.arange(tile.width) + 1, np.arange(tile.width, 0, -1))
            w = np.minimum.outer(ry, rx).astype(np.float32)
            total[y:y + tile.height, x:x + tile.width] += alpha * w
            weight[y:y + tile.height, x:x + tile.width] += w
    mask = np.clip(total / np.maximum(weight, 1e-6), 0, 255).astype(np.uint8)
    Image.fromarray(mask, "L").save(target, format="PNG", optimize=True)
    print(f"tiles={len(xs) * len(ys)} skipped={skipped} side={side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
