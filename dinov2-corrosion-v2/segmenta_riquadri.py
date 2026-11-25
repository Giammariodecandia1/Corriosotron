"""Native-resolution foreground mask for a ROI: BiRefNet on overlapping square tiles.

BiRefNet resizes every input to about 1024 px. On a whole 12 MP photo a lattice
member a few pixels wide disappears into a blob that also covers the ground seen
through the lattice. Running the same model on ROI tiles keeps the members sharp.
Runs in the isolated CPU-only rembg environment, one model load for all tiles.
"""
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
    if len(sys.argv) != 7:
        raise SystemExit("Uso: segmenta_riquadri.py FOTO X0 Y0 X1 Y1 MASCHERA_ROI_PNG")
    source, target = Path(sys.argv[1]), Path(sys.argv[6])
    box = tuple(int(v) for v in sys.argv[2:6])
    session = new_session("birefnet-general", providers=["CPUExecutionProvider"])
    with Image.open(source) as original:
        crop = original.convert("RGB").crop(box)
    width, height = crop.size
    side, xs, ys = plan(width, height)
    total = np.zeros((height, width), np.float32)
    weight = np.zeros((height, width), np.float32)
    for y in ys:
        for x in xs:
            tile = crop.crop((x, y, min(width, x + side), min(height, y + side)))
            cutout = remove(tile, session=session)
            alpha = np.asarray(cutout.convert("RGBA").getchannel("A"), np.float32)
            if alpha.shape != (tile.height, tile.width):
                raise RuntimeError("BiRefNet non ha restituito un riquadro con dimensioni corrette")
            # Trust the tile centre more than its borders, where context is missing.
            ry = np.minimum(np.arange(tile.height) + 1, np.arange(tile.height, 0, -1))
            rx = np.minimum(np.arange(tile.width) + 1, np.arange(tile.width, 0, -1))
            w = np.minimum.outer(ry, rx).astype(np.float32)
            total[y:y + tile.height, x:x + tile.width] += alpha * w
            weight[y:y + tile.height, x:x + tile.width] += w
    mask = np.clip(total / np.maximum(weight, 1e-6), 0, 255).astype(np.uint8)
    Image.fromarray(mask, "L").save(target, format="PNG", optimize=True)
    print(f"tiles={len(xs) * len(ys)} side={side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
