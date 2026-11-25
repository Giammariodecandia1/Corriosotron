"""Local full-size cutouts for visually checking reticular tower masks."""

import gc
import hashlib
import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps
from rembg import new_session, remove


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "model_cache" / "sfondo_rimosso"
MODEL_NAME = "birefnet-general"


def select_photos() -> list[Path]:
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askopenfilenames(
            parent=root,
            title="Seleziona una o più foto del sostegno",
            filetypes=[("Fotografie", "*.jpg *.jpeg *.png *.webp"), ("Tutti i file", "*.*")],
        )
    finally:
        root.destroy()
    return [Path(p) for p in selected]


def output_stem(source: Path) -> str:
    path_id = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{source.stem}-{path_id}"


def process_one(source: Path, session) -> tuple[Path, Path, float]:
    started = time.perf_counter()
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    cutout = remove(image, session=session)
    if not isinstance(cutout, Image.Image) or cutout.size != image.size:
        raise RuntimeError("Il modello non ha restituito una foto della dimensione originale")
    cutout = cutout.convert("RGBA")
    alpha = cutout.getchannel("A")
    stem = output_stem(source)
    cutout_path = OUTPUT_DIR / f"{stem}-trasparente.png"
    mask_path = OUTPUT_DIR / f"{stem}-maschera.png"
    cutout.save(cutout_path)
    alpha.save(mask_path)
    image.close()
    cutout.close()
    alpha.close()
    gc.collect()
    return cutout_path, mask_path, time.perf_counter() - started


def main() -> int:
    photos = select_photos() if len(sys.argv) == 1 else [Path(p) for p in sys.argv[1:]]
    if not photos:
        print("Nessuna foto selezionata.")
        return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Carico BiRefNet in locale sulla CPU. Le foto non vengono inviate online.", flush=True)
    session = new_session(MODEL_NAME, providers=["CPUExecutionProvider"])
    failed = 0
    for number, source in enumerate(photos, 1):
        print(f"[{number}/{len(photos)}] {source.name}", flush=True)
        try:
            cutout_path, mask_path, seconds = process_one(source, session)
            print(f"  PNG: {cutout_path.name}", flush=True)
            print(f"  Maschera: {mask_path.name} | {seconds:.1f} s", flush=True)
        except Exception as exc:
            failed += 1
            print(f"  ERRORE: {exc}", flush=True)
    if os.name == "nt" and os.getenv("SEG_NO_OPEN") != "1":
        os.startfile(OUTPUT_DIR)
    print(f"Completate: {len(photos) - failed}/{len(photos)}. Risultati: {OUTPUT_DIR}")
    print("Controllare che i vuoti del reticolo siano trasparenti e che le aste restino visibili.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
