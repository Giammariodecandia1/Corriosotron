"""Add the colour of each labelled patch to an existing profile, from the original photos.

    python aggiorna_profilo_colore.py CARTELLA_FOTO [ID_PROFILO ...]

Older profiles store only DINOv2 vectors. The colour classifier needs the Lab colour
of the metal under each marker; this script recovers it from the photos used for the
calibration (matched by SHA-256, so the files must be the unmodified originals).
The profile is copied to *.json.bak before being rewritten. Photos are only read.
Without profile IDs every profile in local_profiles is processed.
"""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import main
import color_evidence as ce
from feature_sampling import TILED_MODE, tile_plan


def upgrade(path, photos):
    profile = json.loads(path.read_text("utf-8"))
    samples = profile.get("samples") or []
    todo = [s for s in samples if s.get("color_lab") is None]
    if not todo:
        return f"{profile.get('profile_name')}: già aggiornato"
    filled = missing_photo = off_metal = 0
    groups = {}
    for s in todo:
        groups.setdefault((s["source_sha256"], tuple(s["roi"])), []).append(s)
    for (sha, box), items in groups.items():
        photo = photos.get(sha)
        cfg = items[0].get("extraction", {})
        if photo is None or cfg.get("mode") != TILED_MODE:
            missing_photo += len(items)
            continue
        with Image.open(photo) as source:
            img = ImageOps.exif_transpose(source).convert("RGB")
        metal, reason = main.make_fine_mask(img, box, sha)
        if metal is None:
            raise SystemExit(f"Maschera di dettaglio non riuscita per {photo.name}: {reason}")
        rgb = np.asarray(img.crop(box))
        gw, gh, *_ = tile_plan(rgb.shape[1], rgb.shape[0], cfg["resolution"])
        pw, ph = rgb.shape[1] / gw, rgb.shape[0] / gh
        for s in items:
            r, c = s["patch_yx"]
            lab = ce.sample_color_box(rgb, metal, c * pw, r * ph, (c + 1) * pw + 1, (r + 1) * ph + 1)
            if lab is None:
                off_metal += 1
            else:
                s["color_lab"] = [round(float(v), 3) for v in lab]
                filled += 1
    if filled:
        shutil.copy2(path, path.with_suffix(".json.bak"))
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(profile, ensure_ascii=False), "utf-8")
        tmp.replace(path)
    return (f"{profile.get('profile_name')}: {filled} campioni con colore, {off_metal} fuori dal metallo, "
            f"{missing_photo} senza foto originale (o profilo Rapido)")


def run():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    folder = Path(sys.argv[1])
    photos = {}
    for f in folder.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            photos[hashlib.sha256(f.read_bytes()).hexdigest()] = f
    paths = [main.profile_file(i) for i in sys.argv[2:]] or sorted(main.PROFILE_DIR.glob("*.json"))
    for path in paths:
        print(upgrade(path, photos), flush=True)


if __name__ == "__main__":
    run()
