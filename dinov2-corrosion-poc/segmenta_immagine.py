"""Generate an alpha mask using the isolated, CPU-only rembg/BiRefNet install."""
import sys
from pathlib import Path

from PIL import Image
from rembg import new_session, remove


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Uso: segmenta_immagine.py FOTO_PNG MASCHERA_PNG")
    source, target = map(Path, sys.argv[1:])
    session = new_session("birefnet-general", providers=["CPUExecutionProvider"])
    with Image.open(source) as original:
        expected_size = original.size
        cutout = remove(original.convert("RGB"), session=session)
    if not isinstance(cutout, Image.Image) or cutout.size != expected_size:
        raise RuntimeError("BiRefNet non ha restituito una maschera con dimensioni corrette")
    alpha = cutout.convert("RGBA").getchannel("A")
    alpha.save(target, format="PNG", optimize=True)
    alpha.close()
    cutout.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
