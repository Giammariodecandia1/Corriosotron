"""Generate an alpha mask using the isolated, CPU-only rembg/BiRefNet install."""
import sys
from pathlib import Path

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
