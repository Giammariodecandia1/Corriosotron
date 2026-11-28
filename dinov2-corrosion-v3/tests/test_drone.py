import dataclasses
import io
import math
import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FOTO = Path(r"C:\Users\gdecandia\Desktop\Oggi2\tralicci corrosione")
CAM4 = FOTO / "Camera_4_2026-07-14_11.01.03.000.jpg"
CAM6 = FOTO / "Camera_6_2026-05-08_10.04.57.000.jpg"
HAS_PHOTOS = CAM4.exists() and CAM6.exists()


def leggi(path, size=None):
    from misura import drone
    data = path.read_bytes()
    if size is None:
        with Image.open(io.BytesIO(data)) as im:
            size = im.size
    return drone.leggi_foto(data, size)


@unittest.skipUnless(HAS_PHOTOS, "foto reali non disponibili")
class FotoRealiTest(unittest.TestCase):
    def test_m3t_camera4(self):
        from misura import drone
        m = leggi(CAM4)
        self.assertEqual((m.make, m.model), ("DJI", "M3T"))
        self.assertAlmostEqual(m.focal_mm, 4.4, places=3)
        self.assertEqual(m.rtk_flag, 50)
        self.assertAlmostEqual(m.rel_alt, 35.174, places=3)
        self.assertAlmostEqual(m.gimbal_pitch, -46.1, places=3)
        self.assertEqual(m.lrf_m, 0.0)
        self.assertIsNotNone(m.lat)
        self.assertIsNotNone(m.lon)
        p = drone.scegli_profilo(m)
        self.assertEqual(p.id, "dji_m3t")
        self.assertFalse(p.generico)
        self.assertEqual(drone.misurabile(m, p)[0], True)

    def test_h30t_camera6(self):
        from misura import drone
        m = leggi(CAM6)
        self.assertEqual(m.model, "H30T")
        self.assertEqual(m.image_source, "WideCamera")
        self.assertAlmostEqual(m.lrf_m, 30.234, places=3)
        self.assertEqual(drone.scegli_profilo(m).id, "dji_h30t")

    def test_intrinseci_35mm_e_ridimensionata(self):
        from misura import drone
        full = leggi(CAM4, (8000, 6000))
        half = leggi(CAM4, (4000, 3000))
        p = drone.scegli_profilo(full)
        k_full = drone.intrinseci(full, p)
        k_half = drone.intrinseci(half, p)
        atteso = 24 * 10000 / 43.267
        self.assertAlmostEqual(k_full["fx"] / atteso, 1.0, delta=0.005)
        self.assertAlmostEqual(k_half["fx"] * 2 / k_full["fx"], 1.0, places=9)
        self.assertEqual(k_full["fonte"], "35mm")
        self.assertEqual(k_half["fonte"], "35mm")
        self.assertAlmostEqual(k_half["cx"], 2000.0)
        self.assertAlmostEqual(k_half["cy"], 1500.0)

    def test_zoom_non_misurabile(self):
        from misura import drone
        m = leggi(CAM4)
        p = drone.scegli_profilo(m)
        for variante in (dataclasses.replace(m, focal_mm=4.4 * 1.2), dataclasses.replace(m, digital_zoom=2.0)):
            ok, msg = drone.misurabile(variante, p)
            self.assertFalse(ok)
            self.assertIn("zoom", msg.lower())

    def test_termica_e_orientamento(self):
        from misura import drone
        m = leggi(CAM4)
        p = drone.scegli_profilo(m)
        ok, msg = drone.misurabile(dataclasses.replace(m, image_source="InfraredCamera"), p)
        self.assertFalse(ok)
        self.assertIn("termica", msg)
        ok, msg = drone.misurabile(dataclasses.replace(m, orientation=6), p)
        self.assertFalse(ok)
        self.assertIn("ruotata", msg)

    def test_drone_sconosciuto(self):
        from misura import drone
        m = dataclasses.replace(leggi(CAM4), make="Acme", model="X1")
        p = drone.scegli_profilo(m)
        self.assertTrue(p.generico)
        self.assertEqual(drone.misurabile(m, p)[0], True)


class MetadatiMancantiTest(unittest.TestCase):
    def test_jpeg_senza_exif(self):
        from misura import drone
        buf = io.BytesIO()
        Image.new("RGB", (640, 480), (90, 90, 90)).save(buf, format="JPEG")
        m = drone.leggi_foto(buf.getvalue(), (640, 480))
        self.assertIsNone(m.gimbal_pitch)
        p = drone.scegli_profilo(m)
        self.assertTrue(p.generico)
        ok, msg = drone.misurabile(m, p)
        self.assertFalse(ok)
        self.assertIn("metadati", msg)

    def test_formula_35mm_diagonale(self):
        from misura import drone
        m = drone.FotoMeta(width=4000, height=3000, focal35_mm=24.0)
        k = drone.intrinseci(m, drone.Profilo.generico_default())
        self.assertAlmostEqual(k["fx"], 24 * math.hypot(4000, 3000) / 43.267, places=6)


if __name__ == "__main__":
    unittest.main()
