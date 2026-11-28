import dataclasses
import io
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FOTO = Path(r"C:\Users\gdecandia\Desktop\Oggi2\tralicci corrosione")
CAM4 = FOTO / "Camera_4_2026-07-14_11.01.03.000.jpg"
CAM6 = FOTO / "Camera_6_2026-05-08_10.04.57.000.jpg"
HAS_PHOTOS = CAM4.exists() and CAM6.exists()


def camera_frontale(fx=2000.0):
    from misura.distix.camera_model import CameraModel
    return CameraModel("t", np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
                       image_width=4000, image_height=3000, fx_pix=fx, fy_pix=fx, cx_pix=2000.0, cy_pix=1500.0)


def leggi(path):
    from misura import drone
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        size = im.size
    m = drone.leggi_foto(data, size)
    return m, drone.scegli_profilo(m)


class GeometriaTest(unittest.TestCase):
    def test_raggi_coincidono_con_distix(self):
        from misura import geometria
        cam = camera_frontale()
        cam.set_rotation_from_delta(7.0, -20.0, 3.0)
        rng = np.random.default_rng(1)
        u, v = rng.uniform(0, 4000, 20), rng.uniform(0, 3000, 20)
        dirs = geometria.raggi(cam, u, v)
        for i in range(20):
            _, d = cam.pixel_to_ray(u[i], v[i])
            np.testing.assert_allclose(dirs[i], d, atol=1e-9)

    def test_frontale_area_pixel(self):
        from misura import geometria
        cam = camera_frontale()
        piano = geometria.Piano(np.array([0.0, 10.0, 0.0]), np.array([0.0, -1.0, 0.0]))
        pr = geometria.Proiettore(cam=cam, piano=piano, etichetta="STIMA", campioni=[], info={})
        p = pr.punti(np.array([1950.0, 2050.0, 1950.0]), np.array([1450.0, 1450.0, 1550.0]))
        area = np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
        self.assertAlmostEqual(area, 0.25, delta=0.25 * 0.005)

    def test_interseca_dietro_camera_nan(self):
        from misura import geometria
        out = geometria.interseca_piano(np.zeros(3), np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]),
                                        np.array([0.0, 10.0, 0.0]), np.array([0.0, -1.0, 0.0]))
        self.assertTrue(np.isnan(out).all())


@unittest.skipUnless(HAS_PHOTOS, "foto reali non disponibili")
class StimaFotoTest(unittest.TestCase):
    def scala_mm(self, pr, u, v):
        p = pr.punti(np.array([u, u + 1, u]), np.array([v, v, v + 1]))
        return 1000 * np.sqrt(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])))

    def test_camera4_scala_varia(self):
        from misura import geometria, stima_foto
        m, p = leggi(CAM4)
        pr = stima_foto.costruisci(m, p, [3000, 300, 5000, 5700], None)
        self.assertIsInstance(pr, geometria.Proiettore)
        self.assertEqual(pr.info["fonte_distanza"], "geometria")
        self.assertEqual(pr.etichetta, "STIMA")
        self.assertGreater(self.scala_mm(pr, 4000, 5690), 2 * self.scala_mm(pr, 4000, 310))
        self.assertTrue(5 < pr.info["distanza_m"] < 40, pr.info["distanza_m"])
        self.assertEqual(len(pr.campioni), 200)
        self.assertIn("M3T", pr.info["drone"])

    def test_h30t_usa_laser(self):
        from misura import geometria, stima_foto
        m, p = leggi(CAM6)
        # base fuori foto: niente confronto con la geometria, vale il laser (il disaccordo è in test_revisione)
        pr = stima_foto.costruisci(m, p, [3000, 500, 5064, 6048], None)
        self.assertIsInstance(pr, geometria.Proiettore)
        self.assertEqual(pr.info["fonte_distanza"], "laser")

    def test_base_non_visibile(self):
        from misura import stima_foto
        m, p = leggi(CAM4)
        out = stima_foto.costruisci(m, p, [3000, 300, 5000, 6000], None)
        self.assertIsInstance(out, stima_foto.NessunaStima)
        self.assertIn("base", out.motivo)

    def test_senza_box_e_senza_base(self):
        from misura import stima_foto
        m, p = leggi(CAM4)
        self.assertIsInstance(stima_foto.costruisci(m, p, None, None), stima_foto.NessunaStima)

    def test_raggio_sopra_orizzonte(self):
        from misura import stima_foto
        m, p = leggi(CAM4)
        out = stima_foto.costruisci(dataclasses.replace(m, gimbal_pitch=5.0), p, None, (4000, 1000))
        self.assertIsInstance(out, stima_foto.NessunaStima)

    def test_base_quasi_orizzonte_niente_distanze_assurde(self):
        from misura import drone, stima_foto
        m, p = leggi(CAM4)
        k = drone.intrinseci(m, p)
        # piede 2° sotto l'orizzonte: distanza orizzontale ~1000 m, non plausibile per un sostegno
        v = k["cy"] + k["fy"] * np.tan(np.deg2rad(2.0))
        out = stima_foto.costruisci(dataclasses.replace(m, gimbal_pitch=0.0, gimbal_roll=0.0), p, None, (k["cx"], v))
        self.assertIsInstance(out, stima_foto.NessunaStima)
        self.assertIn("distanza", out.motivo)

    def test_metadati_mancanti(self):
        from misura import drone, stima_foto
        m, p = leggi(CAM4)
        rotta = dataclasses.replace(m, gimbal_yaw=None)
        out = stima_foto.costruisci(rotta, p, [3000, 300, 5000, 5700], None)
        self.assertIsInstance(out, stima_foto.NessunaStima)
        self.assertEqual(out.motivo, drone.misurabile(rotta, p)[1])

    def test_laser_a_zero_ignorato(self):
        from misura import stima_foto
        m, p = leggi(CAM4)
        self.assertEqual(m.lrf_m, 0.0)
        pr = stima_foto.costruisci(m, p, [3000, 300, 5000, 5700], None)
        self.assertEqual(pr.info["fonte_distanza"], "geometria")


if __name__ == "__main__":
    unittest.main()
