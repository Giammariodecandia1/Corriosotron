import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def camera_frontale(fx=2000.0):
    from misura.distix.camera_model import CameraModel
    return CameraModel("t", np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
                       image_width=4000, image_height=3000, fx_pix=fx, fy_pix=fx, cx_pix=2000.0, cy_pix=1500.0)


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
        pr = geometria.Proiettore(cam=cam, piano=piano, etichetta="VALIDATA", campioni=[], info={})
        p = pr.punti(np.array([1950.0, 2050.0, 1950.0]), np.array([1450.0, 1450.0, 1550.0]))
        area = np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
        self.assertAlmostEqual(area, 0.25, delta=0.25 * 0.005)

    def test_interseca_dietro_camera_nan(self):
        from misura import geometria
        out = geometria.interseca_piano(np.zeros(3), np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]),
                                        np.array([0.0, 10.0, 0.0]), np.array([0.0, -1.0, 0.0]))
        self.assertTrue(np.isnan(out).all())


if __name__ == "__main__":
    unittest.main()
