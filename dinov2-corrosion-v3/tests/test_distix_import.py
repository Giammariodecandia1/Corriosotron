import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def camera(name, x):
    from misura.distix.camera_model import CameraModel
    return CameraModel(name, np.array([x, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
                       image_width=4000, image_height=3000, fx_pix=2800.0, fy_pix=2800.0,
                       cx_pix=2000.0, cy_pix=1500.0)


class DistixImportTest(unittest.TestCase):
    def test_triangola_punto_sintetico(self):
        from misura.distix.triangulation import triangulate_from_cameras
        target = np.array([5.0, 30.0, 2.0])
        cams = [camera("A", 0.0), camera("B", 10.0)]
        obs = []
        for cam in cams:
            u, v, _ = cam.project_point(target)
            obs.append((cam, (u, v)))
        res = triangulate_from_cameras(obs)
        self.assertTrue(np.allclose(res.point_3d, target, atol=1e-6), res.point_3d)
        self.assertTrue(res.all_in_front)

    def test_moduli_importabili(self):
        from misura.distix import (bundle_adjustment, calibration_manager, coordinate_system,  # noqa: F401
                                   metadata_reader, rtk_validator, uncertainty_engine)
        self.assertTrue(callable(bundle_adjustment.run_constrained_bundle_adjustment_with_diagnostics))
        self.assertTrue(callable(metadata_reader.dji_euler_to_vectors))
        self.assertTrue(callable(rtk_validator.evaluate_dji_xmp_gnss))


if __name__ == "__main__":
    unittest.main()
