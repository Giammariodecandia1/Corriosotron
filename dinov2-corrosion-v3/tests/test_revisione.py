"""Test delle correzioni dopo la revisione finale (metadati, anomalie, U95, distorsione, validazioni)."""
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class MetadatiTuttiIPercorsiTest(unittest.TestCase):
    def test_classify_riceve_metadati_in_tutti_i_percorsi(self):
        src = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
        self.assertEqual(src.count('meta_bytes=img.info.get("meta_bytes")'), 3)
        self.assertEqual(src.count("analysed_size=img.size"), 3)


class AnomalieTest(unittest.TestCase):
    def test_anomalia_conta_solo_i_suoi_pixel(self):
        import anomaly_clusters
        from misura import misure
        from test_misure import proiettore
        pred = np.ones((100, 100), np.uint8)              # tutto metallo sano
        for i in range(10, 90):                           # A1: diagonale lunga e sottile
            pred[i, i] = 3
        pred[60:72, 20:32] = 2                            # A2: macchia dentro il riquadro di A1, lontana dalla diagonale
        cl = anomaly_clusters.find_clusters(pred)
        self.assertGreaterEqual(len(cl["clusters"]), 2, cl)
        out = misure.misure_automatiche(proiettore(), pred, [1950.0, 1450.0, 2050.0, 1550.0], cl["clusters"],
                                        merge_distance_px=cl["merge_distance_px"])
        for c in cl["clusters"]:
            m = next(x for x in out if x.nome == f"Anomalia {c['id']}")
            self.assertAlmostEqual(m.valore, c["rust_pixels"] * 0.005 ** 2, delta=c["rust_pixels"] * 0.005 ** 2 * 0.01)


class U95DimensioniTest(unittest.TestCase):
    def test_pdf_dimensioni_anomalia_con_u95(self):
        import report_pdf
        from test_pdf_misure import REPORT, estrai_testo
        img = Image.new("RGB", (60, 90), (120, 120, 120))
        m = {"id": "anomalia-A1", "tipo": "anomalia", "nome": "Anomalia A1", "valore": 0.0234, "u95": 0.004,
             "unita": "m²", "etichetta": "STIMA", "extra": {"lunghezza_m": 0.45, "larghezza_m": 0.12,
                                                             "u95_lunghezza_m": 0.09, "u95_larghezza_m": 0.03}}
        t = estrai_testo(report_pdf.build_pdf("f.jpg", img, img, dict(REPORT), misure=[m]))
        self.assertIn("45 \\261 9 \\327 12 \\261 3 cm", t)
        self.assertIn("234 \\261 40 cm\\262", t)   # aree piccole in cm², non «0,02 ± 0,00 m²»

    def test_valore_misura_aree_piccole(self):
        import report_pdf
        self.assertEqual(report_pdf.valore_misura(0.0234, 0.004, "m²"), "234 ± 40 cm²")
        self.assertEqual(report_pdf.valore_misura(1.5, 0.2, "m²"), "1,50 ± 0,20 m²")


class DistorsioneTest(unittest.TestCase):
    def camera(self):
        from misura.distix.calibration_manager import OpticalCalibration
        from misura.distix.camera_model import CameraModel
        cal = OpticalCalibration(4000, 3000, 2800.0, 2800.0, 2010.0, 1490.0, k1=-0.12, k2=0.05, p1=0.001,
                                 p2=-0.0008, k3=-0.01, distortion_active=True, is_complete=True)
        return CameraModel("d", np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
                           image_width=4000, image_height=3000, fx_pix=2800.0, fy_pix=2800.0, cx_pix=2010.0,
                           cy_pix=1490.0, calibration=cal)

    def test_raggi_con_distorsione_coincidono_con_distix(self):
        from misura import geometria
        cam = self.camera()
        u = np.array([0.0, 4000.0, 2000.0, 300.0, 3700.0])
        v = np.array([0.0, 3000.0, 1500.0, 2800.0, 200.0])
        dirs = geometria.raggi(cam, u, v)
        for i in range(len(u)):
            np.testing.assert_allclose(dirs[i], cam.pixel_to_ray(u[i], v[i])[1], atol=1e-9)

    def test_calibrazione_profilo_completa_o_no(self):
        from misura import drone
        m = drone.FotoMeta(width=4000, height=3000, focal35_mm=24.0)
        base = dict(id="x", nome="x", make="X", model="Y", focal_mm=4.4, sensor_w_mm=6.4, native_w=4000,
                    native_h=3000, has_rtk=True, has_lrf=False, lrf_range_m=(3, 100), sigma_gimbal_deg=1.0,
                    sigma_focal_rel=0.02)
        senza_k = drone.Profilo(**base, calibrazione={"image_width": 4000, "fx": 2800, "fy": 2800, "cx": 2000, "cy": 1500})
        con_k = drone.Profilo(**base, calibrazione={"image_width": 4000, "fx": 2800, "fy": 2800, "cx": 2000, "cy": 1500,
                                                   "k1": -0.1, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0})
        a, b = drone.intrinseci(m, senza_k), drone.intrinseci(m, con_k)
        self.assertFalse(a["calibrata"])
        self.assertTrue(b["calibrata"])
        self.assertEqual(b["dist"], (-0.1, 0.0, 0.0, 0.0, 0.0))

    def test_camera_con_distorsione_da_intrinseci(self):
        from misura import geometria
        k = {"fx": 2800.0, "fy": 2800.0, "cx": 2000.0, "cy": 1500.0, "fonte": "calibrazione", "calibrata": True,
             "dist": (-0.1, 0.0, 0.0, 0.0, 0.0)}
        cam = geometria.camera("c", np.zeros(3), -30.0, 10.0, 0.0, k, 4000, 3000)
        self.assertTrue(cam.calibration.distortion_active)
        np.testing.assert_allclose(geometria.raggi(cam, np.array([100.0]), np.array([100.0]))[0],
                                   cam.pixel_to_ray(100.0, 100.0)[1], atol=1e-9)


class ZoomTest(unittest.TestCase):
    def test_camera_zoom_drone_sconosciuto(self):
        from misura import drone
        m = drone.FotoMeta(make="Acme", model="X", image_source="ZoomCamera", width=4000, height=3000,
                           focal35_mm=24.0, lat=45.0, lon=9.0, rel_alt=30.0, gimbal_pitch=-40.0, gimbal_yaw=10.0)
        ok, msg = drone.misurabile(m, drone.scegli_profilo(m))
        self.assertFalse(ok)
        self.assertIn("zoom", msg.lower())


class ValidazioniApiTest(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from misura import api
        self.tmp = tempfile.TemporaryDirectory()
        app = FastAPI()
        app.include_router(api.crea_router(OrderedDict(), cartella=Path(self.tmp.name)))
        self.c = TestClient(app)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sha_con_percorso_rifiutato(self):
        r = self.c.post("/api/misura/sessione/punto", json={"sha_principale": "../../x", "pid": "T1", "sha_foto": "a",
                                                            "u": 1, "v": 1})
        self.assertEqual(r.status_code, 422)

    def test_fattore_non_valido_rifiutato(self):
        r = self.c.post("/api/misura/sessione/calcola", json={"sha_principale": "a", "result_id": "r", "fattore": 0})
        self.assertEqual(r.status_code, 422)

    def test_stima_da_una_foto_rimossa(self):
        r = self.c.post("/api/misura/stima", json={"result_id": "r"})
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
