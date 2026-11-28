import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def proiettore(n_campioni=200):
    """Camera frontale fx=2000 a 10 m da un piano verticale: 5 mm/px ovunque."""
    from misura import geometria
    from misura.distix.camera_model import CameraModel
    cam = CameraModel("t", np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
                      image_width=4000, image_height=3000, fx_pix=2000.0, fy_pix=2000.0, cx_pix=2000.0, cy_pix=1500.0)
    piano = geometria.Piano(np.array([0.0, 10.0, 0.0]), np.array([0.0, -1.0, 0.0]))
    return geometria.Proiettore(cam=cam, piano=piano, etichetta="STIMA",
                                campioni=[(cam, piano)] * n_campioni, info={})


ROI = [1950.0, 1450.0, 2050.0, 1550.0]  # 100×100 px → maschera 100×100: 1 cella = 1 px


def per_nome(misure, nome):
    return next(m for m in misure if m.nome == nome)


class MisureTest(unittest.TestCase):
    def test_area_classe(self):
        from misura import misure
        mask = np.zeros((100, 100), np.uint8)
        mask[10:30, 40:60] = 3  # 400 px
        out = misure.misure_automatiche(proiettore(), mask, ROI, None)
        m = per_nome(out, "Ruggine intensa")
        self.assertEqual(m.tipo, "classe")
        self.assertEqual(m.unita, "m²")
        self.assertAlmostEqual(m.valore, 400 * 0.005 ** 2, delta=400 * 0.005 ** 2 * 0.01)
        self.assertAlmostEqual(m.extra["reale_m2"], 3 * m.valore, places=9)
        self.assertEqual(m.extra["fattore"], 3.0)
        self.assertEqual(m.etichetta, "STIMA")
        self.assertGreater(m.extra["u95_reale_m2"], 0)

    def test_nessun_pixel(self):
        from misura import misure
        out = misure.misure_automatiche(proiettore(), np.zeros((100, 100), np.uint8), ROI, None)
        m = per_nome(out, "Ruggine media")
        self.assertEqual(m.valore, 0.0)
        self.assertEqual(m.u95, 0.0)
        self.assertEqual(m.extra["u95_reale_m2"], 0.0)

    def test_maschera_grande_scalata(self):
        from misura import misure
        # 1000×800 celle che coprono un ROI di 2000×1600 px: ogni cella = 2×2 px = 1 cm²
        mask = np.zeros((800, 1000), np.uint8)
        mask[:400, :] = 2
        out = misure.misure_automatiche(proiettore(), mask, [1000, 700, 3000, 2300], None)
        self.assertAlmostEqual(per_nome(out, "Ruggine media").valore, 400000 * 1e-4, delta=40 * 0.01)

    def test_anomalia(self):
        from misura import misure
        mask = np.zeros((100, 100), np.uint8)
        mask[20:60, 10:30] = 3
        clusters = [{"id": "A1", "bbox_px": [10, 20, 30, 60]}]
        out = misure.misure_automatiche(proiettore(), mask, ROI, clusters)
        a = per_nome(out, "Anomalia A1")
        self.assertEqual(a.tipo, "anomalia")
        self.assertAlmostEqual(a.extra["lunghezza_m"], 0.20, delta=0.002)
        self.assertAlmostEqual(a.extra["larghezza_m"], 0.10, delta=0.001)
        self.assertAlmostEqual(a.valore, 800 * 0.005 ** 2, delta=800 * 0.005 ** 2 * 0.01)

    def test_linea(self):
        from misura import misure
        m = misure.misura_linea(proiettore(), [(1900, 1500), (2100, 1500)], "L1")
        self.assertEqual((m.tipo, m.unita, m.etichetta), ("linea", "m", "STIMA"))
        self.assertAlmostEqual(m.valore, 1.0, delta=0.01)
        self.assertGreater(m.u95, 0)

    def test_poligono(self):
        from misura import misure
        m = misure.misura_area(proiettore(), [(1900, 1400), (2100, 1400), (2100, 1600), (1900, 1600)], "S1")
        self.assertEqual((m.tipo, m.unita), ("area", "m²"))
        self.assertAlmostEqual(m.valore, 1.0, delta=0.01)
        self.assertGreater(m.u95, 0)

    def test_to_dict(self):
        from misura import misure
        d = misure.misura_linea(proiettore(), [(1900, 1500), (2100, 1500)], "L1").to_dict()
        for k in ("id", "tipo", "nome", "valore", "u95", "unita", "etichetta", "extra"):
            self.assertIn(k, d)


if __name__ == "__main__":
    unittest.main()
