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


def proiettore_suolo():
    """Stessa camera, orizzontale, sopra un piano a terra 10 m più in basso: sopra la riga 1500 nessun punto."""
    from misura import geometria
    pr = proiettore(n_campioni=20)
    piano = geometria.Piano(np.array([0.0, 0.0, -10.0]), np.array([0.0, 0.0, 1.0]))
    return geometria.Proiettore(cam=pr.cam, piano=piano, etichetta="STIMA", campioni=[(pr.cam, piano)] * 20, info={})


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

    def test_u95_robusto_ai_casi_estremi(self):
        from misura import misure
        rng = np.random.default_rng(0)
        valori = list(rng.normal(10.0, 0.5, 197)) + [1e3, 1e3, -1e3]  # 3 campioni assurdi su 200
        self.assertAlmostEqual(misure._u95(valori), 1.0, delta=0.25)

    def test_zona_non_proiettabile_segnalata(self):
        from misura import misure
        mask = np.full((100, 100), 3, np.uint8)
        # piano orizzontale 10 m sotto la camera: la metà alta della ROI guarda sopra l'orizzonte
        out = misure.misure_automatiche(proiettore_suolo(), mask, ROI, None)
        m = per_nome(out, "Ruggine intensa")
        self.assertTrue(np.isfinite(m.valore) and m.valore > 0)
        self.assertAlmostEqual(m.extra["non_misurabile_pct"], 50.0, delta=3.0)
        self.assertNotIn("non_misurabile_pct", per_nome(misure.misure_automatiche(proiettore(), mask, ROI, None),
                                                        "Ruggine intensa").extra)

    def test_linea_impossibile_non_rompe(self):
        import json
        from misura import misure
        m = misure.misura_linea(proiettore_suolo(), [(2000, 1600), (2000, 1000)], "L9")  # 2° punto sopra l'orizzonte
        self.assertIn("errore", m.extra)
        self.assertIn("sopra l'orizzonte", m.extra["errore"])
        json.dumps(m.to_dict(), allow_nan=False)  # niente NaN: il server risponde invece di dare errore 500

    def test_to_dict(self):
        from misura import misure
        d = misure.misura_linea(proiettore(), [(1900, 1500), (2100, 1500)], "L1").to_dict()
        for k in ("id", "tipo", "nome", "valore", "u95", "unita", "etichetta", "extra"):
            self.assertIn(k, d)


if __name__ == "__main__":
    unittest.main()
