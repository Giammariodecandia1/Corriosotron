import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CENTRO = np.array([0.0, 15.0, 5.0])
PUNTI = {f"T{i + 1}": np.array([x, 15.0, z]) for i, (x, z) in
         enumerate([(-3, 2), (3, 2), (-3, 8), (3, 8), (0, 5.5), (1.5, 3.5)])}


def scena(yaws=(0.0, -20.0, 25.0), calibrata=True, rtk_float=None, noise_px=0.5, seed=3, togli=()):
    """Camere a 15 m dal piano y=15 (faccia del sostegno), che guardano il centro."""
    from misura.distix.calibration_manager import OpticalCalibration
    from misura.distix.camera_model import CameraModel
    rng = np.random.default_rng(seed)
    cams = {}
    for i, yaw in enumerate(yaws):
        y = np.deg2rad(yaw)
        front = np.array([np.sin(y), np.cos(y), 0.0])
        pos = CENTRO - 15.0 * front
        name = "ABCDEF"[i]
        cams[name] = CameraModel(name, pos, front, np.array([0.0, 0.0, 1.0]), image_width=4000, image_height=3000,
                                 fx_pix=2800.0, fy_pix=2800.0, cx_pix=2000.0, cy_pix=1500.0,
                                 rtk_fix=name != rtk_float, rtk_flag=16 if name == rtk_float else 50, rtk_std=0.01,
                                 rtk_std_hgt=0.02,
                                 calibration=OpticalCalibration(4000, 3000, 2800.0, 2800.0, 2000.0, 1500.0, source="test",
                                                                         is_complete=True, is_validated=True) if calibrata else None)
    tie = {}
    for pid, X in PUNTI.items():
        for name, cam in cams.items():
            if (pid, name) in togli:
                continue
            u, v, _ = cam.project_point(X)
            tie.setdefault(pid, {})[name] = (u + rng.normal(0, noise_px), v + rng.normal(0, noise_px))
    return cams, tie


def sessione_da(cams, tie, cartella):
    from misura import sessione
    s = sessione.Sessione(sha_principale="A", foto=[{"sha": n, "nome": n} for n in cams], tie=tie,
                          cartella=Path(cartella))
    return s, (lambda _s: {n: c.clone() if hasattr(c, "clone") else c for n, c in cams.items()})


class SessioneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def calcola(self, **kw):
        from misura import sessione
        cams, tie = scena(**kw)
        s, fn = sessione_da(cams, tie, self.tmp.name)
        return sessione.calcola(s, camere_fn=fn), cams

    def controllo(self, esito, inizio):
        return next(c for c in esito.controlli if c["testo"].startswith(inizio))

    def test_validata(self):
        from misura import misure
        esito, cams = self.calcola()
        self.assertEqual(esito.etichetta, "VALIDATA", esito.messaggi)
        self.assertTrue(esito.completa)
        n = esito.proiettore.piano.n
        self.assertLess(np.degrees(np.arccos(abs(n @ np.array([0.0, -1.0, 0.0])))), 2.0)
        quadrato = [cams["A"].project_point(np.array(p))[:2] for p in
                    ([-0.5, 15, 4.5], [0.5, 15, 4.5], [0.5, 15, 5.5], [-0.5, 15, 5.5])]
        m = misure.misura_area(esito.proiettore, quadrato, "S1")
        self.assertAlmostEqual(m.valore, 1.0, delta=0.05)
        self.assertEqual(m.etichetta, "VALIDATA")
        self.assertEqual(len(esito.proiettore.campioni), 200)
        self.assertEqual(esito.tie_info["T1"]["n_foto"], 3)

    def test_senza_calibrazione(self):
        esito, _ = self.calcola(calibrata=False)
        self.assertEqual(esito.etichetta, "NON VALIDATA")
        self.assertFalse(self.controllo(esito, "Camera calibrata")["ok"])
        self.assertIsNotNone(esito.proiettore)

    def test_rtk_float(self):
        esito, _ = self.calcola(rtk_float="B")
        self.assertEqual(esito.etichetta, "NON VALIDATA")
        self.assertFalse(self.controllo(esito, "RTK FIX")["ok"])

    def test_pochi_punti(self):
        togli = {("T4", "B"), ("T4", "C"), ("T5", "A"), ("T5", "B"), ("T5", "C"), ("T6", "A"), ("T6", "B"), ("T6", "C")}
        esito, _ = self.calcola(togli=togli)
        self.assertEqual(esito.etichetta, "INCOMPLETA")
        self.assertFalse(esito.completa)
        self.assertTrue(any("T4" in m for m in esito.messaggi), esito.messaggi)
        self.assertIsNone(esito.proiettore)

    def test_foto_allineate(self):
        esito, _ = self.calcola(yaws=(0.0, 3.0, 6.0))
        self.assertNotEqual(esito.etichetta, "VALIDATA")
        self.assertFalse(self.controllo(esito, "Convergenza")["ok"])
        self.assertTrue(any("più di lato" in m for m in esito.messaggi), esito.messaggi)
        # i clic sono giusti: il problema è solo la geometria, non va segnalato come errore di riproiezione
        self.assertTrue(self.controllo(esito, "Riproiezione")["ok"])
        self.assertFalse(any("non tornano" in m for m in esito.messaggi), esito.messaggi)

    def test_salva_carica(self):
        from misura import sessione
        cams, tie = scena()
        s, _ = sessione_da(cams, tie, self.tmp.name)
        s.salva()
        t = sessione.Sessione.carica("A", cartella=Path(self.tmp.name))
        self.assertEqual(t.foto, s.foto)
        self.assertEqual(set(t.tie), set(s.tie))
        self.assertAlmostEqual(t.tie["T1"]["B"][0], s.tie["T1"]["B"][0])
        self.assertIsNone(sessione.Sessione.carica("ZZ", cartella=Path(self.tmp.name)))

    def test_foto_duplicata(self):
        from misura import sessione
        s = sessione.Sessione(sha_principale="P", foto=[{"sha": "P", "nome": "principale"}], tie={},
                              cartella=Path(self.tmp.name))
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (100, 100, 100)).save(buf, format="JPEG")
        prima = s.aggiungi_foto(buf.getvalue(), "extra.jpg")
        seconda = s.aggiungi_foto(buf.getvalue(), "extra_copia.jpg")
        self.assertEqual(len(s.foto), 2)
        self.assertFalse(prima["misurabile"])  # JPEG senza metadati: non misurabile, ma aggiunta con il motivo
        self.assertIn("già presente", seconda["motivo"])
        self.assertTrue((Path(self.tmp.name) / "foto" / f"{prima['sha']}.jpg").exists())

    @unittest.skipUnless(Path(r"C:\Users\gdecandia\Desktop\Oggi2\tralicci corrosione").exists(), "foto reali assenti")
    def test_camere_da_foto_reali(self):
        from misura import sessione
        F = Path(r"C:\Users\gdecandia\Desktop\Oggi2\tralicci corrosione")
        b4 = (F / "Camera_4_2026-07-14_11.01.03.000.jpg").read_bytes()
        s = sessione.Sessione.apri(b4, (4000, 3000), "Camera_4", "princ", cartella=Path(self.tmp.name))
        info = s.aggiungi_foto((F / "Camera_3_2026-04-08_13.49.43.000.jpg").read_bytes(), "Camera_3")
        self.assertTrue(info["misurabile"], info)
        cams = sessione.camere(s)
        self.assertEqual(set(cams), {"princ", info["sha"]})
        np.testing.assert_allclose(cams["princ"].pos, 0, atol=1e-6)
        self.assertEqual(cams["princ"].image_width, 4000)   # principale analizzata ridotta: intrinseci scalati
        self.assertTrue(all(c.has_intrinsics and c.rtk_fix for c in cams.values()))
        self.assertIsNone(cams["princ"].calibration)        # M3T senza profilo di calibrazione
        riaperta = sessione.Sessione.apri(b4, (4000, 3000), "Camera_4", "princ", cartella=Path(self.tmp.name))
        self.assertEqual(len(riaperta.foto), 2)

    def test_suggerisci(self):
        from misura import sessione
        cams, tie = scena()
        s, fn = sessione_da(cams, tie, self.tmp.name)
        curva = sessione.suggerisci(s, "T1", "B", camere_fn=fn)
        self.assertGreater(len(curva), 2)
        u, v, _ = cams["B"].project_point(PUNTI["T1"])
        dist = min(np.hypot(a - u, b - v) for a, b in curva)
        self.assertLess(dist, 200)


if __name__ == "__main__":
    unittest.main()
