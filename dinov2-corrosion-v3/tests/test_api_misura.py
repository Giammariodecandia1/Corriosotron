import base64
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FOTO = Path(r"C:\Users\gdecandia\Desktop\Oggi2\tralicci corrosione")
CAM4 = FOTO / "Camera_4_2026-07-14_11.01.03.000.jpg"
CAM3 = FOTO / "Camera_3_2026-04-08_13.49.43.000.jpg"
HAS_PHOTOS = CAM4.exists() and CAM3.exists()
CHIAVI_V2 = {"report", "report_pdf_base64", "overlay_base64", "ambiguity_mask_base64", "class_mask_base64",
             "class_id_labels", "report_csv"}
BOX = [3000, 300, 5000, 5700]


def voce(meta_bytes=b""):
    pred = np.zeros((100, 100), np.uint8)
    pred[10:50, 20:60] = 3
    img = Image.new("RGB", (100, 100), (120, 120, 120))
    return dict(crop=img, blend=img, pred=pred, report={"operator_notes": "", "profile_name": "t",
                                                       "shot_captured_at": "", "shot_coordinates": ""},
                clusters=None, map_info=None, photo_name="Camera_4.jpg", meta_bytes=meta_bytes,
                analysed_size=(8000, 6000), sha="sha4", roi_box=BOX)


class CacheTest(unittest.TestCase):
    def test_cache_max_5(self):
        import main
        main.ANALISI.clear()
        for i in range(6):
            main.ricorda_analisi(f"r{i}", **voce())
        self.assertEqual(list(main.ANALISI), [f"r{i}" for i in range(1, 6)])

    def test_classify_ricorda_e_chiavi_invariate(self):
        import main
        img = Image.new("RGB", (392, 392), (70, 110, 50))
        a = np.asarray(img).copy()
        a[:, 180:200] = (176, 120, 70)
        a[:, 60:80] = (230, 230, 230)
        img = Image.fromarray(a)
        metal = np.zeros((392, 392), bool)
        metal[:, 60:80] = metal[:, 180:200] = True
        geo = {"box": (0, 0, 392, 392), "left": 0, "top": 0, "rw": 392, "rh": 392, "side": 392,
               "source_width": 392, "source_height": 392}
        vec = np.zeros((28, 28, 384), np.float32)
        vec[:, :, 0] = 1
        p = main.train(vec, img, geo, [{"x": 70 / 392, "y": .3, "label": "healthy"},
                                       {"x": 190 / 392, "y": .6, "label": "medium"}],
                       color_source=(np.asarray(img), metal))
        out = main.classify(vec, img, geo, p, 0.1, metal=metal, meta_bytes=b"head", analysed_size=(392, 392))
        self.assertEqual(set(out), CHIAVI_V2)
        e = main.ANALISI[out["report"]["result_id"]]
        self.assertEqual(e["meta_bytes"], b"head")
        self.assertEqual(e["analysed_size"], (392, 392))
        self.assertEqual(list(e["roi_box"]), [0, 0, 392, 392])
        self.assertEqual(e["pred"].shape[::-1], tuple(out["report"]["mask_dimensions_px"]))


@unittest.skipUnless(HAS_PHOTOS, "foto reali assenti")
class ApiMisuraTest(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from misura import api
        self.tmp = tempfile.TemporaryDirectory()
        self.analisi = OrderedDict()
        self.analisi["r1"] = voce(CAM4.read_bytes()[:2 * 1024 * 1024])
        app = FastAPI()
        app.include_router(api.crea_router(self.analisi, cartella=Path(self.tmp.name)))
        self.c = TestClient(app)

    def tearDown(self):
        self.tmp.cleanup()

    def test_analisi_scaduta(self):
        r = self.c.post("/api/misura/sessione/apri", json={"result_id": "nope"})
        self.assertEqual(r.status_code, 404)
        self.assertIn("ripeti l'analisi", r.json()["detail"])

    def test_pdf_con_misure(self):
        m = {"id": "linea-L1", "tipo": "linea", "nome": "L1", "valore": 1.62, "u95": 0.26, "unita": "m",
             "etichetta": "NON VALIDATA", "extra": {}}
        r = self.c.post("/api/misura/pdf", json={"result_id": "r1", "misure": [m]})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(base64.b64decode(r.json()["report_pdf_base64"]).startswith(b"%PDF"))

    def test_sessione_flusso(self):
        s = self.c.post("/api/misura/sessione/apri", json={"result_id": "r1"}).json()["sessione"]
        self.assertEqual(s["sha_principale"], "sha4")
        self.assertEqual(len(s["foto"]), 1)
        lontana = self.c.post("/api/misura/sessione/foto", data={"sha_principale": "sha4"},
                              files={"file": ("Camera_3.jpg", CAM3.read_bytes(), "image/jpeg")}).json()
        self.assertFalse(lontana["misurabile"], lontana)  # scattata a km di distanza: altro sostegno
        b4 = CAM4.read_bytes()
        self.assertEqual(b4.count(b"+781.875"), 1)
        info = self.c.post("/api/misura/sessione/foto", data={"sha_principale": "sha4"},
                           files={"file": ("Camera_4_piu_10m.jpg", b4.replace(b"+781.875", b"+791.875"),
                                           "image/jpeg")}).json()
        self.assertTrue(info["misurabile"], info)
        self.assertEqual(self.c.get(f"/api/misura/sessione/foto/{info['sha']}").status_code, 200)
        r = self.c.post("/api/misura/sessione/punto", json={"sha_principale": "sha4", "pid": "T1", "sha_foto": "sha4",
                                                            "u": 4000, "v": 5000})
        self.assertEqual(r.status_code, 200, r.text)
        curva = self.c.post("/api/misura/sessione/suggerisci",
                            json={"sha_principale": "sha4", "pid": "T1", "sha_foto": info["sha"]}).json()["curva"]
        self.assertIsInstance(curva, list)
        e = self.c.post("/api/misura/sessione/calcola", json={"sha_principale": "sha4", "result_id": "r1"}).json()
        self.assertEqual(e["etichetta"], "INCOMPLETA")
        self.assertTrue(any("T1" in m for m in e["messaggi"]))
        riaperta = self.c.post("/api/misura/sessione/apri", json={"result_id": "r1"}).json()["sessione"]
        self.assertIn("T1", riaperta["tie"])


if __name__ == "__main__":
    unittest.main()
