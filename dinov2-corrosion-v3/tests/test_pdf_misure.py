import base64
import re
import sys
import unittest
import zlib
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPORT = {"profile_name": "Profilo test", "shot_captured_at": "14/07/2026 11:01:03",
          "shot_coordinates": "45.1 N, 9.2 E", "corrosion_pct_of_visible_metal": 30.0,
          "class_pct_of_visible_metal": {"healthy": 70.0, "medium": 20.0, "intense": 10.0, "uncertain": 0.0},
          "calibrated_classes": ["healthy", "medium", "intense"]}
MISURE = [
    {"id": "classe-3", "tipo": "classe", "nome": "Ruggine intensa", "valore": 0.82, "u95": 0.13, "unita": "m²",
     "etichetta": "STIMA", "extra": {"reale_m2": 2.46, "u95_reale_m2": 0.9, "fattore": 3.0}},
    {"id": "linea-L1", "tipo": "linea", "nome": "L1 montante", "valore": 2.31, "u95": 0.19, "unita": "m",
     "etichetta": "NON VALIDATA", "extra": {}},
    {"id": "linea-L2", "tipo": "linea", "nome": "L2 bullone", "valore": 0.45, "u95": 0.02, "unita": "m",
     "etichetta": "STIMA", "extra": {}},
]
INFO = {"drone": "DJI Mavic 3T (M3T)", "fonte_distanza": "geometria", "distanza_m": 11.4}


def pagine(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf))


def estrai_testo(pdf: bytes) -> str:
    out = []
    for raw in re.findall(rb"stream\r?\n(.*?)endstream", pdf, re.S):
        data = raw.strip()
        if data.endswith(b"~>"):  # reportlab: ASCII85 sopra Flate
            data = base64.a85decode(b"".join(data[:-2].split()))
        try:
            out.append(zlib.decompress(data).decode("latin-1"))
        except zlib.error:
            out.append(data.decode("latin-1"))
    return "\n".join(out)


def build(**kw):
    import report_pdf
    img = Image.new("RGB", (200, 300), (120, 120, 120))
    return report_pdf.build_pdf("foto.jpg", img, img, dict(REPORT), **kw)


class PdfMisureTest(unittest.TestCase):
    def test_senza_misure_identico(self):
        a, b = build(), build(misure=None)
        self.assertEqual(pagine(a), pagine(b))
        self.assertEqual(pagine(a), 1)
        self.assertNotIn("Misure", estrai_testo(b))

    def test_con_misure(self):
        pdf = build(misure=MISURE, misure_info=INFO)
        self.assertEqual(pagine(pdf), 2)
        t = estrai_testo(pdf)
        for atteso in ("Misure", "STIMA", "NON VALIDATA", "Superficie reale stimata", "Area vista",
                       "0,82", "2,31", "0,19 m", "45", "cm", "M3T"):
            self.assertIn(atteso, t)
        self.assertIn("\\261", t.replace("±", "\\261"))  # ± presente (codifica WinAnsi)

    def test_misure_non_misurabili_e_parziali(self):
        speciali = [{"id": "linea-L9", "tipo": "linea", "nome": "L9", "valore": 0.0, "u95": 0.0, "unita": "m",
                     "etichetta": "STIMA", "extra": {"errore": "un punto non cade sulla faccia"}},
                    {**MISURE[0], "extra": {**MISURE[0]["extra"], "non_misurabile_pct": 12.0}}]
        t = estrai_testo(build(misure=speciali, misure_info=INFO))
        self.assertIn("non misurabile", t)
        self.assertIn("parz. 12%", t)  # nel PDF le parentesi sono scritte come \( \)
        self.assertIn("non misurata", t)

    def test_molte_misure_tutte_stampate(self):
        tante = [{"id": f"linea-M{i}", "tipo": "linea", "nome": f"Misura numero {i}", "valore": 1.5, "u95": 0.1,
                  "unita": "m", "etichetta": "STIMA", "extra": {}} for i in range(1, 61)]
        pdf = build(misure=tante, misure_info=INFO)
        t = estrai_testo(pdf)
        for i in (1, 27, 28, 59, 60):
            self.assertIn(f"Misura numero {i})", t)  # ")" chiude la stringa PDF: niente falsi positivi tipo "6" in "60"
        self.assertGreater(pagine(pdf), 2)
        self.assertEqual(t.count("Superficie reale stimata"), 1)  # spiegazione solo in fondo
        self.assertIn(f"Pag. {pagine(pdf)}/{pagine(pdf)}", t)


if __name__ == "__main__":
    unittest.main()
