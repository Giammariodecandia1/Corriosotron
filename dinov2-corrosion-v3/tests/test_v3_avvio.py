import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class AvvioV3Test(unittest.TestCase):
    def test_root_e_porta(self):
        import main
        self.assertEqual(main.ROOT.name, "dinov2-corrosion-v3")
        bat = (main.ROOT / "AVVIA_V3.bat").read_text(encoding="utf-8", errors="ignore")
        self.assertIn("8767", bat)
        self.assertNotIn("8766", bat)

    def test_status(self):
        from fastapi.testclient import TestClient
        import main
        self.assertEqual(TestClient(main.app).get("/api/status").status_code, 200)


if __name__ == "__main__":
    unittest.main()
