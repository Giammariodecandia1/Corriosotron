import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import object_detection as od


def instance_map():
    m = np.zeros((100, 200), np.uint8)
    m[10:30, 20:60] = 1    # object 1: 800 px
    m[50:90, 100:180] = 2  # object 2: 3200 px
    m[60:70, 120:130] = 3  # object 3 inside object 2: 100 px
    return m


class RulesTest(unittest.TestCase):
    def test_parse_keeps_items_by_id(self):
        rules = od.parse_rules(json.dumps([{"n": 1, "mode": "exclude", "class_name": "Isolatori"},
                                           {"n": 2, "mode": "ignore"}]), 3)
        self.assertEqual(sorted(rules), [1, 2])
        self.assertEqual(rules[1]["class_name"], "Isolatori")

    def test_rejects_bad_mode_or_id(self):
        for bad in ([{"n": 1, "mode": "delete"}], [{"n": 4, "mode": "exclude"}], [{"mode": "exclude"}], {"n": 1}):
            with self.assertRaises(ValueError):
                od.parse_rules(json.dumps(bad), 3)

    def test_empty_rules(self):
        self.assertEqual(od.parse_rules("[]", 0), {})


class MasksTest(unittest.TestCase):
    def test_exclude_only(self):
        rules = {1: {"mode": "exclude"}, 2: {"mode": "ignore"}, 3: {"mode": "ignore"}}
        exclude, focus, px = od.object_masks(instance_map(), rules, (0, 0, 200, 100), (200, 100))
        self.assertIsNone(focus)
        self.assertEqual(int(exclude.sum()), 800)
        self.assertEqual(px, {1: 800, 2: 3100, 3: 100})

    def test_focus_and_exclude_inside(self):
        rules = {2: {"mode": "focus"}, 3: {"mode": "exclude"}}
        exclude, focus, _ = od.object_masks(instance_map(), rules, (0, 0, 200, 100), (200, 100))
        self.assertEqual(int(focus.sum()), 3100)
        self.assertEqual(int(exclude.sum()), 100)
        self.assertFalse((focus & exclude).any())

    def test_crop_and_resize_to_analysis_grid(self):
        rules = {2: {"mode": "exclude"}}
        exclude, _, px = od.object_masks(instance_map(), rules, (100, 50, 200, 100), (50, 25))
        self.assertEqual(exclude.shape, (25, 50))
        # object 2 covers x 100-180, y 50-90 of the 100x50 crop -> 40x20 of the 50x25 grid,
        # minus object 3 (5x5 after halving)
        self.assertEqual(int(exclude.sum()), 40 * 20 - 25)
        self.assertEqual(px[2], 40 * 20 - 25)


class FoldersTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.saved = od.MODELS_DIR, od.FOLDERS_FILE
        od.MODELS_DIR = base / "interni"
        od.MODELS_DIR.mkdir()
        od.FOLDERS_FILE = od.MODELS_DIR / "cartelle.json"
        self.extra = base / "esterni"
        self.extra.mkdir()
        (od.MODELS_DIR / "a.pt").write_bytes(b"x")
        (self.extra / "b.pt").write_bytes(b"x")
        (self.extra / "note.txt").write_text("x")

    def tearDown(self):
        od.MODELS_DIR, od.FOLDERS_FILE = self.saved
        self.tmp.cleanup()

    def test_add_list_remove_folder(self):
        self.assertEqual([f["folder"] for f in od.list_files()], [str(od.MODELS_DIR.resolve())])
        od.add_folder(str(self.extra))
        od.add_folder(str(self.extra))  # no duplicates
        files = od.list_files()
        self.assertEqual(len(files), 2)
        self.assertEqual([m["name"] for m in files[1]["models"]], ["b"])
        self.assertTrue(files[0]["internal"])
        od.remove_folder(str(self.extra))
        self.assertEqual(len(od.list_files()), 1)

    def test_missing_folder_and_internal_folder(self):
        with self.assertRaises(ValueError):
            od.add_folder(str(self.extra / "manca"))
        with self.assertRaises(ValueError):
            od.remove_folder(str(od.MODELS_DIR))

    def test_only_pt_inside_configured_folders(self):
        od.resolve_model(str(od.MODELS_DIR / "a.pt"))
        for bad in (self.extra / "b.pt", self.extra / "note.txt", od.MODELS_DIR / "manca.pt"):
            with self.assertRaises(KeyError):
                od.resolve_model(str(bad))  # b.pt: its folder was never added
        od.add_folder(str(self.extra))
        od.resolve_model(str(self.extra / "b.pt"))


class ModelsTest(unittest.TestCase):
    def test_insulator_model_is_listed_with_its_class(self):
        path = od.MODELS_DIR / "Isolatori V1s.pt"
        if not path.is_file():
            self.skipTest("modello isolatori non presente")
        m = od.model_info(str(path))
        self.assertIsNone(m["error"])
        self.assertEqual(m["task"], "detect")
        self.assertEqual(list(m["classes"].values()), ["Isolatori"])
        self.assertEqual(m["default_modes"], {"Isolatori": "exclude"})
        self.assertTrue(m["enabled_by_default"])
        self.assertEqual(m["imgsz"], 960)


if __name__ == "__main__":
    unittest.main()
