"""Software tests of the per-pixel colour decision with synthetic pixels: NOT accuracy tests."""
import base64, io, sys, unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
import color_evidence as ce
from feature_sampling import merge_samples

WHITE, SHADE, TAN, BROWN, TEAL = (225, 228, 230), (95, 97, 99), (205, 175, 130), (120, 85, 55), (60, 190, 170)


def lab(rgb):
    return ce.rgb_to_lab(np.array([[rgb]], np.uint8))[0, 0]


def bank():
    return {"healthy": np.stack([lab(WHITE)] * 3), "medium": np.stack([lab(TAN)] * 3), "intense": np.stack([lab(BROWN)] * 3)}


class ColourTests(unittest.TestCase):
    def test_lab_reference_values(self):
        np.testing.assert_allclose(lab((255, 255, 255)), [100, 0, 0], atol=0.6)
        np.testing.assert_allclose(lab((255, 0, 0)), [53.2, 80.1, 67.2], atol=0.6)

    def test_shadow_on_sound_steel_is_not_rust(self):
        # Regression: with lightness in the distance, shaded white faces matched dark rust.
        ids, raw, _ = ce.classify_pixels(np.stack([lab(WHITE), lab(SHADE), lab(TAN), lab(BROWN), lab(TEAL)]), bank())
        self.assertEqual(ids.tolist(), [ce.ID_HEALTHY, ce.ID_HEALTHY, ce.ID_MEDIUM, ce.ID_INTENSE, ce.ID_UNCERTAIN])
        self.assertEqual(raw[:4].tolist(), [False, False, True, True])

    def test_single_rust_level_and_black_pixels(self):
        only = {"healthy": bank()["healthy"], "intense": bank()["intense"]}
        ids, _, _ = ce.classify_pixels(np.stack([lab(TAN), lab((8, 6, 5))]), only)
        self.assertEqual(ids.tolist(), [ce.ID_INTENSE, ce.ID_UNCERTAIN])

    def test_sample_color_needs_metal_under_the_click(self):
        rgb = np.zeros((40, 40, 3), np.uint8); rgb[:, 18:24] = TAN
        metal = np.zeros((40, 40), bool); metal[:, 18:24] = True
        np.testing.assert_allclose(ce.sample_color(rgb, metal, 20, 20), lab(TAN), atol=0.01)
        self.assertIsNone(ce.sample_color(rgb, metal, 5, 5))

    def test_merge_fills_missing_colour_once(self):
        profile = {"samples": [{"sample_id": "a", "label": "healthy", "color_lab": None}]}
        self.assertEqual(merge_samples(profile, [{"sample_id": "a", "label": "healthy", "color_lab": [80, 0, 0]}]), 1)
        self.assertEqual(profile["samples"][0]["color_lab"], [80, 0, 0])
        self.assertEqual(merge_samples(profile, [{"sample_id": "a", "label": "healthy", "color_lab": [1, 1, 1]}]), 0)

    def _scene(self):
        """Three vertical 'members' on green ground: white, tan (rust), white in shade."""
        img = Image.new("RGB", (392, 392), (70, 110, 50)); a = np.asarray(img).copy()
        metal = np.zeros((392, 392), bool)
        for x, colour in ((60, WHITE), (180, TAN), (300, SHADE)):
            a[:, x:x + 20] = colour; metal[:, x:x + 20] = True
        img = Image.fromarray(a)
        geo = {"box": (0, 0, 392, 392), "left": 0, "top": 0, "rw": 392, "rh": 392, "side": 392, "source_width": 392, "source_height": 392}
        vec = np.zeros((28, 28, 384), np.float32); vec[:, :, 0] = 1   # DINO sees no difference at all
        return img, metal, geo, vec

    def test_train_and_classify_decide_by_colour_inside_metal(self):
        img, metal, geo, vec = self._scene()
        points = [{"x": 70 / 392, "y": .3, "label": "healthy"}, {"x": 190 / 392, "y": .6, "label": "medium"},
                  {"x": 10 / 392, "y": .9, "label": "healthy"}]            # last marker is on the ground
        p = main.train(vec, img, geo, points, color_source=(np.asarray(img), metal))
        self.assertEqual(p.pop("markers_off_metal"), 1)
        self.assertEqual(sum(s["color_lab"] is not None for s in p["samples"]), 2)
        out = main.classify(vec, img, geo, p, 0.1, metal=metal, markers_off_metal=1)
        r = out["report"]; cls = np.asarray(Image.open(io.BytesIO(base64.b64decode(out["class_mask_base64"]))))
        self.assertEqual(r["decision_basis"], "pixel_color")
        self.assertTrue(np.all(cls[:, :55] == 0) and np.all(cls[:, 85:175] == 0))         # ground never counted
        self.assertTrue(np.all(cls[5:-5, 62:78] == 1) and np.all(cls[5:-5, 302:318] == 1))  # sunlit and shaded steel
        self.assertTrue(np.all(cls[5:-5, 182:198] == 2))
        counts = r["class_pixel_counts"]; metal_px = sum(counts[k] for k in ("healthy", "medium", "intense", "uncertain"))
        self.assertEqual(r["corrosion_pct_of_visible_metal"], round(100 * (counts["medium"] + counts["intense"]) / metal_px, 2))
        self.assertAlmostEqual(r["corrosion_pct_of_visible_metal"], 100 / 3, delta=1.0)
        self.assertEqual(sum(counts.values()), cls.size)
        self.assertTrue(any("1 marker" in w for w in r["quality_warnings"]))

    def test_profile_without_colours_keeps_dino_path(self):
        img, metal, geo, vec = self._scene()
        vec[:, 12:15] = 0; vec[:, 12:15, 1] = 1
        p = main.train(vec, img, geo, [{"x": 70 / 392, "y": .3, "label": "healthy"}, {"x": 190 / 392, "y": .6, "label": "medium"}])
        self.assertTrue(all(s["color_lab"] is None for s in p["samples"]))
        r = main.classify(vec, img, geo, p, 0.1, metal=metal)["report"]
        self.assertEqual(r["decision_basis"], "dino_patches")
        self.assertTrue(any("senza colori" in w for w in r["quality_warnings"]))


if __name__ == "__main__":
    unittest.main()
