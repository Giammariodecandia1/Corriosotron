"""Software tests of anomaly clustering on synthetic class maps: NOT accuracy tests."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anomaly_clusters as ac

HEALTHY, MEDIUM, INTENSE = 1, 2, 3


def tower():
    """400x200 frame, all visible metal (healthy): 80 000 metal pixels, merge radius 4 px."""
    pred = np.full((400, 200), HEALTHY, np.uint8)
    pred[20:80, 20:60] = MEDIUM            # big patch: 2400 px = 3% -> rilevante
    pred[20:35, 20:60] = INTENSE           # 600 of them intense -> 25%
    pred[150:170, 20:40] = MEDIUM          # two fragments 3 px apart -> one anomaly,
    pred[150:170, 43:63] = MEDIUM          #   800 px = 1% -> da monitorare
    pred[250:260, 150:160] = MEDIUM        # 100 px = 0.125% -> below threshold, no ID
    pred[320:340, 20:40] = MEDIUM          # two patches 30 px apart -> two anomalies,
    pred[320:340, 70:90] = MEDIUM          #   400 px = 0.5% each -> da monitorare
    return pred


class DilateAndLabel(unittest.TestCase):
    def test_dilate_single_pixel_is_square(self):
        m = np.zeros((9, 9), bool); m[4, 4] = True
        d = ac.dilate(m, 2)
        self.assertEqual(int(d.sum()), 25)
        self.assertTrue(d[2:7, 2:7].all())

    def test_diagonal_pixels_are_connected(self):
        m = np.zeros((3, 3), bool); m[0, 0] = m[1, 1] = m[2, 2] = True
        self.assertEqual(len(np.unique(ac.label(m)[m])), 1)

    def test_u_shape_merges_into_one_component(self):
        m = np.zeros((5, 7), bool)
        m[0:4, 1] = True; m[0:4, 5] = True; m[4, 1:6] = True
        self.assertEqual(len(np.unique(ac.label(m)[m])), 1)

    def test_separate_blobs_stay_separate(self):
        m = np.zeros((5, 9), bool); m[1:4, 1:3] = True; m[1:4, 6:8] = True
        self.assertEqual(len(np.unique(ac.label(m)[m])), 2)


class FindClusters(unittest.TestCase):
    def setUp(self):
        self.out = ac.find_clusters(tower())
        self.by_id = {c["id"]: c for c in self.out["clusters"]}

    def test_merge_distance_follows_tower_extent(self):
        self.assertEqual(self.out["tower_extent_px"], [0, 0, 200, 400])
        self.assertEqual(self.out["merge_distance_px"], 4)

    def test_ids_follow_size_and_bands(self):
        sizes = [c["rust_pixels"] for c in self.out["clusters"]]
        self.assertEqual(sizes, [2400, 800, 400, 400])
        self.assertEqual([c["band"] for c in self.out["clusters"]],
                         ["rilevante", "da monitorare", "da monitorare", "da monitorare"])
        self.assertEqual(self.out["below_threshold_count"], 1)

    def test_close_fragments_merge_and_area_counts_only_rust(self):
        a2 = self.by_id["A2"]
        self.assertEqual(a2["bbox_px"], [20, 150, 63, 170])
        self.assertEqual(a2["pct_of_visible_metal"], 1.0)

    def test_largest_anomaly_details(self):
        a1 = self.by_id["A1"]
        self.assertEqual(a1["pct_of_visible_metal"], 3.0)
        self.assertEqual(a1["intense_pct_of_anomaly"], 25.0)
        self.assertEqual(a1["extent_pct_of_tower_height"], 15.0)
        self.assertEqual(a1["bbox_px"], [20, 20, 60, 80])

    def test_no_rust_gives_no_anomalies(self):
        out = ac.find_clusters(np.full((50, 50), HEALTHY, np.uint8))
        self.assertEqual(out["clusters"], [])
        self.assertTrue(out["enabled"])

    def test_background_only_is_safe(self):
        out = ac.find_clusters(np.zeros((50, 50), np.uint8))
        self.assertEqual(out["clusters"], [])


if __name__ == "__main__":
    unittest.main()
