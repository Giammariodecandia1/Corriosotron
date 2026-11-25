"""Group rust pixels into numbered anomalies (A1, A2, ...) for field inspection.

Size is measured relative to the tower in the photo, because the distance to the tower
(hence cm per pixel) is not reliably known:
- fragments closer than MERGE_FRACTION of the tower extent are merged into one anomaly
  (rust broken by bolts, joints or small sound areas on the same member);
- an anomaly's size is its real rust pixels as % of the visible metal, the same
  denominator as the headline corrosion percentage;
- bands: >= RELEVANT_PCT "rilevante", >= MONITOR_PCT "da monitorare", smaller ones get
  no ID. IDs follow size order, so A1 is always the largest.

Future: with a laser rangefinder distance (DJI XMP LRFTargetDistance > 0) pixel areas
could also be given in cm^2; see `physical_scale` in the result, currently always None.
"""
import numpy as np

MERGE_FRACTION = 0.01
RELEVANT_PCT = 2.0
MONITOR_PCT = 0.5
BANDS = (("rilevante", RELEVANT_PCT), ("da monitorare", MONITOR_PCT))


def dilate(mask, radius):
    """Square binary dilation by `radius` pixels (separable, via cumulative sums)."""
    if radius <= 0:
        return mask.copy()
    out = mask
    for axis in (0, 1):
        n = out.shape[axis]
        cs = np.cumsum(out, axis=axis, dtype=np.int32)
        pad_shape = list(out.shape)
        pad_shape[axis] = 1
        cs = np.concatenate([np.zeros(pad_shape, np.int32), cs], axis=axis)
        idx = np.arange(n)
        hi = np.minimum(idx + radius + 1, n)
        lo = np.maximum(idx - radius, 0)
        window = np.take(cs, hi, axis=axis) - np.take(cs, lo, axis=axis)
        out = window > 0
    return out


def label(mask):
    """8-connected component labels (0 = background), run-based union-find."""
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    prev = []
    for y in range(h):
        row = mask[y]
        if not row.any():
            prev = []
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)  # ends exclusive
        cur, j = [], 0
        for s, e in zip(starts.tolist(), ends.tolist()):
            while j < len(prev) and prev[j][1] < s:
                j += 1
            lab, k = 0, j
            while k < len(prev) and prev[k][0] <= e:
                root = find(prev[k][2])
                if lab == 0:
                    lab = root
                elif root != lab:
                    lo, hi = min(root, lab), max(root, lab)
                    parent[hi] = lo
                    lab = lo
                k += 1
            if lab == 0:
                parent.append(len(parent))
                lab = len(parent) - 1
            labels[y, s:e] = lab
            cur.append((s, e, lab))
        prev = cur
    roots = np.array([find(i) for i in range(len(parent))], np.int32)
    return roots[labels]


def band_for(pct):
    for name, threshold in BANDS:
        if pct >= threshold:
            return name
    return None


def find_clusters(pred, rust_ids=(2, 3), intense_id=3, metal_ids=(1, 2, 3, 4)):
    """pred: class-id map (stable ids of main.classify). Returns a JSON-ready dict."""
    metal = np.isin(pred, metal_ids)
    rust = np.isin(pred, rust_ids)
    metal_pixels = int(metal.sum())
    result = {"enabled": True, "merge_fraction_of_tower": MERGE_FRACTION,
              "thresholds_pct_of_visible_metal": {name: t for name, t in BANDS},
              "tower_extent_px": None, "merge_distance_px": None, "physical_scale": None,
              "clusters": [], "below_threshold_count": 0}
    if not metal_pixels or not rust.any():
        return result
    ys, xs = np.nonzero(metal)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    tower_h = y1 - y0
    # Dilating both fragments by `radius` bridges gaps up to 2 * radius, so the radius is
    # half of the merge distance (MERGE_FRACTION of the tower extent).
    radius = max(1, round(MERGE_FRACTION * float(np.hypot(x1 - x0, tower_h)) / 2))
    result["tower_extent_px"] = [x0, y0, x1, y1]
    result["merge_distance_px"] = int(2 * radius)

    groups = label(dilate(rust, radius))
    ry, rx = np.nonzero(rust)
    ids, inverse = np.unique(groups[ry, rx], return_inverse=True)
    n = len(ids)
    size = np.bincount(inverse, minlength=n)
    intense = np.bincount(inverse, weights=(pred[ry, rx] == intense_id), minlength=n)
    bx0 = np.full(n, rx.max()); np.minimum.at(bx0, inverse, rx)
    by0 = np.full(n, ry.max()); np.minimum.at(by0, inverse, ry)
    bx1 = np.zeros(n, int); np.maximum.at(bx1, inverse, rx)
    by1 = np.zeros(n, int); np.maximum.at(by1, inverse, ry)

    numbered, below = [], 0
    for i in np.argsort(-size, kind="stable"):
        pct = 100 * float(size[i]) / metal_pixels
        band = band_for(pct)
        if band is None:
            below += 1
            continue
        numbered.append({
            "id": f"A{len(numbered) + 1}", "band": band,
            "rust_pixels": int(size[i]),
            "pct_of_visible_metal": round(pct, 2),
            "intense_pct_of_anomaly": round(100 * float(intense[i]) / float(size[i]), 1),
            "extent_pct_of_tower_height": round(100 * float(by1[i] + 1 - by0[i]) / tower_h, 1),
            "bbox_px": [int(bx0[i]), int(by0[i]), int(bx1[i]) + 1, int(by1[i]) + 1],
        })
    result["clusters"] = numbered
    result["below_threshold_count"] = below
    return result
