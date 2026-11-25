"""Per-pixel colour evidence for rust on lattice members.

Why this exists: at the detail setting one DINOv2 patch covers about 21x21 source
pixels while a lattice member is 10-25 pixels wide, so a patch token mostly
describes what is behind the member. Rust is first of all a colour change of the
metal itself, measurable per pixel inside a sharp metal mask. The DINOv2 patch
score is kept as context; this module adds the pixel-level evidence.
"""
import numpy as np

K_NEIGHBOURS = 3
SAMPLE_RADIUS_PX = 4      # around the operator's click, in source pixels
MIN_SAMPLE_PIXELS = 6


def rgb_to_lab(rgb):
    """sRGB uint8 (...,3) -> CIE L*a*b* float32, D65."""
    c = np.asarray(rgb, np.float32) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], np.float32)
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1).astype(np.float32)


def box_blur3(rgb):
    """3x3 mean: removes JPEG chroma noise without mixing a member with its background much."""
    a = np.asarray(rgb, np.float32)
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
    h, w = a.shape[:2]
    out = np.zeros_like(a)
    for dy in range(3):
        for dx in range(3):
            out += p[dy:dy + h, dx:dx + w]
    return out / 9.0


def sample_color(rgb, metal, x, y, radius=SAMPLE_RADIUS_PX):
    """Median Lab of the metal pixels around (x, y); None when the point is not on metal.

    rgb: HxWx3 uint8, metal: HxW bool, both in the same pixel grid as x, y.
    """
    h, w = metal.shape
    r = max(1, int(round(radius)))
    x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    keep = metal[y0:y1, x0:x1]
    if int(keep.sum()) < MIN_SAMPLE_PIXELS:
        return None
    return np.median(rgb_to_lab(rgb[y0:y1, x0:x1][keep]), axis=0).astype(np.float32)


def sample_color_box(rgb, metal, x0, y0, x1, y1):
    """Median Lab of the metal pixels in a rectangle (used to upgrade older profiles,
    which only remember the labelled patch and not the exact click)."""
    h, w = metal.shape
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
    if x0 >= x1 or y0 >= y1:
        return None
    keep = metal[y0:y1, x0:x1]
    if int(keep.sum()) < MIN_SAMPLE_PIXELS:
        return None
    return np.median(rgb_to_lab(rgb[y0:y1, x0:x1][keep]), axis=0).astype(np.float32)


# Decision constants, chosen on the operator's 117 labelled patches (two photos):
# leave-one-out and train-on-one-photo/test-on-the-other. Not a validated threshold.
L_WEIGHT = 0.0        # rust versus healthy ignores lightness: with any weight on it, the
                      # shaded faces of a sound white tower matched the dark-rust examples
MARGIN = 1.0          # minimum distance advantage (a*, b* units) to decide
FAR = 10.0            # farther than this from every example: colour never labelled
TOO_DARK_L = 20.0     # below this lightness the chroma of a JPEG pixel is noise
ID_HEALTHY, ID_MEDIUM, ID_INTENSE, ID_UNCERTAIN = 1, 2, 3, 4


def has_color_classes(bank):
    return "healthy" in bank and ("medium" in bank or "intense" in bank)


def classify_pixels(lab_pixels, bank):
    """Class id per metal pixel plus the raw rust decision before abstention.

    Rust versus healthy compares chromaticity (a*, b*) with the closest labelled
    examples; medium versus intense then uses full Lab, where lightness matters.
    """
    lab_pixels = np.asarray(lab_pixels, np.float32).reshape(-1, 3)
    scale = np.array([L_WEIGHT, 1, 1], np.float32)
    rust_names = [k for k in ("medium", "intense") if k in bank]
    weighted = {"healthy": bank["healthy"] * scale,
                "rust": np.concatenate([bank[k] for k in rust_names]) * scale}
    d = class_distances(lab_pixels * scale, weighted, ("healthy", "rust"))
    margin = d[0] - d[1]
    raw_rust = margin > 0
    if len(rust_names) == 2:
        level = class_distances(lab_pixels, bank, ("medium", "intense"))
        rust_id = np.where(level[0] <= level[1], ID_MEDIUM, ID_INTENSE)
    else:
        rust_id = np.full(len(lab_pixels), ID_MEDIUM if rust_names[0] == "medium" else ID_INTENSE)
    ids = np.where(margin >= MARGIN, rust_id, np.where(margin <= -MARGIN, ID_HEALTHY, ID_UNCERTAIN))
    ids[(d.min(axis=0) > FAR) | (lab_pixels[:, 0] < TOO_DARK_L)] = ID_UNCERTAIN
    return ids.astype(np.uint8), raw_rust, rust_id.astype(np.uint8)


def color_bank(samples, labels):
    """label -> (N,3) Lab array, from profile samples that carry a colour."""
    bank = {}
    for label in labels:
        rows = [s["color_lab"] for s in samples if s.get("label") == label and s.get("color_lab") is not None]
        if rows:
            bank[label] = np.asarray(rows, np.float32).reshape(-1, 3)
    return bank


def class_distances(lab_pixels, bank, names):
    """(K,N) mean Lab distance from each pixel to its closest examples of each class."""
    flat = np.asarray(lab_pixels, np.float32).reshape(-1, 3)
    out = np.full((len(names), len(flat)), np.inf, np.float32)
    for i, name in enumerate(names):
        ref = bank.get(name)
        if ref is None or not len(ref):
            continue
        k = min(K_NEIGHBOURS, len(ref))
        for start in range(0, len(flat), 65536):
            d = np.linalg.norm(flat[start:start + 65536, None, :] - ref[None], axis=2)
            out[i, start:start + 65536] = np.partition(d, k - 1, axis=1)[:, :k].mean(axis=1)
    return out
