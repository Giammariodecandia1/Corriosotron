"""Automatic exclusion of power cables from the ROI metal mask (optional, experimental).

1. Tower zone: the lattice is dense metal, cables (even a few side by side) are sparse. Windows with
   enough metal -> blobs -> convex hull, slightly grown.
2. Cables: long straight (or slightly sagging) lines of metal OUTSIDE the tower zone.
3. A line seen outside on BOTH sides of the tower crosses it ("enters and exits"): only then it is
   followed inside the zone, and only where the metal across it is as thin as the cable (a crossing
   or parallel lattice member is thicker and is kept). Lines that end at the tower are removed
   outside only.

Limits: it needs some background around the tower inside the ROI (with a tight ROI few cables are
visible outside); curved jumpers/drops near the cross-arms are not straight lines and stay.
"""
import math

import cv2
import numpy as np


def _odd(v):
    v = int(round(v))
    return v + 1 if v % 2 == 0 else v


def tower_zone(M, density=0.14):
    h, w = M.shape
    diag = math.hypot(w, h)
    k = _odd(0.035 * diag)
    dens = cv2.blur(M.astype(np.float32), (k, k))
    core = (dens >= density).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core)
    keep = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 0.004 * w * h:
            keep[lab == i] = 1
    ys, xs = np.nonzero(keep)
    if len(xs) == 0:
        return np.zeros((h, w), bool)
    hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
    zone = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(zone, hull, 1)
    zone = cv2.dilate(zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(0.01 * diag),) * 2))
    return zone > 0


def _groups(M, T):
    h, w = M.shape
    diag = math.hypot(w, h)
    out = (M & ~T).astype(np.uint8) * 255
    segs = cv2.HoughLinesP(out, 1, np.pi / 720, threshold=int(0.02 * diag), minLineLength=0.035 * diag, maxLineGap=0.04 * diag)
    if segs is None:
        return [], out
    groups = []
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4).astype(float):
        ang = math.atan2(y2 - y1, x2 - x1) % math.pi
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        for g in groups:
            d = abs(ang - g["ang"]); d = min(d, math.pi - d)
            gx, gy = -math.sin(g["ang"]), math.cos(g["ang"])
            if d < math.radians(2.0) and abs((mx - g["x"]) * gx + (my - g["y"]) * gy) < 0.008 * diag:
                g["segs"].append((x1, y1, x2, y2))
                break
        else:
            groups.append({"ang": ang, "x": x1, "y": y1, "segs": [(x1, y1, x2, y2)]})
    return groups, out


def _cables(groups, out, T):
    h, w = T.shape
    diag = math.hypot(w, h)
    dist = cv2.distanceTransform(out, cv2.DIST_L2, 3)
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
    cables = []
    for g in groups:
        L = sum(math.hypot(s[2] - s[0], s[3] - s[1]) for s in g["segs"])
        ang = g["ang"]; ux, uy = math.cos(ang), math.sin(ang); nx, ny = -uy, ux
        pts = []
        for x1, y1, x2, y2 in g["segs"]:
            n = int(math.hypot(x2 - x1, y2 - y1))
            for t in np.linspace(0, 1, max(2, n // 2)):
                pts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
        pts = np.array(pts)
        u = pts[:, 0] * ux + pts[:, 1] * uy
        v = pts[:, 0] * nx + pts[:, 1] * ny
        coef = np.polyfit(u, v, 2 if (u.max() - u.min()) > 0.3 * diag else 1)
        if np.percentile(np.abs(np.polyval(coef, u) - v), 90) > 0.006 * diag:
            continue
        us = corners @ np.array([ux, uy])
        uu = np.arange(us.min(), us.max(), 2.0); vv = np.polyval(coef, uu)
        xs, ys = uu * ux + vv * nx, uu * uy + vv * ny
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        inside = np.zeros(len(uu), bool)
        inside[ok] = T[ys[ok].astype(int), xs[ok].astype(int)]
        crosses = False
        if inside.any():
            a, b = uu[inside].min(), uu[inside].max()
            crosses = bool((u < a - 0.01 * diag).any() and (u > b + 0.01 * diag).any())
        reach_border = (u.min() - us.min() < 0.02 * diag) or (us.max() - u.max() < 0.02 * diag)
        if not crosses and L < (0.06 if reach_border else 0.10) * diag:
            continue
        xi = np.clip(pts[:, 0].astype(int), 0, w - 1); yi = np.clip(pts[:, 1].astype(int), 0, h - 1)
        dv = dist[yi, xi]; dv = dv[dv > 0]
        width = float(np.clip(2 * np.median(dv) + 1, 2, 0.008 * diag)) if len(dv) else 3.0
        cables.append({"ang": ang, "coef": coef, "width": width, "crosses": crosses})
    return cables


def _band(M, cables, T, margin=2):
    h, w = M.shape
    out = np.zeros((h, w), bool)
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
    for c in cables:
        ux, uy = math.cos(c["ang"]), math.sin(c["ang"]); nx, ny = -uy, ux
        us = corners @ np.array([ux, uy])
        u = np.arange(us.min(), us.max(), 1.0); v = np.polyval(c["coef"], u)
        half = c["width"] / 2 + margin
        thin_max = c["width"] * 1.6 + 2
        reach = int(math.ceil(thin_max + 6)); offs = np.arange(-reach, reach + 1); sel = np.abs(offs) <= half
        for x, y in zip(u * ux + v * nx, u * uy + v * ny):
            ix, iy = int(round(x)), int(round(y))
            if not (0 <= ix < w and 0 <= iy < h):
                continue
            px = np.clip(np.round(x + offs * nx).astype(int), 0, w - 1)
            py = np.clip(np.round(y + offs * ny).astype(int), 0, h - 1)
            if T[iy, ix]:
                if not c["crosses"]:
                    continue
                prof = M[py, px]
                if not prof[reach - 1:reach + 2].any():
                    continue
                lo = hi = reach
                while lo > 0 and prof[lo - 1]:
                    lo -= 1
                while hi < len(prof) - 1 and prof[hi + 1]:
                    hi += 1
                if hi - lo + 1 > thin_max:
                    continue
            out[py[sel], px[sel]] = True
    return out & M


def find_cables(metal):
    """metal: bool ROI-size metal mask. Returns (cable mask bool, info dict)."""
    M = cv2.erode(metal.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    T = tower_zone(M)
    groups, out = _groups(M, T)
    cables = _cables(groups, out, T)
    C = _band(M, cables, T) if cables else np.zeros(M.shape, bool)
    metal_px = int(M.sum())
    return C, {"cables": len(cables), "crossing_tower": sum(c["crosses"] for c in cables),
               "widths_px": [round(c["width"], 1) for c in cables],
               "pixels_removed": int(C.sum()), "pixels_removed_inside_tower": int((C & T).sum()),
               "pct_of_metal": round(100 * float(C.sum()) / max(1, metal_px), 2)}
