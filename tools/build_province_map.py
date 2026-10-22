"""Assign every Vic3 province to an AoE (Vic2) province and state.

Vic2 and Vic3 use different map projections, so we build a smooth warp from
Vic3 pixel space to Vic2 pixel space in two steps:

1. A thin-plate spline through alignment points. The points come from the
   Paradox Game Converters' EU4->Vic2 and EU4->Vic3 province mappings (MIT
   licensed): a group of EU4 provinces that both converters map as a unit
   gives one matching point on each map. Vanilla Vic2 and AoE share a pixel
   grid, so a warp fitted on vanilla Vic2 applies to AoE unchanged.
2. A coastline refinement: warped Vic2 land and sea are nudged onto Vic3's,
   fixing coasts the spline misses while the interior stays as the spline
   put it.

AoE land the warp lays on another landmass than its own (Alaska over the tip
of Chukotka, which AoE's map cuts at 180°) is dropped, so each Vic3 landmass
only takes AoE land from the Vic2 landmass it pairs with (see Landmasses).

Each Vic3 land province then takes the AoE state province covering most of
its pixels, or if it lands in AoE sea or wasteland the one its land is
nearest, on its own landmass and across the map's x seam. Every non-empty
AoE state is guaranteed at least one Vic3 province.

Outputs (build/):
  province_map.csv      Vic3 province -> AoE province and state
  aoe_at_vic3.npy       AoE province id under every Vic3 pixel (0 = none, or AoE land of another landmass)
  warp_grid.npz         the final warp sampled on a grid of Vic3 pixels
  warp_points.csv       alignment points with cross-validated residuals
  debug_states.png      Vic3 map coloured by AoE state
  debug_coast.png       land/sea agreement between the warped AoE map and Vic3
redraw_provinces.py then redraws the province borders and rewrites province_map.csv and debug_states.png for
the redrawn map.
"""
import csv
import time
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, map_coordinates

import config
import pdx
from mapdata import label_centroids, province_borders, read_vic2_map, read_vic3_map, sums_to_centroid

COORD_SCALE = 1000.0   # coordinates are divided by this before RBF fitting
BASE_STEP = 8          # Vic3 pixels between samples of the spline warp
REFINE_STEP = 4        # Vic3 pixels per cell of the region refinement
REGION_SEARCH = 4         # cells searched around a mislabelled cell
MOVE_MIN_SHARE = 0.1      # a border province may switch to an AoE province covering this share of it
TOPOLOGY_WEIGHT = 400     # pixels of mismatch traded per topology violation fixed
AOE_BRIDGE = 3            # AoE water this narrow (pixels) doesn't separate provinces
VANILLA_PARTNER_SHARE = 0.6  # an AoE state 'partners' a vanilla state when it holds this much of it
VANILLA_WEIGHT = 0.25     # share of a province's pixels a vanilla-border tiebreak is worth
BALANCE_WEIGHT = 0.5      # pixels of area error traded per mismatched pixel when balancing
COVER_SEARCH_LIMIT = 300  # AoE provinces explored when looking for a swap chain
REGION_SCHEDULE = [(4.0, 40), (2.5, 40), (2.0, 30)]  # (smoothing sigma in cells, iterations)


def log(msg, _t0=time.time()):
    print(f"[{time.time() - _t0:6.1f}s] {msg}", flush=True)


def norm_hex(p):
    return "x" + p[1:].upper()


# ------------------------------------------------------- control points ----

def read_converter_links(path, target_key):
    """Latest mapping block as {frozenset(eu4 ids): [target ids]}."""
    root = pdx.parse_file(path)
    latest = next(value for _, _, value in root.pairs() if isinstance(value, pdx.Block))
    groups = {}
    for link in latest.get_all("link"):
        eu4 = frozenset(int(v) for v in link.get_all("eu4"))
        targets = link.get_all(target_key)
        if eu4 and targets:
            groups.setdefault(eu4, []).extend(targets)
    return groups


def control_points(v2_vanilla, v3):
    """[(x3, y3, x2, y2, uncertainty, eu4 ids)]; uncertainty grows with group spread."""
    v2_links = read_converter_links(config.EU4_TO_VIC2, "v2")
    v3_links = read_converter_links(config.EU4_TO_VIC3, "vic3")
    shared = [k for k in v2_links if k in v3_links]
    log(f"converter groups: {len(v2_links)} Vic2, {len(v3_links)} Vic3, {len(shared)} shared")

    n2 = int(v2_vanilla.labels.max()) + 1
    s2 = label_centroids(v2_vanilla.labels, n2)
    s3 = label_centroids(v3.labels, len(v3.provinces))
    v3_index = {p: i for i, p in enumerate(v3.provinces)}

    rows = []
    for key in shared:
        i2 = [int(p) for p in v2_links[key] if p.isdigit() and int(p) < n2 and int(p) not in v2_vanilla.sea]
        i3 = [v3_index[norm_hex(p)] for p in v3_links[key] if norm_hex(p) in v3_index]
        if not i2 or not i3:
            continue
        a = [s[i2].sum() for s in s2]
        b = [s[i3].sum() for s in s3]
        if a[0] == 0 or b[0] == 0:
            continue
        x2, y2, spread2 = sums_to_centroid(*a, v2_vanilla.width)
        x3, y3, spread3 = sums_to_centroid(*b, v3.width)
        if spread2 > 200 or spread3 > 300:
            continue
        uncertainty = 1.0 + (spread2 / 30.0) ** 2
        rows.append((x3, y3, x2, y2, uncertainty, sorted(key)))
    log(f"usable alignment points: {len(rows)}")
    return rows


# ---------------------------------------------------------- spline warp ----

class SplineWarp:
    """Smooth map from Vic3 pixel coordinates to (unwrapped) Vic2 pixel coordinates."""

    def __init__(self, src, dst, weight, w3, w2, smoothing):
        # Repeat points near the map seam one world-width over so the fit
        # stays continuous across the wrap.
        left, right = src[:, 0] < 0.15 * w3, src[:, 0] > 0.85 * w3
        src = np.vstack([src, src[left] + [w3, 0], src[right] - [w3, 0]])
        dst = np.vstack([dst, dst[left] + [w2, 0], dst[right] - [w2, 0]])
        weight = np.concatenate([weight, weight[left], weight[right]])
        self.rbf = RBFInterpolator(src / COORD_SCALE, dst / COORD_SCALE,
                                   kernel="thin_plate_spline", smoothing=smoothing * weight)

    def __call__(self, pts, chunk=4000):
        pts = np.asarray(pts, dtype=float) / COORD_SCALE
        out = np.empty_like(pts)
        for i in range(0, len(pts), chunk):
            out[i:i + chunk] = self.rbf(pts[i:i + chunk])
        return out * COORD_SCALE


def unwrap_targets(src, dst, w3, w2):
    """Make Vic2 x continuous in Vic3 x (the two maps put the seam in different places)."""
    scale = w2 / w3
    u = 2 * np.pi * (dst[:, 0] - src[:, 0] * scale) / w2
    offset = (np.arctan2(np.sin(u).mean(), np.cos(u).mean()) % (2 * np.pi)) / (2 * np.pi) * w2
    pred = src[:, 0] * scale + offset
    resid = (dst[:, 0] - pred + w2 / 2) % w2 - w2 / 2
    return np.column_stack([pred + resid, dst[:, 1]])


def cv_residuals(src, dst, weight, w3, w2, smoothing, folds=5, seed=0):
    order = np.random.default_rng(seed).permutation(len(src))
    resid = np.empty(len(src))
    for f in range(folds):
        test = order[f::folds]
        train = np.setdiff1d(order, test)
        pred = SplineWarp(src[train], dst[train], weight[train], w3, w2, smoothing)(src[test])
        resid[test] = np.hypot(*(pred - dst[test]).T)
    return resid


def fit_spline(points, w3, w2):
    src = np.array([(p[0], p[1]) for p in points])
    dst = unwrap_targets(src, np.array([(p[2], p[3]) for p in points]), w3, w2)
    weight = np.array([p[4] for p in points])

    best = None
    for smoothing in (1e-3, 1e-2, 1e-1, 1.0):
        r = cv_residuals(src, dst, weight, w3, w2, smoothing) / np.sqrt(weight)
        score = np.percentile(r, 90)
        log(f"  smoothing {smoothing:g}: weighted CV residual 90th pct {score:.2f}px")
        if best is None or score < best[1]:
            best = (smoothing, score)
    smoothing = best[0]

    keep = np.ones(len(src), bool)
    for _ in range(2):
        r = cv_residuals(src[keep], dst[keep], weight[keep], w3, w2, smoothing) / np.sqrt(weight[keep])
        cut = max(4 * np.median(r), 10.0)
        idx = np.flatnonzero(keep)
        keep[idx[r > cut]] = False
        log(f"  outlier pass: dropped {(r > cut).sum()} points")
    resid = np.full(len(src), np.nan)
    resid[keep] = cv_residuals(src[keep], dst[keep], weight[keep], w3, w2, smoothing)
    log(f"spline: smoothing {smoothing:g}, {keep.sum()} points, CV residual median "
        f"{np.nanmedian(resid):.2f} / 90th {np.nanpercentile(resid, 90):.2f} Vic2 px")
    return SplineWarp(src[keep], dst[keep], weight[keep], w3, w2, smoothing), src, dst, keep, resid


# -------------------------------------------------------- warp on grid ----

class GridWarp:
    """A warp sampled at Vic3 pixel positions origin + i*step, interpolated bilinearly.

    Coordinates everywhere are pixel-index coordinates: pixel i spans [i - 0.5, i + 0.5].
    """

    def __init__(self, x, y, step, origin, w3, w2):
        self.x, self.y, self.step, self.origin, self.w3, self.w2 = x, y, step, origin, w3, w2

    def __call__(self, px, py):
        # Wrap px into the map and shift the Vic2 result by whole world widths.
        k = np.floor(px / self.w3)
        coords = [(py - self.origin) / self.step, (px - k * self.w3 - self.origin) / self.step]
        x2 = map_coordinates(self.x, coords, order=1, mode="nearest") + k * self.w2
        y2 = map_coordinates(self.y, coords, order=1, mode="nearest")
        return x2, y2


def spline_grid(spline, w3, h3, w2):
    mx, my = np.meshgrid(np.arange(0, w3 + BASE_STEP, BASE_STEP, dtype=float),
                         np.arange(0, h3 + BASE_STEP, BASE_STEP, dtype=float))
    out = spline(np.column_stack([mx.ravel(), my.ravel()]))
    return GridWarp(out[:, 0].reshape(mx.shape), out[:, 1].reshape(mx.shape), BASE_STEP, 0.0, w3, w2)


def region_labels(v2_vanilla, v3, land3):
    """Label every Vic2 and Vic3 province 1 (land) or 0 (sea); -1 (Vic3 lakes, unknown colours) is ignored.

    Only coastlines are matched. Matching the converters' province groups as well dragged land out of place
    wherever a group's two sides cover different ground (EU4 Kharkov is Kharkov and Luhansk in Vic2 but a
    few provinces by Poltava in Vic3), and the warp fitted Vic3's rivers and coasts worse for it.
    """
    n2 = int(v2_vanilla.labels.max()) + 1
    g2 = np.ones(n2, np.int32)
    g2[[p for p in v2_vanilla.sea if p < n2]] = 0
    g2[0] = -1
    g3 = np.where(land3, 1, -1).astype(np.int32)
    v3_index = {p: i for i, p in enumerate(v3.provinces)}
    sea3 = [v3_index[norm_hex(p)] for b in v3.sea_states.values() for p in b.get("provinces").values()]
    g3[sea3] = 0
    return g2, g3


def region_cells(v3, g3, s):
    """Vic3 region label at the centre of each REFINE_STEP cell."""
    h, w = v3.height // s, v3.width // s
    return g3[v3.labels[s // 2::s, s // 2::s][:h, :w]]


def warped_cells(warp_fn, g2_img, hc, wc, s, w2, h2, ux=0.0, uy=0.0):
    """Vic2 region label under each cell after warping (-1 outside the Vic2 map)."""
    cy, cx = np.mgrid[0:hc, 0:wc].astype(float)
    x2, y2 = warp_fn((cx + ux + 0.5) * s - 0.5, (cy + uy + 0.5) * s - 0.5)
    yi, xi = np.rint(y2).astype(np.int64), np.rint(x2).astype(np.int64) % w2
    inside = (yi >= 0) & (yi < h2)
    out = np.full((hc, wc), -1, np.int32)
    out[inside] = g2_img[yi[inside], xi[inside]]
    return out


def region_agreement(fixed, moving):
    """Share of labelled Vic3 cells whose warped Vic2 label matches."""
    valid = (fixed >= 0) & (moving >= 0)
    return (fixed[valid] == moving[valid]).mean()


def refine_with_regions(base, fixed, g2_img, w2, h2):
    """Displace Vic3 sample positions so warped Vic2 region labels (region_labels: land and sea) line up
    with Vic3's.

    Each iteration, every cell whose warped label is wrong looks for the nearest
    nearby displacement that would give it the right label; cells already right
    hold still. The accumulated displacement is Gaussian-smoothed (demons style).
    """
    s = REFINE_STEP
    hc, wc = fixed.shape
    cy, cx = np.mgrid[0:hc, 0:wc].astype(float)
    ux, uy = np.zeros((hc, wc)), np.zeros((hc, wc))
    r = REGION_SEARCH
    offsets = sorted(((dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                      if 0 < dx * dx + dy * dy <= r * r), key=lambda d: d[0] ** 2 + d[1] ** 2)
    labelled = fixed >= 0
    for sigma, iters in REGION_SCHEDULE:
        for _ in range(iters):
            moving = warped_cells(base, g2_img, hc, wc, s, w2, h2, ux, uy)
            todo = labelled & (moving != fixed)
            fx, fy = np.zeros((hc, wc)), np.zeros((hc, wc))
            for dy, dx in offsets:
                shifted = np.roll(moving, (-dy, -dx), axis=(0, 1))
                if dy > 0:
                    shifted[-dy:] = -2
                elif dy < 0:
                    shifted[:-dy] = -2
                hit = todo & (shifted == fixed)
                fx[hit], fy[hit] = dx, dy
                todo &= ~hit
            ux = gaussian_filter(ux + fx, sigma, mode=("nearest", "wrap"))
            uy = gaussian_filter(uy + fy, sigma, mode=("nearest", "wrap"))
        moving = warped_cells(base, g2_img, hc, wc, s, w2, h2, ux, uy)
        log(f"  region refinement sigma {sigma}: agreement {region_agreement(fixed, moving):.2%}, "
            f"max shift {np.hypot(ux, uy).max() * s:.0f} Vic3 px")
    x2, y2 = base((cx + ux + 0.5) * s - 0.5, (cy + uy + 0.5) * s - 0.5)
    return GridWarp(x2, y2, s, s / 2 - 0.5, base.w3, w2)


# ---------------------------------------------------------- assignment ----

def aoe_under_vic3(warp, aoe, w3, h3):
    """AoE province id under every Vic3 pixel centre (0 outside the Vic2 map)."""
    out = np.zeros((h3, w3), dtype=np.int16)
    cols = np.arange(w3, dtype=float)
    for y0 in range(0, h3, 256):
        rows = np.arange(y0, min(y0 + 256, h3), dtype=float)
        ry, rx = np.meshgrid(rows, cols, indexing="ij")
        x2, y2 = warp(rx.ravel(), ry.ravel())
        xi = np.rint(x2).astype(np.int64) % aoe.width
        yi = np.rint(y2).astype(np.int64)
        inside = (yi >= 0) & (yi < aoe.height)
        ids = np.zeros(len(xi), np.int16)
        ids[inside] = aoe.labels[yi[inside], xi[inside]]
        out[y0:y0 + len(rows)] = ids.reshape(len(rows), w3)
    return out


def landmasses(mask):
    """Connected components of mask (4-connected), joined across the x seam; 0 outside mask."""
    lab, n = label(mask)
    parent = np.arange(n + 1)

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    a, b = lab[:, 0], lab[:, -1]
    both = (a > 0) & (b > 0)
    for x, y in set(zip(a[both].tolist(), b[both].tolist())):
        rx, ry = root(x), root(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)
    return np.array([root(x) for x in range(n + 1)], np.int32)[lab]


def top_pairs(of, to, n_to):
    """{x: the value of `to` most often paired with x} over the pairs (of[i], to[i]); values below n_to."""
    uniq, counts = np.unique(of * n_to + to, return_counts=True)
    x, y = uniq // n_to, uniq % n_to
    order = np.lexsort((-counts, x))
    first = np.r_[True, x[order][1:] != x[order][:-1]]
    return dict(zip(x[order][first].tolist(), y[order][first].tolist()))


class Landmasses:
    """Vic3 landmasses paired with AoE's. A Vic2 landmass pairs with the Vic3 landmass most of it is warped onto,
    when that Vic3 landmass gets most of its AoE land from it. A Vic2 island that Vic3 joins to another landmass
    (Kutch, Baffin Island) pairs with nothing.
      mass    Vic3 landmass per Vic3 pixel (0 = sea)
      home    per AoE province: the Vic3 landmass its Vic2 landmass pairs with (0 = none)
      paired  per Vic3 landmass: whether it pairs with a Vic2 landmass"""

    def __init__(self, v3, land3, aoe, aoe_at):
        k = int(aoe.labels.max()) + 1
        valid = np.zeros(k, bool)
        valid[[p for p in aoe.state_of() if p < k and p not in aoe.sea]] = True
        self.mass = landmasses(land3[v3.labels])
        mass2 = landmasses(valid[aoe.labels])
        n2, n3 = int(mass2.max()) + 1, int(self.mass.max()) + 1
        sel = mass2 > 0
        of2 = np.zeros(k, np.int64)                    # each AoE province's Vic2 landmass (most of its pixels)
        main2 = top_pairs(aoe.labels[sel].astype(np.int64), mass2[sel].astype(np.int64), n2)
        of2[list(main2)] = list(main2.values())
        a = aoe_at.astype(np.int64)
        on = (self.mass > 0) & valid[a]
        x2, x3 = of2[a[on]], self.mass[on].astype(np.int64)
        to3, to2 = top_pairs(x2, x3, n3), top_pairs(x3, x2, n2)
        pairs = {m2: m3 for m2, m3 in to3.items() if to2.get(m3) == m2}
        self.paired = np.zeros(n3, bool)
        self.paired[list(pairs.values())] = True
        lut = np.zeros(int(mass2.max()) + 1, np.int32)
        lut[list(pairs)] = list(pairs.values())
        self.home = lut[of2]

    def foreign(self, ids, mass):
        """Where AoE provinces ids lie on Vic3 landmasses mass that pair with another Vic2 landmass than theirs."""
        h = self.home[ids]
        return self.paired[mass] & (h > 0) & (h != mass)


def keep_to_landmass(aoe_at, lm):
    """aoe_at without AoE land warped onto a Vic3 landmass paired with another (Alaska over Chukotka's tip, which
    AoE's map cuts at 180°); those pixels count as AoE sea, so they take the nearest AoE land of their own."""
    out = aoe_at.copy()
    out[lm.foreign(aoe_at, lm.mass)] = 0
    return out


def fill_nearest(ids, known, need, lm, band=512, margin=256):
    """ids (AoE province per Vic3 pixel) with every need pixel set to the id of the nearest known pixel, across
    the x seam too. Where that comes from a landmass paired with another than the pixel's own, the nearest id
    at home there (or of no landmass's pair) within margin is taken instead: the warp puts Chukotka's tip in the
    Bering Sea, nearer Alaska than Navarin."""
    h, w = ids.shape
    out = ids.copy()
    cols = np.r_[w - margin:w, 0:w, 0:margin]
    for y0 in range(0, h, band):
        y1 = min(h, y0 + band)
        n = need[y0:y1]
        if not n.any():
            continue
        a, b = max(0, y0 - margin), min(h, y1 + margin)
        k = known[a:b][:, cols]
        if not k.any():
            continue
        v = ids[a:b][:, cols]
        iy, ix = distance_transform_edt(~k, return_distances=False, return_indices=True)
        got = v[iy[y0 - a:y1 - a, margin:margin + w], ix[y0 - a:y1 - a, margin:margin + w]]
        mass = lm.mass[y0:y1]
        wrong = n & lm.foreign(got, mass)
        hv = lm.home[v]
        for m in np.unique(mass[wrong]).tolist():
            ys, xs = np.nonzero(wrong & (mass == m))
            r0, r1 = max(0, ys.min() + y0 - a - margin), min(b - a, ys.max() + y0 - a + margin + 1)
            c0, c1 = xs.min(), xs.max() + 2 * margin + 1
            seeds = k[r0:r1, c0:c1] & ((hv[r0:r1, c0:c1] == 0) | (hv[r0:r1, c0:c1] == m))
            if seeds.any():
                jy, jx = distance_transform_edt(~seeds, return_distances=False, return_indices=True)
                py, px = ys + y0 - a - r0, xs + margin - c0
                got[ys, xs] = v[r0:r1, c0:c1][jy[py, px], jx[py, px]]
        out[y0:y1][n] = got[n]
    return out


def assign(v3, aoe, aoe_at, warp, land3, group=None, fixed=None):
    """group (optional): {AoE province: group key}, such as redraw_provinces.py's (state, owner) units. When given,
    the coverage and topology steps keep every Vic3 province with an AoE province of its majority group (see
    cover_and_balance). fixed (optional): {Vic3 province index: AoE province} that skip the vote, such as the
    provinces the redraw carved for a trading post."""
    n3 = len(v3.provinces)
    state_of = aoe.state_of()
    k = int(aoe.labels.max()) + 1
    valid = np.zeros(k, bool)
    valid[[p for p in state_of if p < k and p not in aoe.sea]] = True

    keys = []
    for y0 in range(0, v3.height, 256):
        lab = v3.labels[y0:y0 + 256].ravel()
        a = aoe_at[y0:y0 + 256].ravel().astype(np.int64)
        sel = land3[lab] & valid[a]
        keys.append(lab[sel].astype(np.int64) * k + a[sel])
    uniq, counts = np.unique(np.concatenate(keys), return_counts=True)
    v3i, aid = uniq // k, uniq % k
    order = np.lexsort((-counts, v3i))
    first = np.r_[True, v3i[order][1:] != v3i[order][:-1]]
    best = dict(zip(v3i[order][first].tolist(), aid[order][first].tolist()))
    best_count = dict(zip(v3i[order][first].tolist(), counts[order][first].tolist()))
    method = {i: "majority" for i in best}
    pixel_count = np.bincount(v3.labels.ravel(), minlength=n3)

    # Vic3 province centroids, warped into Vic2 pixel space.
    cx, cy, _ = sums_to_centroid(*label_centroids(v3.labels, n3), v3.width)
    wx, wy = warp(cx, cy)
    wx %= aoe.width

    # Vic3 provinces that land in AoE sea/wasteland: the AoE province most of their land gets from the nearest
    # AoE land (fill_nearest, as the redraw fills it), else the AoE state pixel nearest their warped centroid.
    dist, (ny, nx) = distance_transform_edt(~valid[aoe.labels], return_indices=True)
    missing = [i for i in np.flatnonzero(land3) if i not in best]
    if missing:
        need = np.zeros(n3, bool)
        need[missing] = True
        need = need[v3.labels]
        got = fill_nearest(aoe_at, valid[aoe_at], need, Landmasses(v3, land3, aoe, aoe_at))[need].astype(np.int64)
        near = top_pairs(v3.labels[need][valid[got]].astype(np.int64), got[valid[got]], k)
        xi = np.rint(wx[missing]).astype(int) % aoe.width
        yi = np.clip(np.rint(wy[missing]).astype(int), 0, aoe.height - 1)
        for i, y, x in zip(missing, yi, xi):
            best[i] = near.get(i, int(aoe.labels[ny[y, x], nx[y, x]]))
            best_count[i] = 0
            method[i] = "nearest"
    for i, a in (fixed or {}).items():
        best[i], best_count[i], method[i] = a, int(counts[(v3i == i) & (aid == a)].sum()), "redraw"

    # Province adjacency on both maps. AoE's is taken across water up to
    # AOE_BRIDGE pixels wide, so rivers drawn as water (the Danube) don't count
    # as separating the provinces on either bank.
    bridged = aoe.labels.copy()
    near = ~valid[bridged] & (dist <= AOE_BRIDGE)
    bridged[near] = aoe.labels[ny[near], nx[near]]
    aoe_adjacent, aoe_adjacent_strong = set(), set()
    for (a, b), (length, _, _) in province_borders(bridged).items():
        if valid[a] and valid[b]:
            aoe_adjacent.add((a, b))
            if length >= 3:
                aoe_adjacent_strong.add((a, b))
    neighbours = defaultdict(list)
    for (i, j), (length, _, _) in province_borders(v3.labels).items():
        if i in best and j in best:
            neighbours[i].append((j, length))
            neighbours[j].append((i, length))
    vanilla_state = v3.province_state()
    vanilla_of = {i: vanilla_state.get(p) for i, p in enumerate(v3.provinces) if i in best}
    covered, switched = cover_and_balance(best, best_count, method, v3i, aid, counts, pixel_count,
                                          neighbours, aoe_adjacent, aoe_adjacent_strong, vanilla_of, state_of, group)

    # Small AoE states (enclaves, trading posts) can lose every vote. Give each
    # one the Vic3 province where it has the largest share of pixels, or failing
    # that the nearest Vic3 province, taking it from a state with provinces to spare.
    state_provs = {}
    for i, a in best.items():
        state_provs.setdefault(state_of[a], []).append(i)
    a_stats = label_centroids(aoe.labels, k)
    land_idx = np.flatnonzero(land3)
    rescued = []
    for state, provs in aoe.states.items():
        if not provs or state in state_provs:
            continue
        mine = np.flatnonzero(np.isin(aid, provs))
        candidates = sorted(((counts[j] / pixel_count[v3i[j]], int(v3i[j]), int(aid[j])) for j in mine),
                            reverse=True)
        sx, sy, _ = sums_to_centroid(*(s[provs].sum() for s in a_stats), aoe.width)
        dx = (wx[land_idx] - sx + aoe.width / 2) % aoe.width - aoe.width / 2
        nearest = land_idx[np.argsort(np.hypot(dx, wy[land_idx] - sy))[:20]]
        biggest = max(provs, key=lambda p: a_stats[0][p])
        candidates += [(0.0, int(i), biggest) for i in nearest]
        for share, i, a in candidates:
            donor = state_of[best[i]]
            if len(state_provs[donor]) > 1:
                state_provs[donor].remove(i)
                best[i], best_count[i], method[i] = a, int(share * pixel_count[i]), "enclave"
                state_provs[state] = [i]
                rescued.append(state)
                break
    log(f"assigned {len(best)} Vic3 land provinces ({len(missing)} by nearest AoE province, "
        f"{covered} small AoE provinces covered, {switched} switched for area/topology, "
        f"{len(rescued)} enclave states rescued)")
    return best, best_count, pixel_count, method, state_of, (cx, cy, wx, wy)


def cover_and_balance(best, best_count, method, v3i, aid, counts, pixel_count,
                      neighbours, aoe_adjacent, aoe_adjacent_strong, vanilla_of, state_of, group=None):
    """Adjust the majority vote so AoE provinces keep their size and topology.

    Majority voting drops AoE provinces narrower than a Vic3 province (Lorraine's
    pieces, the small HRE states) and can join or separate provinces AoE doesn't
    (Warmia cut off from Elbing). First every AoE province overlapping Vic3 land
    gets a Vic3 province, via swap chains if needed. Then a local search lets
    border provinces switch sides when that lowers mismatched pixels, area error
    and topology violations against AoE's own province adjacency. Near-ties are
    broken toward Vic3's own historical borders: when one AoE state holds most of
    a vanilla Vic3 state, that vanilla state's provinces lean toward it.

    With group ({AoE province: group key}), a Vic3 province only moves between AoE provinces of one group, except
    to give a group with no Vic3 province its first one. That is for a map whose province borders were already
    drawn along the groups' borders (redraw_provinces.py: AoE states under one owner), so the search can't undo
    them.
    """
    votes = defaultdict(dict)
    target = Counter()
    for v, a, n in zip(v3i.tolist(), aid.tolist(), counts.tolist()):
        votes[v][a] = n
        target[a] += n
    owned = Counter(best.values())
    area = Counter()
    for i, a in best.items():
        area[a] += int(pixel_count[i])
    holdings = Counter(group.get(a) for a in best.values()) if group else None

    def same_group(a, b):
        return group is None or group.get(a) == group.get(b)

    def move(i, b, how):
        a = best[i]
        owned[a] -= 1
        area[a] -= int(pixel_count[i])
        owned[b] += 1
        area[b] += int(pixel_count[i])
        if group:
            holdings[group.get(a)] -= 1
            holdings[group.get(b)] += 1
        best[i], best_count[i], method[i] = b, votes[i].get(b, 0), how

    candidates = defaultdict(list)
    for i, d in votes.items():
        for a, n in d.items():
            candidates[a].append((n / pixel_count[i], i))
    for a in candidates:
        candidates[a].sort(reverse=True)

    def cover(start):
        """Augmenting path: start takes a Vic3 province from b, b takes one from c,
        ... until someone with a province to spare gives one up."""
        parent = {start: None}
        queue = [start]
        first = group is not None and not holdings[group.get(start)]   # start's group has no Vic3 province yet
        while queue and len(parent) < COVER_SEARCH_LIMIT:
            a = queue.pop(0)
            for _, i in candidates[a]:
                b = best[i]
                if b in parent or not (same_group(a, b) or (a == start and first)):
                    continue
                parent[b] = (a, i)
                if owned[b] > 1:
                    x = b
                    while parent[x] is not None:
                        prev, j = parent[x]
                        move(j, prev, "coverage")
                        x = prev
                    return True
                queue.append(b)
        return False

    covered = sum(cover(a) for a in sorted((a for a in target if owned[a] == 0), key=lambda a: target[a]))

    # Vanilla state -> the AoE state holding most of it (if it holds at least VANILLA_PARTNER_SHARE).
    vanilla_area, overlap = Counter(), Counter()
    for i, a in best.items():
        v = vanilla_of.get(i)
        if v:
            vanilla_area[v] += int(pixel_count[i])
            overlap[(v, state_of[a])] += int(pixel_count[i])
    partner = {}
    for (v, s_), n in overlap.items():
        if n >= VANILLA_PARTNER_SHARE * vanilla_area[v] and n > overlap.get((v, partner.get(v)), 0):
            partner[v] = s_

    def off_partner(i, a):
        pv = partner.get(vanilla_of.get(i))
        return int(pv is not None and state_of[a] != pv)

    # Local search over border provinces. Cost of an assignment: mismatched pixels
    # + BALANCE_WEIGHT * area error + TOPOLOGY_WEIGHT per topology violation
    # (a border AoE doesn't have, a missing AoE border, an AoE province split in two).
    def key(a, b):
        return (a, b) if a < b else (b, a)

    pair_len = Counter()
    for i, nb in neighbours.items():
        for j, length in nb:
            if i < j and best[i] != best[j]:
                pair_len[key(best[i], best[j])] += length

    def violation(pair, length):
        if length > 0:
            return 0 if pair in aoe_adjacent else 1
        return 1 if pair in aoe_adjacent_strong and owned[pair[0]] and owned[pair[1]] else 0

    def splits(a, i):
        """Would taking i away from a cut a's territory in two?"""
        mine = [j for j, _ in neighbours[i] if best[j] == a]
        if len(mine) <= 1:
            return False
        goal, seen, queue = set(mine[1:]), {mine[0], i}, [mine[0]]
        while queue and goal and len(seen) < 400:
            x = queue.pop()
            for y, _ in neighbours[x]:
                if y not in seen and best[y] == a:
                    seen.add(y)
                    goal.discard(y)
                    queue.append(y)
        return bool(goal) and len(seen) < 400

    def move_cost(i, b):
        a, p = best[i], int(pixel_count[i])
        cost = votes[i].get(a, 0) - votes[i].get(b, 0)
        cost += VANILLA_WEIGHT * p * (off_partner(i, b) - off_partner(i, a))
        cost += BALANCE_WEIGHT * (abs(area[a] - p - target[a]) + abs(area[b] + p - target[b])
                                  - abs(area[a] - target[a]) - abs(area[b] - target[b]))
        change = Counter()
        for j, length in neighbours[i]:
            c = best[j]
            if c != a:
                change[key(a, c)] -= length
            if c != b:
                change[key(b, c)] += length
        topo = sum(violation(pr, pair_len[pr] + d) - violation(pr, pair_len[pr]) for pr, d in change.items() if d)
        topo += int(not any(best[j] == b for j, _ in neighbours[i]))
        return cost + TOPOLOGY_WEIGHT * topo, change

    def total_violations():
        extra = sum(1 for pr, n in pair_len.items() if n > 0 and pr not in aoe_adjacent)
        missing = sum(1 for pr in aoe_adjacent_strong if pair_len[pr] <= 0 and owned[pr[0]] and owned[pr[1]])
        wrong = sum(int(pixel_count[i]) - votes[i].get(best[i], 0) for i in votes)
        return extra, missing, wrong

    log("  topology before local search: %d extra borders, %d missing AoE borders, %d px off" % total_violations())
    switched = 0
    for _ in range(30):
        changed = 0
        for i in sorted(votes):
            if len(votes[i]) < 2 or i not in neighbours:
                continue
            a, p = best[i], int(pixel_count[i])
            if owned[a] <= 1 or all(best[j] == a for j, _ in neighbours[i]):
                continue
            options = [(move_cost(i, b), b) for b, n in votes[i].items()
                       if b != a and n >= MOVE_MIN_SHARE * p and same_group(a, b)]
            if not options:
                continue
            (cost, change), b = min(options, key=lambda o: o[0][0])
            # The split check is the expensive part: only run it for a move we'd otherwise take.
            if cost < -1e-9 and splits(a, i):
                cost += TOPOLOGY_WEIGHT
            if cost < -1e-9:
                for pr, d in change.items():
                    pair_len[pr] += d
                move(i, b, "balanced")
                changed += 1
        switched += changed
        if not changed:
            break
    log("  topology after local search: %d extra borders, %d missing AoE borders, %d px off" % total_violations())
    return covered, switched


# -------------------------------------------------------------- output ----

def state_colors(keys, seed=7):
    rng = np.random.default_rng(seed)
    return {k: tuple(int(c) for c in rng.integers(70, 245, 3)) for k in sorted(keys)}


def write_debug_images(v3, aoe, aoe_at, land3, best, state_of):
    n3 = len(v3.provinces)
    lake = np.array([p in v3.lakes for p in v3.provinces])
    colors = state_colors(aoe.states)
    per_label = np.zeros((n3, 3), np.uint8)
    per_label[:] = (25, 45, 85)
    per_label[lake] = (70, 110, 160)
    state_idx = np.full(n3, -1)
    key_index = {k: i for i, k in enumerate(sorted(aoe.states))}
    for i, a in best.items():
        per_label[i] = colors[state_of[a]]
        state_idx[i] = key_index[state_of[a]]

    lab = v3.labels[::2, ::2]
    img = per_label[lab]
    s = state_idx[lab]
    border = np.zeros(s.shape, bool)
    border[:, :-1] |= s[:, :-1] != s[:, 1:]
    border[:-1, :] |= s[:-1, :] != s[1:, :]
    img[border & (s >= 0)] = (20, 20, 20)
    Image.fromarray(img).save(config.BUILD / "debug_states.png", optimize=True)

    aoe_land = np.zeros(int(aoe.labels.max()) + 1, bool)
    aoe_land[[p for p in aoe.names if p < len(aoe_land) and p not in aoe.sea]] = True
    a_land = aoe_land[aoe_at[::2, ::2].astype(np.int64)]
    v_land = land3[lab]
    coast = np.full(lab.shape + (3,), 255, np.uint8)
    coast[v_land & a_land] = (200, 200, 200)
    coast[v_land & ~a_land] = (220, 40, 40)
    coast[~v_land & a_land] = (40, 90, 230)
    Image.fromarray(coast).save(config.BUILD / "debug_coast.png", optimize=True)
    log(f"land/sea agreement {(v_land == a_land).mean():.1%} (debug images written)")


def main(reuse_warp=False):
    config.BUILD.mkdir(exist_ok=True)
    log("reading maps")
    v3 = read_vic3_map(config.VIC3)
    vanilla = read_vic2_map(config.VIC2 / "map", with_states=False)
    aoe = read_vic2_map(config.AOE / "map")
    log(f"Vic3 {v3.width}x{v3.height}, {len(v3.provinces)} provinces; "
        f"AoE {aoe.width}x{aoe.height}, {len(aoe.names)} provinces in {len(aoe.states)} states")

    points = control_points(vanilla, v3)
    spline, src, dst, keep, resid = fit_spline(points, v3.width, vanilla.width)
    with open(config.BUILD / "warp_points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eu4_provinces", "vic3_x", "vic3_y", "vic2_x", "vic2_y", "uncertainty", "used", "cv_residual_px"])
        for p, s, d, k, r in zip(points, src, dst, keep, resid):
            w.writerow([" ".join(map(str, p[5])), f"{s[0]:.1f}", f"{s[1]:.1f}", f"{d[0]:.1f}", f"{d[1]:.1f}",
                        f"{p[4]:.2f}", int(k), "" if np.isnan(r) else f"{r:.2f}"])

    land3 = land_provinces(v3)
    if reuse_warp and (config.BUILD / "warp_grid.npz").exists():
        log("reusing the warp from the previous run")
        warp = load_warp(v3, aoe)
    else:
        log("refining warp by matching coastlines")
        base = spline_grid(spline, v3.width, v3.height, aoe.width)
        g2, g3 = region_labels(vanilla, v3, land3)
        warp = refine_with_regions(base, region_cells(v3, g3, REFINE_STEP), g2[vanilla.labels],
                                   vanilla.width, vanilla.height)
    np.savez_compressed(config.BUILD / "warp_grid.npz", x=warp.x, y=warp.y, step=warp.step, origin=warp.origin,
                        w3=v3.width, w2=aoe.width)

    log("warping AoE map onto Vic3 pixels")
    aoe_at = aoe_under_vic3(warp, aoe, v3.width, v3.height)
    aoe_at = keep_to_landmass(aoe_at, Landmasses(v3, land3, aoe, aoe_at))
    np.save(config.BUILD / "aoe_at_vic3.npy", aoe_at)

    result = assign(v3, aoe, aoe_at, warp, land3)
    write_province_map(v3, aoe, *result)
    write_debug_images(v3, aoe, aoe_at, land3, result[0], result[4])


def land_provinces(v3):
    """Per Vic3 province: listed in a land state region."""
    index = {p: i for i, p in enumerate(v3.provinces)}
    land3 = np.zeros(len(v3.provinces), bool)
    for block in v3.land_states.values():
        land3[[index[norm_hex(p)] for p in block.get("provinces").values()]] = True
    return land3


def load_warp(v3, aoe):
    g = np.load(config.BUILD / "warp_grid.npz")
    return GridWarp(g["x"], g["y"], float(g["step"]), float(g["origin"]), v3.width, aoe.width)


def write_province_map(v3, aoe, best, best_count, pixel_count, method, state_of, centroids):
    """build/province_map.csv from assign()'s result."""
    cx, cy, wx, wy = centroids
    vanilla_state = v3.province_state()
    with open(config.BUILD / "province_map.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["vic3_province", "vanilla_vic3_state", "aoe_province", "aoe_province_name", "aoe_state",
                    "pixels", "share", "method", "vic3_x", "vic3_y", "vic2_x", "vic2_y"])
        for i in sorted(best, key=lambda i: v3.provinces[i]):
            a = best[i]
            w.writerow([v3.provinces[i], vanilla_state.get(v3.provinces[i], ""), a, aoe.names.get(a, ""),
                        state_of[a], int(pixel_count[i]), f"{best_count[i] / max(pixel_count[i], 1):.3f}", method[i],
                        f"{cx[i]:.1f}", f"{cy[i]:.1f}", f"{wx[i]:.1f}", f"{wy[i]:.1f}"])

    used_states = {state_of[a] for a in best.values()}
    lost = sorted(s for s in set(aoe.states) - used_states if aoe.states[s])
    log(f"AoE states with Vic3 land: {len(used_states)} of {sum(1 for p in aoe.states.values() if p)} non-empty")
    if lost:
        log(f"non-empty AoE states with no Vic3 province: {' '.join(lost)}")


if __name__ == "__main__":
    import sys
    main(reuse_warp="--reuse-warp" in sys.argv)
