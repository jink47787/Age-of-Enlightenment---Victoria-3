"""Redraw Vic3 province borders so state and country borders follow the AoE map.

Runs after build_province_map.py and uses its warp (build/aoe_at_vic3.npy: the AoE province under every Vic3
pixel), so the two projections are compared through the same fitted warp, never pixel for pixel. Coastlines never
change: only land pixels of land provinces are repainted. Sea, lakes and the provinces adjacencies.csv uses are
left as they are (but for step 6), and land never moves between passable and impassable provinces.

1. Target: the AoE unit under each Vic3 land pixel, a unit being an AoE state's land under one 1700 owner (unowned
   land counts as one owner). Those are the borders Vic3 shows: state regions and countries. AoE province borders
   inside a unit have no counterpart in Vic3 (pops and buildings are per state), so they are not drawn. Vic3 land
   under AoE sea or wasteland takes the unit of the nearest AoE land, on its own landmass and across the map's
   x seam (build_province_map.fill_nearest, which the assignment uses too).
2. Smoothing: Gaussian-blurred unit indicators, argmax, so province lines don't copy the warp's pixel noise.
3. Snapping: in each province that straddles a border, land not under its main unit is opened with a disk
   (judged across province lines, so a strip of one unit through several provinces counts as a whole). Pieces
   under MIN_PIECE px go back to the main unit, and so do strips narrower than 2 * OPEN_R + 1 px when OPEN_R > 0:
   moving them would only make vanilla lines ragged. The rest moves, so Vic3 lines end up on AoE's.
4. Leftover pieces of one unit that touch form a region. Existing borders move first: the region joins a
   neighbouring province of its unit, or else each source province's piece of it joins one, as long as no
   province grows past GROW_MAX times its size. Only what can't join becomes a new province (MIN_NEW px or more,
   cut along the old province lines if it is large). Provinces named in vanilla scripts (events, releasables...)
   are never split, since the mod doesn't replace those files: their pieces can only join a neighbour.
5. Every piece of a unit of MIN_PIECE_KEEP px or more (an exclave, a trading post) gets a province of its own.
6. A unit still without a province of its own (a trading post the smoothing cut below MIN_PIECE_KEEP px, or one
   inside a province scripts name, like Bombay or Madras) gets one carved where the warp puts it, as large as its
   warped area and at least MIN_PIECE_KEEP px, and on the coast if it is on AoE's and Vic3's is within COAST_R px
   (the warp puts these few-px posts a pixel or two inland). This is the one place provinces scripts name are
   cut, and provinces only sea crossings use, more than CROSSING_R px from the crossing's ends (Nagapattinam's
   coast is the Palk Strait crossing's): they keep their id and all but the post's few pixels. The assignment
   then gives that province to the unit.

New provinces go into their parent's vanilla state region (mapdata.read_vic3_map with config.REDRAW), with its
terrain, and in its impassable / prime land lists when the parent is listed there.

Outputs (build/redraw/):
  provinces.png        the redrawn province image (vanilla colours plus one new colour per new province)
  new_provinces.csv    new province -> parent, terrain, impassable, prime land, AoE state, owner, pixels, source
                       provinces, and the AoE province a step-6 province is reserved for
  reason.npy           why each pixel moved: 0 it didn't, 1 snapped back, 2 kept (too small, a sliver or a
                       province scripts name), 3 joined a neighbour, 4 new province, 5 carved for a unit piece,
                       6 carved for a unit left without a province
and build/province_map.csv again, assigned on the redrawn map. write_mod_files() puts provinces.png and
province_terrains.txt into mod/map_data.
"""
import csv
import heapq
import re
import shutil
import time
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import config
import pdx
from aoehistory import read_provinces
from build_province_map import (Landmasses, assign, fill_nearest, land_provinces, load_warp, norm_hex,
                                write_debug_images, write_province_map)
from mapdata import read_new_provinces, read_vic2_map, read_vic3_map

SIGMA = 1.5          # px, smoothing of the warped unit field
OPEN_R = 0           # px, opening radius for a province's minor pieces (0: no opening, every strip moves)
MIN_PIECE = 10       # px, smaller minor pieces go back to the province's main unit
SNAP_MAX = 0.15      # ...but a band over this share of its province is a real border and moves
GROW_MAX = 2.0       # a leftover piece joins a same-unit neighbour unless that grows past this multiple of its size
SPLIT_FRAC = 1.75    # a new region bigger than this multiple of its source provinces' median size is cut along them
MIN_NEW = 60         # px, smallest new province (vanilla: 5% of European provinces are under ~80 px)...
MIN_PIECE_KEEP = 20  # px, ...except the only land of a unit piece (a state, exclave or trading post)
PLACE_R = 32         # px, how far from its warped pixels a unit without a province may get its land
COAST_R = 6          # px, ...and how far a unit on AoE's coast may move to start on Vic3's (the warp's coastal error)
CROSSING_R = 12      # px around a sea crossing's end (adjacencies.csv) that its province always keeps
SLIVER_R = 2         # px, radius of DISK
SLIVER = 0.5         # a new or cut-down province keeps at least this share under an opening with DISK (vanilla:
                     # 98% of provinces keep 0.74 or more); thinner ones are bands within the warp's error
UNOWNED = "~"        # owner of native land
COLOUR_STEP = 0x0B0705   # new colours: the parent's colour plus steps of this until unused

FOUR = ndimage.generate_binary_structure(2, 1)
DISK = (np.add.outer(np.arange(-SLIVER_R, SLIVER_R + 1) ** 2, np.arange(-SLIVER_R, SLIVER_R + 1) ** 2)
        <= SLIVER_R ** 2 + 1)


def log(msg, _t0=time.time()):
    print(f"[{time.time() - _t0:6.1f}s] {msg}", flush=True)


class VanillaMap:
    """The vanilla Vic3 map and the warped AoE map, with per-province arrays indexed like v3.provinces."""

    def __init__(self):
        self.v3 = v3 = read_vic3_map(config.VIC3)
        self.aoe = read_vic2_map(config.AOE / "map")
        self.aoe_at = np.load(config.BUILD / "aoe_at_vic3.npy")
        self.index = {p: i for i, p in enumerate(v3.provinces)}
        self.n3 = len(v3.provinces)
        self.pixels = np.bincount(v3.labels.ravel(), minlength=self.n3)
        self.land = land_provinces(v3)
        self.sea = np.zeros(self.n3, bool)         # in a sea state region (lakes aren't)
        self.sea[[self.index[norm_hex(p)] for b in v3.sea_states.values() for p in b.get("provinces").values()]] = True
        sea2 = np.isin(self.aoe.labels, list(self.aoe.sea))
        self.aoe_coast = np.zeros(int(self.aoe.labels.max()) + 1, bool)   # per AoE province: land touching its sea
        self.aoe_coast[self.aoe.labels[~sea2 & ndimage.binary_dilation(sea2, structure=FOUR)]] = True
        self.impassable = self.listed("impassable")
        self.prime_land = self.listed("prime_land")
        self.frozen, self.post_ok, self.crossing_ends, self.pieces = self.frozen_provinces()
        self.scripted = np.zeros(self.n3, bool)
        self.scripted[[self.index[p] for p in script_references() if p in self.index]] = True
        self.scripted &= self.land

    def listed(self, key):
        out = np.zeros(self.n3, bool)
        for block in self.v3.land_states.values():
            ids = block.get(key)
            if isinstance(ids, pdx.Block):
                out[[self.index[norm_hex(p)] for p in ids.values()]] = True
        return out

    def frozen_provinces(self):
        """Land provinces never repainted (the ones adjacencies.csv uses); which of those a trading post may still
        take land from (used only by sea crossings with set ends), and the pixels it may not take (within
        CROSSING_R of an end); and vanilla's piece count per province."""
        labels = self.v3.labels
        frozen = np.zeros(self.n3, bool)
        post_ok = np.ones(self.n3, bool)
        ends = np.zeros(labels.shape, bool)
        for row in pdx.read_text(config.VIC3 / "map_data" / "adjacencies.csv").splitlines()[1:]:
            cells = [c.strip() for c in row.split(";")]
            xy = [int(c) if c.lstrip("-").isdigit() else -1 for c in (cells + [""] * 8)[4:8]]
            for p, (x, y) in zip(cells[:2], (xy[:2], xy[2:])):
                if p.lower().startswith("x") and norm_hex(p) in self.index:
                    i = self.index[norm_hex(p)]
                    frozen[i] = True
                    if cells[2] == "sea" and x >= 0 and y >= 0:
                        ends[self.v3.height - 1 - y, x] = True     # y counts from the bottom
                    else:
                        post_ok[i] = False
        ends = ndimage.distance_transform_edt(~ends) <= CROSSING_R
        pieces = np.zeros(self.n3, np.int32)
        for i, sl in enumerate(ndimage.find_objects(labels + 1)):
            if sl is not None and self.land[i]:
                pieces[i] = ndimage.label(labels[sl] == i, structure=FOUR)[1]
        return frozen & self.land, post_ok, ends, pieces


def script_references():
    """Provinces named in vanilla scripts outside history (events, releasables, strategic regions...)."""
    refs = set()
    for sub in ("common", "events"):
        for path in (config.VIC3 / sub).rglob("*.txt"):
            if "history" not in path.parts:
                refs |= {norm_hex(m) for m in re.findall(r"\bx[0-9A-Fa-f]{6}\b", pdx.read_text(path))}
    return refs


def aoe_units(aoe):
    """{AoE province in a state: (state, 1700 owner tag or UNOWNED)}."""
    provs = read_provinces()
    return {a: (s, provs[a].owner if a in provs and provs[a].owner else UNOWNED) for a, s in aoe.state_of().items()}


# ---------------------------------------------------------------- target ----

def unit_codes(vm, unit):
    """(names, lut): unit code per AoE province, names[code] = (state, owner); -1 = sea, wasteland or none."""
    names = sorted(set(unit.values()))
    idx = {n: k for k, n in enumerate(names)}
    lut = np.full(max(int(vm.aoe.labels.max()), int(vm.aoe_at.max())) + 1, -1, np.int16)
    for a, u in unit.items():
        lut[a] = idx[u]
    return names, lut


def smooth_field(t, land, sigma, tile=512):
    """Argmax of Gaussian-blurred unit indicators over land; sea keeps t."""
    h, w = t.shape
    m = int(4 * sigma) + 1
    out = t.copy()
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            a, b, c, d = max(0, y0 - m), min(h, y0 + tile + m), max(0, x0 - m), min(w, x0 + tile + m)
            sub, l = t[a:b, c:d], land[a:b, c:d]
            vals = np.unique(sub[l])
            vals = vals[vals >= 0]
            if len(vals) < 2:
                continue
            best = np.full(sub.shape, -1.0, np.float32)
            arg = sub.copy()
            for v in vals:
                s = ndimage.gaussian_filter(((sub == v) & l).astype(np.float32), sigma)
                up = s > best
                best[up], arg[up] = s[up], v
            core = (slice(y0 - a, y0 - a + min(tile, h - y0)), slice(x0 - c, x0 - c + min(tile, w - x0)))
            out[y0:y0 + tile, x0:x0 + tile] = np.where(l, arg, sub)[core]
    return out


# ------------------------------------------------------------ the redraw ----

def components_same_value(mask, value):
    """4-connected components of the pixels in mask, joining only neighbours with equal value.
    Returns (n, comp id per masked pixel in np.nonzero order, ys, xs)."""
    ys, xs = np.nonzero(mask)
    n = len(ys)
    pos = np.full(mask.shape, -1, np.int64)
    pos[ys, xs] = np.arange(n)
    rows, cols = [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = pos[:mask.shape[0] - dy, :mask.shape[1] - dx]
        b = pos[dy:, dx:]
        va = value[:mask.shape[0] - dy, :mask.shape[1] - dx]
        vb = value[dy:, dx:]
        ok = (a >= 0) & (b >= 0) & (va == vb)
        rows.append(a[ok])
        cols.append(b[ok])
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(n, n))
    k, comp = connected_components(g, directed=False)
    return k, comp, ys, xs


def open_labels(field, mask, r, band=512):
    """Morphological opening of every unit's part of `mask` with a disk of radius r, all units at once:
    a pixel survives if some disk inside mask, all of one unit, covers it."""
    h, w = field.shape
    offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= r * r + 1]
    out = np.zeros((h, w), bool)
    for y0 in range(0, h, band):
        a, b = max(0, y0 - 2 * r), min(h, y0 + band + 2 * r)
        f = np.pad(np.where(mask[a:b], field[a:b].astype(np.int32), -1), 2 * r, constant_values=-1)
        hh, ww = b - a + 2 * r, w + 2 * r          # erosion computed on a ring r px wider than the band

        def shift(arr, dy, dx, pad):
            return arr[pad + dy:pad + dy + hh, pad + dx:pad + dx + ww]
        core = shift(f, 0, 0, r) >= 0
        for dy, dx in offs:
            core &= shift(f, dy, dx, r) == shift(f, 0, 0, r)
        cval = np.where(core, shift(f, 0, 0, r), -2)
        cval = np.pad(cval, r, constant_values=-2)
        centre = f[2 * r:2 * r + b - a, 2 * r:2 * r + w]
        hit = np.zeros((b - a, w), bool)
        for dy, dx in offs:
            hit |= cval[2 * r + dy:2 * r + dy + b - a, 2 * r + dx:2 * r + dx + w] == centre
        hit &= centre >= 0
        out[y0:min(h, y0 + band)] = hit[y0 - a:y0 - a + min(band, h - y0)]
    return out


def sliver(mask):
    """Under SLIVER of the mask survives an opening with DISK: mostly parts under 2 * SLIVER_R + 1 px wide."""
    m = np.pad(mask, SLIVER_R + 1)
    return ndimage.binary_opening(m, structure=DISK).sum() < SLIVER * m.sum()


def restore_slivered(vm, new_labels, new_rows, affected, reason):
    """A province that lost land and ended up a sliver (and was not one in vanilla) takes back what it gave its
    neighbours, and what it gave new provinces as long as those keep MIN_NEW px."""
    labels = vm.v3.labels
    slices = ndimage.find_objects(labels + 1)
    area = np.bincount(new_labels.ravel(), minlength=vm.n3 + len(new_rows))
    restored = 0
    for i in affected.tolist():
        sl = slices[i]
        m0 = labels[sl] == i
        gone = m0 & (new_labels[sl] != i)
        if not gone.any() or not sliver(new_labels[sl] == i) or sliver(m0):
            continue
        sub = new_labels[sl]
        back = gone & (sub < vm.n3)
        for q, c in zip(*np.unique(sub[gone & (sub >= vm.n3)], return_counts=True)):
            if area[q] - c >= MIN_NEW:
                back |= gone & (sub == q)
        for q, c in zip(*np.unique(sub[back], return_counts=True)):
            area[q] -= c
        sub[back] = i
        reason[sl][back] = 2
        restored += 1
    return restored


def keep_unit_pieces(vm, field, land_px, new_labels, new_rows, main_of, splittable, cls, reason, names):
    """Every piece of a unit in the target (a state, an exclave like Avignon, a trading post) gets a province
    of its own unit: a piece of MIN_PIECE_KEEP px or more that no province of that unit covers becomes a new
    province (its land in splittable provinces of one class, never cutting a province below MIN_NEW px)."""
    n = vm.n3 + len(new_rows)
    main = np.r_[main_of, [row[2] for row in new_rows]].astype(np.int64)
    ok = np.r_[splittable, np.ones(len(new_rows), bool)]
    cl = np.r_[cls, [row[3] for row in new_rows]].astype(np.int64)
    area = np.bincount(new_labels.ravel(), minlength=n)
    fl = np.where(land_px, field.astype(np.int32), -1)
    boxes = ndimage.find_objects(fl + 1)
    carved = 0
    for v in range(min(len(names), len(boxes))):
        if boxes[v] is None:
            continue
        sl = tuple(slice(max(0, b.start - 1), b.stop + 1) for b in boxes[v])
        cc, k = ndimage.label(fl[sl] == v, structure=FOUR)
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        sub = new_labels[sl]
        for j in np.flatnonzero(sizes >= MIN_PIECE_KEEP).tolist():
            m = cc == j
            qs, qc = np.unique(sub[m], return_counts=True)
            if any(main[q] == v and c >= min(0.5 * area[q], 0.5 * sizes[j]) for q, c in zip(qs.tolist(), qc.tolist())):
                continue
            take = m & ok[sub]
            if not take.any():
                continue
            c0 = np.bincount(cl[sub[take]]).argmax()
            take &= cl[sub] == c0
            for q, c in zip(*np.unique(sub[take], return_counts=True)):
                if area[q] - c < MIN_NEW:            # never cut a province below MIN_NEW (or away entirely)
                    take &= sub != q
            cc2, k2 = ndimage.label(take, structure=FOUR)
            if not k2:
                continue
            s2 = np.bincount(cc2.ravel())
            s2[0] = 0
            if s2.max() < MIN_PIECE_KEEP:
                continue
            part = cc2 == s2.argmax()
            idx = vm.n3 + len(new_rows)
            srcs = Counter(sub[part].tolist())
            area[list(srcs)] -= np.array(list(srcs.values()))
            sub[part] = idx
            reason[sl][part] = 5
            # sources as vanilla provinces (a carve out of a new province counts that one's sources)
            vsrc = Counter()
            for q, c in srcs.items():
                if q < vm.n3:
                    vsrc[q] += c
                else:
                    vsrc[new_rows[q - vm.n3][1].most_common(1)[0][0]] += c
            new_rows.append((idx, vsrc, int(v), int(c0)))
            main = np.r_[main, v]
            ok = np.r_[ok, True]
            cl = np.r_[cl, c0]
            area = np.r_[area, int(part.sum())]
            carved += 1
    return carved


def place_lost_units(vm, lut, land_px, new_labels, new_rows, main_of, cls, reason):
    """A unit no province belongs to after the steps above gets one where the warp puts it: the land nearest its
    warped pixels, grown nearest-first to as many px as those (at least MIN_PIECE_KEEP), from provinces of one
    class that keep MIN_NEW px. A unit on AoE's coast starts on Vic3's coast when that is within COAST_R px, so a
    trading post keeps its port. Unlike the other steps it also cuts provinces scripts name (their id stays on the
    rest) and frozen ones that only sea crossings use, away from the crossings' ends. A rest of a source province that the cut leaves smaller than the new province
    and touching it (the end of an island, say) goes with it. Returns {new row index: the unit's AoE province with
    the most warped pixels}, for the assignment."""
    h, w = new_labels.shape
    n = vm.n3 + len(new_rows)
    main = np.r_[main_of, [row[2] for row in new_rows]].astype(np.int64)
    area = np.bincount(new_labels.ravel(), minlength=n).astype(np.int64)
    is_land = np.r_[vm.land, np.ones(len(new_rows), bool)]
    held = np.zeros(int(lut.max()) + 1, bool)
    held[main[(area > 0) & is_land & (main >= 0)]] = True
    t = lut[vm.aoe_at]
    a_count = np.bincount(vm.aoe_at.ravel().astype(np.int64), minlength=len(lut))
    boxes = ndimage.find_objects(t.astype(np.int32) + 1)
    ok = np.r_[vm.land & (~vm.frozen | vm.post_ok), np.ones(len(new_rows), bool)]
    cl = np.r_[cls, [row[3] for row in new_rows]].astype(np.int64)
    codes = lut[np.flatnonzero(vm.aoe_coast)]
    coastal = np.zeros(len(lut), bool)                   # per unit code: has land on AoE's coast
    coastal[codes[codes >= 0]] = True
    shore = land_px & ndimage.binary_dilation(vm.sea[vm.v3.labels], structure=FOUR)
    posts, skipped = {}, 0
    for v in np.flatnonzero(~held[:len(boxes)]).tolist():
        if boxes[v] is None:
            continue
        ys0, xs0 = boxes[v]
        y0, y1 = max(0, ys0.start - PLACE_R), min(h, ys0.stop + PLACE_R)
        x0, x1 = max(0, xs0.start - PLACE_R), min(w, xs0.stop + PLACE_R)
        sub = new_labels[y0:y1, x0:x1]
        seed = t[y0:y1, x0:x1] == v
        spare = area - MIN_NEW
        free = (land_px[y0:y1, x0:x1] & ok[sub] & (spare[sub] > 0) & ~np.isin(sub, list(posts))
                & ~vm.crossing_ends[y0:y1, x0:x1])
        d = ndimage.distance_transform_edt(~seed)
        start_at = free & (d <= PLACE_R)
        on_coast = free & shore[y0:y1, x0:x1] & (d <= COAST_R)
        if coastal[v] and on_coast.any():
            start_at = on_coast
        fy, fx = np.nonzero(start_at)
        if not len(fy):
            skipped += 1
            continue
        j = int(np.argmin(d[fy, fx]))
        start = (int(fy[j]), int(fx[j]))
        c0 = int(cl[sub[start]])
        free &= cl[sub] == c0
        want = max(MIN_PIECE_KEEP, int(seed.sum()))
        heap, seen, take, given = [(d[start], start)], {start}, [], Counter()
        while heap and len(take) < want:
            _, (y, x) = heapq.heappop(heap)
            q = int(sub[y, x])
            if given[q] >= spare[q]:
                continue
            given[q] += 1
            take.append((y, x))
            for ny, nx in ((y, x + 1), (y, x - 1), (y + 1, x), (y - 1, x)):
                if 0 <= ny < y1 - y0 and 0 <= nx < x1 - x0 and free[ny, nx] and (ny, nx) not in seen:
                    seen.add((ny, nx))
                    heapq.heappush(heap, (d[ny, nx], (ny, nx)))
        if len(take) < min(want, MIN_PIECE_KEEP):
            skipped += 1
            continue
        part = np.zeros(sub.shape, bool)
        part[tuple(np.array(take).T)] = True
        near = ndimage.binary_dilation(part, structure=FOUR)
        inner = np.zeros(sub.shape, bool)
        inner[1:-1, 1:-1] = True
        for q in list(given):
            cc, k = ndimage.label((sub == q) & ~part, structure=FOUR)
            sizes = np.bincount(cc.ravel())
            sizes[0] = 0
            for p_ in range(1, k + 1):
                piece = cc == p_
                if (p_ != sizes.argmax() and sizes[p_] < part.sum() and (piece & near).any()
                        and not (piece & ~inner).any()):
                    part |= piece
                    given[q] += int(sizes[p_])
        idx = vm.n3 + len(new_rows)
        srcs = Counter(sub[part].tolist())
        area[list(srcs)] -= np.array(list(srcs.values()))
        sub[part] = idx
        reason[y0:y1, x0:x1][part] = 6
        vsrc = Counter()
        for q, c in srcs.items():
            vsrc[q if q < vm.n3 else new_rows[q - vm.n3][1].most_common(1)[0][0]] += c
        new_rows.append((idx, vsrc, int(v), c0))
        main = np.r_[main, v]
        ok = np.r_[ok, True]
        cl = np.r_[cl, c0]
        area = np.r_[area, int(part.sum())]
        units = np.flatnonzero(lut == v)
        posts[idx] = int(units[np.argmax(a_count[units])])
    return posts, skipped


def redraw(vm, unit):
    """Returns (new labels, new rows, unit names, per-province class incl. new ones, receiving-ok incl. new ones,
    reason per pixel). New rows are (index, Counter of vanilla source provinces, unit code, class)."""
    labels = vm.v3.labels
    h, w = labels.shape
    names, lut = unit_codes(vm, unit)
    land_px = vm.land[labels]
    editable = vm.land & ~vm.frozen
    locked = vm.scripted.astype(np.int64)            # never split: its leftover pieces can only join a neighbour
    cls = vm.impassable.astype(np.int64)             # 1 = impassable; land never moves between the two classes
    log(f"{int(vm.land.sum())} land provinces ({int((vm.land & vm.impassable).sum())} impassable), "
        f"{int(vm.frozen.sum())} frozen (adjacencies.csv), {int(vm.scripted.sum())} named in vanilla scripts")

    known = lut[vm.aoe_at] >= 0
    t = lut[fill_nearest(vm.aoe_at, known, land_px & ~known, Landmasses(vm.v3, vm.land, vm.aoe, vm.aoe_at))]
    t[land_px & (t < 0)] = 0
    field = smooth_field(t, land_px, SIGMA)
    log("target field smoothed")

    nl = len(names)
    key = labels[land_px].astype(np.int64) * nl + field[land_px]
    uk, cnt = np.unique(key, return_counts=True)
    prov_of, code_of = uk // nl, uk % nl
    main_of = np.full(vm.n3, -1, np.int64)       # majority unit code; the main piece's for redrawn provinces
    order = np.lexsort((-cnt, prov_of))
    first = np.r_[True, prov_of[order][1:] != prov_of[order][:-1]]
    main_of[prov_of[order][first]] = code_of[order][first]
    affected = np.flatnonzero((np.bincount(prov_of, minlength=vm.n3) > 1) & editable)
    log(f"{len(affected)} provinces straddle a state or country border")

    # main piece per province; snapping of the land not under its province's main unit
    slices = ndimage.find_objects(labels + 1)
    main_seed = {}
    for i in affected.tolist():
        sl = slices[i]
        m = labels[sl] == i
        f = field[sl]
        best = None
        for v in np.unique(f[m]).tolist():
            cc, n = ndimage.label(m & (f == v), structure=FOUR)
            sizes = np.bincount(cc.ravel())[1:]
            j = int(sizes.argmax())
            if best is None or sizes[j] > best[0]:
                best = (int(sizes[j]), v, np.argwhere(cc == j + 1)[0])
        _, main_of[i], main_seed[i] = best
    minor = land_px & editable[labels] & (field != main_of[labels])
    keep = open_labels(field, minor, OPEN_R) if OPEN_R else minor.copy()
    k, comp, ys, xs = components_same_value(keep, field)
    small = np.bincount(comp, minlength=k) < MIN_PIECE
    keep[ys[small[comp]], xs[small[comp]]] = False
    back = minor & ~keep
    # a band that is a large share of its province is not a sliver: the border moves after all
    key = labels[back].astype(np.int64) * nl + field[back]
    uk, cnt = np.unique(key, return_counts=True)
    big = uk[cnt > SNAP_MAX * vm.pixels[uk // nl]]
    back &= ~np.isin(labels.astype(np.int64) * nl + field, big)
    f_all = np.where(back, main_of[labels], field)
    reason = np.where(back, 1, 0).astype(np.uint8)
    log(f"snapping returned {int(back.sum())} px to their province's main unit")

    orphan = np.full((h, w), -1, np.int32)        # unit code of each leftover pixel
    stayed = 0
    for i in affected.tolist():
        sl = slices[i]
        m = labels[sl] == i
        f = f_all[sl]
        mv = main_of[i]
        vanilla_piece, _ = ndimage.label(m, structure=FOUR)
        cc, _ = ndimage.label(m & (f == mv), structure=FOUR)
        sy, sx = main_seed[i]
        own = m & (cc == cc[sy, sx])
        # main-unit land in another vanilla piece of the province (an island, say) stays with it
        own |= m & (f == mv) & (vanilla_piece != vanilla_piece[sy, sx])
        left = m & ~own
        # a province is never cut below MIN_NEW px or down to a sliver: then it stays whole, and the unit smoothing
        # gives it to one side
        if own.sum() < MIN_NEW or (sliver(own) and not sliver(m)):
            reason[sl][left] = 2
            stayed += 1
            continue
        orphan[sl][left] = f[left]
    log(f"{stayed} provinces too small to cut stay whole")

    # leftover regions: same unit, same class, touching; pieces of provinces scripts name stay apart
    is_orphan = orphan >= 0
    k, comp, ys, xs = components_same_value(is_orphan, (orphan.astype(np.int64) * 2 + cls[labels]) * 2
                                            + locked[labels])
    src = labels[ys, xs]
    size = np.bincount(comp, minlength=k)
    region_code = np.zeros(k, np.int64)
    region_code[comp] = orphan[ys, xs]
    region_cls = np.zeros(k, np.int64)
    region_cls[comp] = cls[src]
    region_locked = np.zeros(k, bool)
    region_locked[comp] = locked[src].astype(bool)
    area0 = vm.pixels.astype(np.int64)
    src_area = np.zeros(k)
    per = defaultdict(set)
    rs = np.unique(comp * vm.n3 + src)
    for r_, s_ in zip((rs // vm.n3).tolist(), (rs % vm.n3).tolist()):
        per[r_].add(s_)
    for r_, ss in per.items():
        src_area[r_] = np.median([area0[s] for s in ss])

    new_labels = labels.copy()
    grown = np.bincount(labels[~is_orphan], minlength=vm.n3).astype(np.int64)  # size without its leftovers, so far
    new_rows = []

    def join_target(part, code, c):
        """The vanilla province a piece (indices into ys, xs) joins: of unit `code` and class c, touching the piece
        most, and not growing past GROW_MAX times its size (a piece under MIN_NEW px may always join). None if none."""
        hits = []
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = ys[part] + dy, (xs[part] + dx) % w
            ok = (ny >= 0) & (ny < h)
            ny, nx = ny[ok], nx[ok]
            q = new_labels[ny, nx]
            ok = q < vm.n3
            ny, nx, q = ny[ok], nx[ok], q[ok]
            # the neighbour's own land: never a leftover still waiting to move, but land that already joined it is
            good = ((~is_orphan[ny, nx] | (reason[ny, nx] == 3)) & editable[q] & (main_of[q] == code) & (cls[q] == c)
                    & land_px[ny, nx])
            hits.append(q[good])
        qs, cnt = np.unique(np.concatenate(hits), return_counts=True)
        for j in np.argsort(-cnt, kind="stable").tolist():
            if grown[qs[j]] + len(part) <= GROW_MAX * area0[qs[j]] or len(part) < MIN_NEW:
                return int(qs[j])
        return None

    def join(part, q):
        new_labels[ys[part], xs[part]] = q
        reason[ys[part], xs[part]] = 3
        grown[q] += len(part)

    def stay(part):
        reason[ys[part], xs[part]] = 2
        np.add.at(grown, src[part], 1)

    pix_of = np.split(np.argsort(comp, kind="stable"), np.cumsum(size)[:-1])
    joined = joined_pieces = slivers = tiny = kept_locked = kept_locked_px = 0
    for r_ in np.argsort(-size).tolist():
        px = pix_of[r_]
        code, c = int(region_code[r_]), int(region_cls[r_])
        # existing borders move first: the region joins a neighbouring province of its unit, or else each source
        # province's piece of it joins one on its own
        q = join_target(px, code, c)
        if q is not None:
            join(px, q)
            joined += 1
            continue
        rest = [px]
        if len(per[r_]) > 1:
            rest = []
            for part in sorted(split_by_source(px, ys, xs, src, min_size=0), key=len, reverse=True):
                q = join_target(part, code, c)
                if q is None:
                    rest.append(part)
                else:
                    join(part, q)
                    joined_pieces += 1
        # only what can't join becomes a new province
        for piece in (connected_parts(np.concatenate(rest), ys, xs) if rest else []):
            if region_locked[r_]:
                stay(piece)
                kept_locked += 1
                kept_locked_px += len(piece)
                continue
            if len(piece) < MIN_NEW:
                stay(piece)
                tiny += 1
                continue
            parts = [piece]
            if len(piece) > SPLIT_FRAC * src_area[r_] and len(np.unique(src[piece])) > 1:
                parts = split_by_source(piece, ys, xs, src)
            for part in parts:
                py, pxx = ys[part], xs[part]
                box = np.zeros((py.max() - py.min() + 1, pxx.max() - pxx.min() + 1), bool)
                box[py - py.min(), pxx - pxx.min()] = True
                if sliver(box):                      # a band along a border moved within the warp's error: no province
                    stay(part)
                    slivers += 1
                    continue
                idx = vm.n3 + len(new_rows)
                new_labels[py, pxx] = idx
                reason[py, pxx] = 4
                new_rows.append((idx, Counter(src[part].tolist()), code, c))
    log(f"{k} leftover regions: {joined} joined a neighbouring province of their unit, {joined_pieces} more pieces "
        f"joined one separately; {len(new_rows)} new provinces; {tiny} pieces too small for a province of their own "
        f"and {slivers} slivers stay where they were")
    log(f"{kept_locked} pieces ({kept_locked_px} px) of provinces named in vanilla scripts stay unsplit")
    log(f"{restore_slivered(vm, new_labels, new_rows, affected, reason)} provinces left as slivers took their "
        f"land back")
    carved = keep_unit_pieces(vm, field, land_px, new_labels, new_rows, main_of, editable & ~vm.scripted, cls,
                              reason, names)
    log(f"{carved} unit pieces left without a province got one carved from their own land")
    posts, skipped = place_lost_units(vm, lut, land_px, new_labels, new_rows, main_of, cls, reason)
    log(f"{len(posts)} units with no province of their own got one where the warp puts them"
        + (f" ({skipped} with no land to take nearby)" if skipped else ""))
    cls_all = np.r_[cls, [row[3] for row in new_rows]].astype(np.int64)
    ok_all = np.r_[editable, np.ones(len(new_rows), bool)]
    return new_labels, new_rows, names, cls_all, ok_all, reason, posts


def connected_parts(px, ys, xs):
    """4-connected components of a set of region pixels (indices into ys, xs)."""
    sy, sx = ys[px], xs[px]
    box = np.zeros((sy.max() - sy.min() + 1, sx.max() - sx.min() + 1), bool)
    box[sy - sy.min(), sx - sx.min()] = True
    cc, _ = ndimage.label(box, structure=FOUR)
    which = cc[sy - sy.min(), sx - sx.min()]
    return [px[which == p] for p in np.unique(which).tolist()]


def split_by_source(px, ys, xs, src, min_size=MIN_NEW):
    """Cut a region along the old province lines; parts under min_size px join the neighbouring part they touch most."""
    sy, sx, ss = ys[px], xs[px], src[px]
    y0, x0 = sy.min(), sx.min()
    box = np.full((sy.max() - y0 + 3, sx.max() - x0 + 3), -1, np.int64)
    box[sy - y0 + 1, sx - x0 + 1] = ss
    k, comp, by, bx = components_same_value(box >= 0, box)
    part = np.full(box.shape, -1, np.int64)
    part[by, bx] = comp
    while True:
        sizes = np.bincount(part[part >= 0], minlength=k)
        live = np.flatnonzero(sizes)
        small = [p for p in live.tolist() if sizes[p] < min_size]
        if not small or len(live) == 1:
            break
        p = min(small, key=lambda q: sizes[q])
        m = part == p
        ring = ndimage.binary_dilation(m, structure=FOUR) & ~m & (part >= 0)
        nb = Counter(part[ring].tolist())
        if not nb:
            break
        part[m] = nb.most_common(1)[0][0]
    which = part[sy - y0 + 1, sx - x0 + 1]
    return [px[which == p] for p in np.unique(which).tolist()]


# ------------------------------------------------------------ validation ----

def validate(vm, new_labels, cls_all):
    """Coastline, frozen provinces (but for posts away from sea crossings' ends) and the impassable area unchanged;
    no province lost; no province in more pieces than in vanilla (new ones: one piece). Returns (problems,
    provinces in too many pieces)."""
    labels = vm.v3.labels
    changed = new_labels != labels
    problems = []
    land_px = vm.land[labels]
    if (changed & ~land_px).any():
        problems.append(f"{int((changed & ~land_px).sum())} sea/lake pixels changed")
    frozen = changed & vm.frozen[labels] & (~vm.post_ok[labels] | vm.crossing_ends | (new_labels < vm.n3))
    if frozen.any():
        problems.append(f"{int(frozen.sum())} pixels of frozen provinces changed")
    moved = changed & (cls_all[new_labels] != cls_all[labels])
    if moved.any():
        problems.append(f"{int(moved.sum())} px moved between passable and impassable provinces")
    touched = np.unique(np.concatenate([labels[changed], new_labels[changed]]))
    n = len(cls_all)
    present = np.bincount(new_labels.ravel(), minlength=n) > 0
    lost = [i for i in range(n) if not present[i] and (i >= vm.n3 or vm.land[i])]
    if lost:
        problems.append(f"{len(lost)} provinces lost: {lost[:10]}")
    slices = ndimage.find_objects(new_labels + 1)
    split = []
    for i in touched.tolist():
        sl = slices[i]
        if sl is None:
            continue
        allowed = vm.pieces[i] if i < vm.n3 else 1
        if ndimage.label(new_labels[sl] == i, structure=FOUR)[1] > allowed:
            split.append(i)
    if split:
        problems.append(f"{len(split)} provinces in more pieces than before: {split[:10]}")
    return problems, split


def fix_split(vm, new_labels, split, cls_all, ok_all):
    """A province in too many pieces keeps its largest ones; the rest join the neighbour (same class) they touch
    most. Pieces with no such neighbour (islands) are always kept, so the ones given away are those that can go."""
    slices = ndimage.find_objects(new_labels + 1)
    for i in split:
        sl = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in slices[i])
        sub = new_labels[sl]
        cc, n = ndimage.label(sub == i, structure=FOUR)
        sizes = np.bincount(cc.ravel())
        target = {}
        for p in range(1, n + 1):
            m = cc == p
            q = sub[ndimage.binary_dilation(m, structure=FOUR) & ~m]
            q = q[(q != i) & ok_all[q] & (cls_all[q] == cls_all[i])]
            if len(q):
                target[p] = Counter(q.tolist()).most_common(1)[0][0]
        allowed = vm.pieces[i] if i < vm.n3 else 1
        spare = max(0, allowed - (n - len(target)))          # kept pieces left after the islands
        for p in sorted(target, key=lambda p: -sizes[p])[spare:]:
            sub[cc == p] = target[p]


def unused_colours(vm, rows):
    used = {int(p[1:], 16) for p in vm.v3.provinces}
    out = []
    for _, srcs, _, _ in rows:
        c = int(vm.v3.provinces[srcs.most_common(1)[0][0]][1:], 16)
        while c in used or c == 0:
            c = (c + COLOUR_STEP) & 0xFFFFFF
        used.add(c)
        out.append(f"x{c:06X}")
    return out


# ---------------------------------------------------------------- output ----

def write_redraw(vm, new_labels, rows, names, reason, posts):
    """build/redraw/provinces.png, new_provinces.csv and reason.npy."""
    ids = unused_colours(vm, rows)
    provinces = vm.v3.provinces + ids
    config.REDRAW.mkdir(parents=True, exist_ok=True)
    terrain = {norm_hex(k): v for k, v in re.findall(r'^(x[0-9A-Fa-f]{6})="(\w+)"', pdx.read_text(
        config.VIC3 / "map_data" / "province_terrains.txt"), re.MULTILINE)}
    present = np.bincount(new_labels.ravel(), minlength=len(provinces))
    with open(config.REDRAW / "new_provinces.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["province", "parent", "terrain", "impassable", "prime_land", "aoe_state", "owner", "pixels",
                    "sources", "aoe_province"])
        for (idx, srcs, code, c), pid in zip(rows, ids):
            parent = srcs.most_common(1)[0][0]
            w.writerow([pid, vm.v3.provinces[parent], terrain.get(vm.v3.provinces[parent], "plains"), c,
                        int(vm.prime_land[parent]), *names[code], int(present[idx]),
                        " ".join(vm.v3.provinces[s] for s in srcs), posts.get(idx, "")])
    colours = np.array([int(p[1:], 16) for p in provinces], np.uint32)
    packed = colours[new_labels]
    rgb = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=-1).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(config.REDRAW / "provinces.png")
    np.save(config.REDRAW / "reason.npy", reason)
    changed = new_labels != vm.v3.labels
    log(f"{int(changed.sum())} px repainted; {len(np.unique(vm.v3.labels[changed]))} provinces lost land, "
        f"{len({i for i in np.unique(new_labels[changed]).tolist() if i < vm.n3})} gained some, "
        f"{len(ids)} new provinces (median {int(np.median(present[vm.n3:])) if ids else 0} px)")


def write_mod_files():
    """mod/map_data/provinces.png and province_terrains.txt (vanilla plus one line per new province)."""
    if not (config.REDRAW / "provinces.png").exists():
        raise SystemExit(f"{config.REDRAW / 'provinces.png'} is missing: run redraw_provinces.py first")
    out = config.MOD / "map_data"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.REDRAW / "provinces.png", out / "provinces.png")
    rows = read_new_provinces(config.REDRAW)
    terrains = pdx.read_text(config.VIC3 / "map_data" / "province_terrains.txt").rstrip("\n") + "\n"
    terrains += "".join(f'{r["province"]}="{r["terrain"]}"\n' for r in rows)
    (out / "province_terrains.txt").write_text(terrains, encoding="utf-8", newline="")
    print(f"wrote provinces.png and province_terrains.txt ({len(rows)} new provinces) to {out}")


def main():
    log("reading maps")
    vm = VanillaMap()
    unit = aoe_units(vm.aoe)
    new_labels, rows, names, cls_all, ok_all, reason, posts = redraw(vm, unit)
    for _ in range(3):
        problems, split = validate(vm, new_labels, cls_all)
        if not split:
            break
        fix_split(vm, new_labels, split, cls_all, ok_all)
    problems, _ = validate(vm, new_labels, cls_all)
    if problems:
        raise SystemExit("redrawn map failed its checks: " + "; ".join(problems))
    log("checks passed: coastline, frozen provinces and passable/impassable land unchanged, no province lost "
        "or in more pieces than in vanilla")
    write_redraw(vm, new_labels, rows, names, reason, posts)

    log("assigning the redrawn provinces to AoE provinces")
    v3 = read_vic3_map(config.VIC3, config.REDRAW)
    land3 = land_provinces(v3)
    index = {p: i for i, p in enumerate(v3.provinces)}
    fixed = {index[r["province"]]: int(r["aoe_province"])
             for r in read_new_provinces(config.REDRAW) if r["aoe_province"]}
    result = assign(v3, vm.aoe, vm.aoe_at, load_warp(v3, vm.aoe), land3, unit, fixed)
    write_province_map(v3, vm.aoe, *result)
    write_debug_images(v3, vm.aoe, vm.aoe_at, land3, result[0], result[4])


if __name__ == "__main__":
    main()
