"""Side-by-side check of one area: AoE's 1700 map in Vic2 vs. the generated Vic3 map.

    python compare_region.py NAME AOE_PROVINCE [AOE_PROVINCE ...] [--radius 70]

Writes build/compare_NAME.png. Both halves are coloured by 1700 owner and
labelled with AoE province ids, so misassigned provinces stand out. On the
Vic3 side, thin lines are AoE province borders as mapped; thick lines are
country borders.
"""
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

import config
import pdx
from aoedata import read_province_map
from aoehistory import read_countries, read_provinces
from mapdata import label_centroids, read_vic2_map, read_vic3_map, sums_to_centroid

PANEL = 760


def borders(arr):
    b = np.zeros(arr.shape, bool)
    b[:, :-1] |= arr[:, :-1] != arr[:, 1:]
    b[:-1, :] |= arr[:-1, :] != arr[1:, :]
    return b


def label_points(draw, labels, ids, x0, y0, scale, w):
    """Draw each id at the centroid of its pixels in a crop."""
    stats = label_centroids(labels, int(labels.max()) + 1)
    cx, cy, _ = sums_to_centroid(*stats, labels.shape[1])
    for i in ids:
        if i < len(stats[0]) and stats[0][i] > 6:
            draw.text(((cx[i] + 0.5) * scale, (cy[i] + 0.5) * scale), str(i), fill=(0, 0, 0), anchor="mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("provinces", nargs="+", type=int)
    ap.add_argument("--radius", type=int, default=70, help="half-size of the Vic2 crop in pixels")
    args = ap.parse_args()

    countries = read_countries()
    provs = read_provinces()
    color = {t: c.color for t, c in countries.items()}
    defs = pdx.parse_file(config.MOD / "common" / "country_definitions" / "aoe_countries.txt")
    color.update({t: tuple(int(x) for x in b.get("color").values()) for t, _, b in defs.pairs()})

    # ---- Vic2 side
    aoe = read_vic2_map(config.AOE / "map")
    pts = np.array([np.argwhere(aoe.labels == p).mean(axis=0) for p in args.provinces])
    cy, cx = pts.mean(axis=0).astype(int)
    r = args.radius
    crop = aoe.labels[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r]
    owner_ids = {}
    rgb = np.zeros(crop.shape + (3,), np.uint8)
    rgb[:] = (40, 60, 100)
    owner = np.full(crop.shape, -1)
    for pid in np.unique(crop):
        p = provs.get(int(pid))
        if p and p.owner:
            rgb[crop == pid] = color.get(p.owner, (128, 128, 128))
            owner[crop == pid] = owner_ids.setdefault(p.owner, len(owner_ids))
        elif int(pid) not in aoe.sea:
            rgb[crop == pid] = (200, 200, 200)
    rgb[borders(crop)] = (rgb[borders(crop)] * 0.55).astype(np.uint8)
    rgb[borders(owner)] = (0, 0, 0)
    # Vic2 province maps load south-up; flip so both panels are north-up.
    crop, rgb = crop[::-1], rgb[::-1]
    scale2 = PANEL / crop.shape[1]
    left = Image.fromarray(np.ascontiguousarray(rgb)).resize((PANEL, int(crop.shape[0] * scale2)), Image.NEAREST)
    label_points(ImageDraw.Draw(left), np.ascontiguousarray(crop), [int(x) for x in np.unique(crop)], 0, 0, scale2, crop.shape[1])

    # ---- Vic3 side
    rows = read_province_map()
    v3 = read_vic3_map(config.VIC3)
    index = {p: i for i, p in enumerate(v3.provinces)}
    aoe_of = np.zeros(len(v3.provinces), np.int32)
    for row in rows:
        aoe_of[index[row["vic3_province"]]] = row["aoe_province"]
    owner_of = {}
    states = pdx.parse_file(config.MOD / "common" / "history" / "states" / "aoe_states.txt")
    for _, _, root in states.pairs():
        for _, _, sb in root.pairs():
            for cs in sb.get_all("create_state"):
                for p in cs.get("owned_provinces").values():
                    owner_of[index["x" + p[1:].upper()]] = cs.get("country").removeprefix("c:")
    sel = [row for row in rows if row["aoe_province"] in args.provinces]
    vx = np.mean([row["vic3_x"] for row in sel])
    vy = np.mean([row["vic3_y"] for row in sel])
    r3 = int(r * v3.width / aoe.width)
    y0, x0 = max(int(vy) - r3, 0), max(int(vx) - r3, 0)
    lab = v3.labels[y0:int(vy) + r3, x0:int(vx) + r3]
    a = aoe_of[lab]
    rgb = np.zeros(lab.shape + (3,), np.uint8)
    rgb[:] = (40, 60, 100)
    o = np.full(lab.shape, -1)
    tag_ids = {}
    for i in np.unique(lab):
        if i in owner_of:
            m = lab == i
            rgb[m] = color.get(owner_of[i], (128, 128, 128))
            o[m] = tag_ids.setdefault(owner_of[i], len(tag_ids))
    rgb[borders(a)] = (rgb[borders(a)] * 0.55).astype(np.uint8)
    rgb[borders(o)] = (0, 0, 0)
    scale3 = PANEL / lab.shape[1]
    right = Image.fromarray(rgb).resize((PANEL, int(lab.shape[0] * scale3)), Image.NEAREST)
    label_points(ImageDraw.Draw(right), a, [int(x) for x in np.unique(a) if x], 0, 0, scale3, lab.shape[1])

    out = Image.new("RGB", (PANEL * 2 + 10, max(left.height, right.height)), (255, 255, 255))
    out.paste(left, (0, 0))
    out.paste(right, (PANEL + 10, 0))
    path = config.BUILD / f"compare_{args.name}.png"
    out.save(path)
    print(f"wrote {path}  (left: AoE in Vic2, right: generated Vic3)")


if __name__ == "__main__":
    main()
