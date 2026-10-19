"""Loaders for Vic2 (AoE / vanilla) and Vic3 map data."""
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

import pdx

Image.MAX_IMAGE_PIXELS = None


def load_packed_rgb(path):
    """Image as a 2-D uint32 array of 0xRRGGBB values."""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint32)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


# ---------------------------------------------------------------- Vic2 ----

@dataclass
class Vic2Map:
    root: Path
    labels: np.ndarray                       # province id per pixel, 0 = unknown colour
    names: dict = field(default_factory=dict)  # id -> name from definition.csv
    sea: set = field(default_factory=set)      # sea_starts (includes lakes)
    states: dict = field(default_factory=dict)  # state key -> [province ids]

    @property
    def width(self):
        return self.labels.shape[1]

    @property
    def height(self):
        return self.labels.shape[0]

    def state_of(self):
        """province id -> state key."""
        return {p: s for s, provs in self.states.items() for p in provs}


def read_definition(path):
    colors, names = {}, {}
    for line in pdx.read_text(path).splitlines()[1:]:
        parts = line.split(";")
        if len(parts) < 4 or not parts[0].strip().isdigit():
            continue
        try:
            pid, r, g, b = (int(x) for x in parts[:4])
        except ValueError:
            continue
        colors.setdefault((r << 16) | (g << 8) | b, pid)
        names[pid] = parts[4].strip() if len(parts) > 4 else ""
    return colors, names


def label_pixels(packed, color_to_id):
    colors = np.array(sorted(color_to_id), dtype=np.uint32)
    ids = np.array([color_to_id[c] for c in colors], dtype=np.int32)
    idx = np.clip(np.searchsorted(colors, packed), 0, len(colors) - 1)
    return np.where(colors[idx] == packed, ids[idx], 0).astype(np.int32)


def read_vic2_map(map_dir, with_states=True):
    map_dir = Path(map_dir)
    colors, names = read_definition(map_dir / "definition.csv")
    labels = label_pixels(load_packed_rgb(map_dir / "provinces.bmp"), colors)
    default = pdx.parse_file(map_dir / "default.map")
    sea = {int(v) for v in default.get("sea_starts", pdx.Block()).values() if v.isdigit()}
    states = {}
    if with_states:
        for key, _, provs in pdx.parse_file(map_dir / "region.txt").pairs():
            if isinstance(provs, pdx.Block):
                states[key] = [int(v) for v in provs.values() if v.isdigit()]
    return Vic2Map(map_dir, labels, names, sea, states)


# ---------------------------------------------------------------- Vic3 ----

_HEX = re.compile(r"x[0-9A-Fa-f]{6}")


def hex_id(packed):
    return f"x{packed:06X}"


@dataclass
class Vic3Map:
    labels: np.ndarray        # index into `provinces` per pixel
    provinces: list           # index -> "xRRGGBB"
    land_states: dict         # state key -> parsed Block (vanilla land state regions)
    sea_states: dict          # state key -> parsed Block
    lakes: set                # "xRRGGBB"

    @property
    def width(self):
        return self.labels.shape[1]

    @property
    def height(self):
        return self.labels.shape[0]

    def province_state(self):
        """'xRRGGBB' -> vanilla state key (land and sea)."""
        out = {}
        for states in (self.land_states, self.sea_states):
            for key, block in states.items():
                for p in block.get("provinces", pdx.Block()).values():
                    out[p.upper().replace("X", "x", 1)] = key
        return out


def read_vic3_state_regions(vic3_game):
    land, sea = {}, {}
    for path in sorted((Path(vic3_game) / "map_data" / "state_regions").glob("*.txt")):
        target = sea if path.name.startswith("99_") else land
        for key, _, block in pdx.parse_file(path).pairs():
            if isinstance(block, pdx.Block):
                target[key] = block
    return land, sea


def read_new_provinces(redraw_dir):
    """Rows of redraw_provinces.py's new_provinces.csv ([] if there is none)."""
    path = Path(redraw_dir) / "new_provinces.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_new_provinces(land_states, rows):
    """Put each new province in its parent's state region block: in `provinces`, and in `impassable` /
    `prime_land` where the parent is listed there."""
    norm = lambda p: p.upper().replace("X", "x", 1)
    state_of = {norm(p): key for key, block in land_states.items() for p in block.get("provinces").values()}
    for r in rows:
        block = land_states[state_of[r["parent"]]]
        for list_key in ("provinces", "impassable", "prime_land"):
            ids = block.get(list_key)
            if isinstance(ids, pdx.Block) and r["parent"] in {norm(p) for p in ids.values()}:
                ids.append(r["province"])


def read_vic3_map(vic3_game, redraw_dir=None):
    """The vanilla Vic3 map; with redraw_dir (redraw_provinces.py's output, once it has a provinces.png), the
    redrawn provinces.png, and the new provinces added to the vanilla state region blocks."""
    vic3_game = Path(vic3_game)
    redrawn = redraw_dir is not None and (Path(redraw_dir) / "provinces.png").exists()
    packed = load_packed_rgb((Path(redraw_dir) if redrawn else vic3_game / "map_data") / "provinces.png")
    colors, inverse = np.unique(packed, return_inverse=True)
    labels = inverse.reshape(packed.shape).astype(np.int32)
    provinces = [hex_id(int(c)) for c in colors]
    land, sea = read_vic3_state_regions(vic3_game)
    if redrawn:
        add_new_provinces(land, read_new_provinces(redraw_dir))
    default_map = pdx.read_text(vic3_game / "map_data" / "default.map")
    lakes_block = re.search(r"lakes\s*=\s*\{([^}]*)\}", default_map)
    lakes = {p.upper().replace("X", "x", 1) for p in _HEX.findall(lakes_block.group(1))} if lakes_block else set()
    return Vic3Map(labels, provinces, land, sea, lakes)


# ------------------------------------------------------------- helpers ----

def label_centroids(labels, n_labels, wrap=True):
    """Per-label pixel count and circular-in-x centroid.

    Returns (count, sum_cos, sum_sin, sum_y): enough to combine labels into
    groups before turning the sums into centroids with sums_to_centroid.
    """
    h, w = labels.shape
    theta = 2 * np.pi * (np.arange(w) + 0.5) / w
    cos_col, sin_col = np.cos(theta), np.sin(theta)
    count = np.zeros(n_labels)
    sum_cos = np.zeros(n_labels)
    sum_sin = np.zeros(n_labels)
    sum_y = np.zeros(n_labels)
    for y0 in range(0, h, 256):
        chunk = labels[y0:y0 + 256]
        flat = chunk.ravel()
        rows = chunk.shape[0]
        count += np.bincount(flat, minlength=n_labels)
        sum_cos += np.bincount(flat, weights=np.tile(cos_col, rows), minlength=n_labels)
        sum_sin += np.bincount(flat, weights=np.tile(sin_col, rows), minlength=n_labels)
        ys = np.repeat(np.arange(y0, y0 + rows) + 0.5, w)
        sum_y += np.bincount(flat, weights=ys, minlength=n_labels)
    return count, sum_cos, sum_sin, sum_y


def province_borders(labels):
    """Shared borders between labels, wrapping in x.

    Returns {(a, b): (length, mid_x, mid_y)} with a < b, where (mid_x, mid_y)
    is the mean position of the border in pixel-index coordinates.
    """
    h, w = labels.shape
    n = int(labels.max()) + 1
    parts = []
    right = np.roll(labels, -1, axis=1)
    ys, xs = np.nonzero(labels != right)
    parts.append((labels[ys, xs], right[ys, xs], xs + 0.5, ys.astype(float)))
    ys, xs = np.nonzero(labels[:-1] != labels[1:])
    parts.append((labels[ys, xs], labels[ys + 1, xs], xs.astype(float), ys + 0.5))
    a = np.concatenate([p[0] for p in parts]).astype(np.int64)
    b = np.concatenate([p[1] for p in parts]).astype(np.int64)
    x = np.concatenate([p[2] for p in parts])
    y = np.concatenate([p[3] for p in parts])
    keys, inverse = np.unique(np.minimum(a, b) * n + np.maximum(a, b), return_inverse=True)
    length = np.bincount(inverse)
    mx = np.bincount(inverse, weights=x) / length
    my = np.bincount(inverse, weights=y) / length
    return {(int(k // n), int(k % n)): (int(c), float(px), float(py))
            for k, c, px, py in zip(keys, length, mx, my)}


def sums_to_centroid(count, sum_cos, sum_sin, sum_y, width):
    """(x, y, spread_x): spread_x is an approximate std-dev in pixels."""
    angle = np.arctan2(sum_sin, sum_cos) % (2 * np.pi)
    x = angle / (2 * np.pi) * width - 0.5
    r = np.clip(np.hypot(sum_cos, sum_sin) / np.maximum(count, 1), 1e-9, 1.0)
    spread = width / (2 * np.pi) * np.sqrt(-2 * np.log(r))
    return x, sum_y / np.maximum(count, 1) - 0.5, spread
