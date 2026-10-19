"""Test whether Vic3 accepts an edited provinces.png: split one province and give half to a neighbouring state.

    python test_split_province.py                      # default: x076DE0 (Pyrenees, Spain -> France)
    python test_split_province.py xPROV STATE_KEY      # split xPROV, the new half joins STATE_KEY

Run after build_mod.py (which wipes mod/). Writes into mod/:
  map_data/provinces.png          vanilla image with half of the province repainted in a new colour
  map_data/province_terrains.txt  vanilla terrains + the new province (same terrain as the old one)
  map_data/state_regions/...      the new province added to the receiving state
  common/history/states/...       the new province added to the receiving owner's create_state
and build/split_test.png: the area before and after, country borders in black, the new province in red.
"""
import re
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

import config
import mapdata
import pdx

DEFAULT_PROVINCE = "x076DE0"
DEFAULT_STATE = "STATE_AQUITAINE"


def norm(p):
    return "x" + p.strip('"')[1:].upper()


def mod_state_regions():
    """state key -> (path, Block) for the mod's land states."""
    out = {}
    for path in sorted((config.MOD / "map_data" / "state_regions").glob("*.txt")):
        for key, _, block in pdx.parse_file(path).pairs():
            out[key] = (path, block)
    return out


def province_owners(history_path):
    """'xRRGGBB' -> (state key, country tag) from the mod's create_state blocks."""
    out = {}
    for _, _, root in pdx.parse_file(history_path).pairs():
        for skey, _, sb in root.pairs():
            for cs in sb.get_all("create_state"):
                for p in cs.get("owned_provinces").values():
                    out[norm(p)] = (skey.removeprefix("s:"), cs.get("country").removeprefix("c:"))
    return out


def script_references():
    """Provinces named in vanilla scripts the mod doesn't replace (events, decisions, defines...)."""
    refs = set()
    for sub in ("common", "events"):
        for path in (config.VIC3 / sub).rglob("*.txt"):
            if "history" not in path.parts:
                refs |= {norm(m) for m in re.findall(r"\bx[0-9A-Fa-f]{6}\b", pdx.read_text(path))}
    return refs


def check_province(prov, state, regions, owners):
    """Refuse provinces other files point at, so the test only exercises the image + province lists."""
    problems = []
    for key, (_, b) in regions.items():
        hubs = {norm(b.get(h)) for h in ("city", "port", "farm", "mine", "wood") if b.get(h)}
        special = {norm(p) for k in ("impassable", "prime_land") if b.get(k) for p in b.get(k).values()}
        if prov in hubs:
            problems.append(f"{prov} is a hub of {key}")
        if prov in special:
            problems.append(f"{prov} is impassable/prime land in {key}")
    adjacencies = pdx.read_text(config.VIC3 / "map_data" / "adjacencies.csv").upper()
    if prov[1:] in adjacencies:
        problems.append(f"{prov} is in adjacencies.csv")
    if prov in script_references():
        problems.append(f"{prov} is referenced by vanilla scripts")
    if prov not in owners:
        problems.append(f"{prov} has no owner in the mod's state history")
    if state not in regions:
        problems.append(f"{state} is not a mod state")
    if problems:
        sys.exit("\n".join(problems))


def split_mask(labels, p_idx, target):
    """Pixels of the province on the receiving side: the half facing its border with `target`."""
    mask = labels == p_idx
    touching = mask & ndimage.binary_dilation(target, structure=ndimage.generate_binary_structure(2, 1))
    if not touching.any():
        sys.exit("the province doesn't border the receiving state's owned land")
    ys, xs = np.nonzero(mask)
    ty, tx = np.nonzero(touching)
    d = np.array([tx.mean() - xs.mean(), ty.mean() - ys.mean()])
    proj = (xs - xs.mean()) * d[0] + (ys - ys.mean()) * d[1]
    new = np.zeros_like(mask)
    new[ys[proj > np.median(proj)], xs[proj > np.median(proj)]] = True
    for name, part in (("new", new), ("remaining", mask & ~new)):
        n = ndimage.label(part)[1]
        if n != 1:
            sys.exit(f"the {name} half would be in {n} pieces; pick another province")
    if not (new & touching).any():
        sys.exit("the new half doesn't touch the receiving state")
    return new


def unused_color(start, used):
    c = start
    while c in used or c == 0:
        c = (c + 0x0B0705) & 0xFFFFFF
    return c


def insert_id(text, block_start, list_key, new_id, quoted):
    """Add new_id to the first `list_key = { ... }` after block_start."""
    m = re.compile(rf"{list_key}\s*=\s*\{{([^}}]*)\}}").search(text, block_start)
    item = f'"{new_id}"' if quoted else new_id
    return text[:m.end(1)].rstrip() + f" {item} " + text[m.end(1):]


def main():
    prov = norm(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROVINCE
    state = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_STATE
    history_path = config.MOD / "common" / "history" / "states" / "aoe_states.txt"
    regions = mod_state_regions()
    owners = province_owners(history_path)
    check_province(prov, state, regions, owners)

    packed = mapdata.load_packed_rgb(config.VIC3 / "map_data" / "provinces.png")
    colors, inv = np.unique(packed, return_inverse=True)
    labels = inv.reshape(packed.shape)
    ids = [mapdata.hex_id(int(c)) for c in colors]
    p_idx = ids.index(prov)

    # The receiving side: the land of `state` held by a country other than the split province's owner.
    state_owners = {o[1] for o in owners.values() if o[0] == state}
    target_owner = next((t for t in sorted(state_owners) if t != owners[prov][1]), None)
    if target_owner is None:
        sys.exit(f"{state} has no owner other than {owners[prov][1]}")
    in_target = np.array([owners.get(p) == (state, target_owner) for p in ids])
    new_mask = split_mask(labels, p_idx, in_target[labels])

    new_color = unused_color(int(colors[p_idx]), set(colors.tolist()))
    new_id = mapdata.hex_id(new_color)
    if any(new_id in pdx.read_text(path) for path, _ in regions.values()):
        sys.exit(f"{new_id} is already in the mod's state regions; rebuild the mod before running this again")

    before = packed.copy()
    packed[new_mask] = new_color
    rgb = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=-1).astype(np.uint8)
    out_png = config.MOD / "map_data" / "provinces.png"
    Image.fromarray(rgb, "RGB").save(out_png)

    terrains = pdx.read_text(config.VIC3 / "map_data" / "province_terrains.txt")
    terrain = re.search(rf'^{prov}="(\w+)"', terrains, re.MULTILINE | re.IGNORECASE).group(1)
    (config.MOD / "map_data" / "province_terrains.txt").write_text(
        terrains.rstrip("\n") + f'\n{new_id}="{terrain}"\n', encoding="utf-8", newline="")

    region_path, _ = regions[state]
    text = pdx.read_text(region_path)
    text = insert_id(text, re.search(rf"^{state}\s*=\s*\{{", text, re.MULTILINE).start(), "provinces", new_id, True)
    region_path.write_text(text, encoding="utf-8-sig", newline="")

    text = pdx.read_text(history_path)
    block = re.search(rf"s:{state}\s*=\s*\{{", text).start()
    create = re.compile(rf"country\s*=\s*c:{target_owner}\b").search(text, block).start()
    text = insert_id(text, create, "owned_provinces", new_id, False)
    history_path.write_text(text, encoding="utf-8-sig", newline="")

    owner_of = {p: o[1] for p, o in owners.items()}
    owner_of_after = dict(owner_of, **{new_id: target_owner})
    preview = config.BUILD / "split_test.png"
    box = render_pair(before, packed, owner_of, owner_of_after, labels == p_idx, new_mask, preview)

    ys, xs = np.nonzero(labels == p_idx)
    print(f"split {prov} ({terrain}, {int((labels == p_idx).sum())} px, {owners[prov][0]} / {owners[prov][1]})")
    print(f"  {prov} keeps {int((labels == p_idx).sum() - new_mask.sum())} px in {owners[prov][0]}")
    print(f"  new {new_id} gets {int(new_mask.sum())} px -> {state} / {target_owner}")
    print(f"  centre pixel x={int(xs.mean())} y={int(ys.mean())}; preview crop x {box[0]}-{box[2]}, y {box[1]}-{box[3]}")
    print(f"wrote {out_png}, province_terrains.txt, {region_path.name}, {history_path.name}, {preview}")


def render_pair(before, after, owner_before, owner_after, focus, new_mask, path):
    """Before/after crop around the province: owners as flat colours, country borders black, new province red."""
    ys, xs = np.nonzero(focus)
    y0, y1 = max(ys.min() - 60, 0), ys.max() + 60
    x0, x1 = xs.min() - 60, xs.max() + 60
    tiles = []
    for packed, owner_of in ((before, owner_before), (after, owner_after)):
        crop = packed[y0:y1, x0:x1]
        colors, inv = np.unique(crop, return_inverse=True)
        tag = np.array([owner_of.get(mapdata.hex_id(int(c)), "") for c in colors])[inv.reshape(crop.shape)]
        palette = {"": (60, 90, 140)}
        rng = np.random.default_rng(7)
        for t in sorted(set(owner_before.values()) & set(np.unique(tag))):
            palette[t] = tuple(rng.integers(110, 230, 3))
        rgb = np.zeros(crop.shape + (3,), np.uint8)
        for t, c in palette.items():
            rgb[tag == t] = c
        border = np.zeros(crop.shape, bool)
        border[:, 1:] |= tag[:, 1:] != tag[:, :-1]
        border[1:, :] |= tag[1:, :] != tag[:-1, :]
        province_edge = np.zeros(crop.shape, bool)
        province_edge[:, 1:] |= crop[:, 1:] != crop[:, :-1]
        province_edge[1:, :] |= crop[1:, :] != crop[:-1, :]
        rgb[province_edge] = (rgb[province_edge] * 0.75).astype(np.uint8)
        if packed is after:
            nm = new_mask[y0:y1, x0:x1]
            rgb[nm] = (220, 40, 40)
        rgb[border] = 0
        tiles.append(np.kron(rgb, np.ones((6, 6, 1), np.uint8)))
    gap = np.full((tiles[0].shape[0], 18, 3), 255, np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([tiles[0], gap, tiles[1]], axis=1)).save(path)
    return int(x0), int(y0), int(x1), int(y1)


if __name__ == "__main__":
    main()
