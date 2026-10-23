"""Readers for AoE (Vic2) text data: localisation, province positions, region names."""
import csv
import re
from pathlib import Path

import pdx

import config
from aoehistory import TAG_RENAMES


def read_localisation(loc_dir=None):
    """{key: English text}. Vic2 loads files alphabetically and the first definition wins."""
    out = {}
    for path in sorted(Path(loc_dir or config.AOE / "localisation").glob("*.csv"), key=lambda p: p.name.lower()):
        for line in pdx.read_text(path).splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split(";")
            if len(parts) > 1 and parts[0] and parts[1].strip() and parts[0] not in out:
                out[parts[0]] = parts[1].strip()
    # Country names follow aoehistory.TAG_RENAMES; the dropped AoE countries' names go.
    for old, new in TAG_RENAMES.items():
        for suffix in ("", "_ADJ"):
            out.pop(new + suffix, None)
            if old + suffix in out:
                out[new + suffix] = out.pop(old + suffix)
    return out


def read_positions(map_dir=None):
    """{province: (x, y)} in Vic2 map pixel coordinates, preferring the city position.

    positions.txt measures y from the bottom of the map, which is also the row
    index of the province map as loaded (Vic2's BMPs are stored south-up).
    """
    root = pdx.parse_file(Path(map_dir or config.AOE / "map") / "positions.txt")
    out = {}
    for key, _, block in root.pairs():
        if not key.isdigit() or not isinstance(block, pdx.Block):
            continue
        for kind in ("city", "factory", "building_construction", "unit"):
            pos = block.get(kind)
            if isinstance(pos, pdx.Block) and pos.get("x") and pos.get("y"):
                out[int(key)] = (float(pos.get("x")), float(pos.get("y")))
                break
    return out


_REGION_LINE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*\{[^}]*\}\s*#\s*(.+?)\s*$", re.MULTILINE)


def read_region_comments(map_dir=None):
    """{state key: name} from trailing comments in region.txt (`SWE_309 = { ... } #Vestlandet`)."""
    text = pdx.read_text(Path(map_dir or config.AOE / "map") / "region.txt")
    return {m.group(1): m.group(2) for m in _REGION_LINE.finditer(text)}


def read_province_map():
    """Rows of build/province_map.csv with numeric fields converted."""
    rows = []
    with open(config.BUILD / "province_map.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["aoe_province"] = int(r["aoe_province"])
            r["pixels"] = int(r["pixels"])
            for k in ("vic3_x", "vic3_y", "vic2_x", "vic2_y"):
                r[k] = float(r[k])
            rows.append(r)
    return rows
