"""Cross-check the generated mod for broken references before loading it in Vic3.

Checks that every culture, religion, state, country tag, law, scripted effect,
building and flag texture the mod refers to is defined, either by the mod or by
the vanilla files the mod doesn't replace.
"""
import json
import re
import sys
from collections import Counter

import config
import pdx


def keys_in(folder, mod_first=True, replaced=False):
    """Top-level keys defined in vanilla + mod files of a common/ folder."""
    out = set()
    dirs = [config.MOD / "common" / folder] + ([] if replaced else [config.VIC3 / "common" / folder])
    for d in dirs:
        for path in d.glob("*.txt"):
            out |= {k for k, _, v in pdx.parse_file(path).pairs() if isinstance(v, pdx.Block) or folder == "scripted_effects"}
    return out


def main():
    mod = config.MOD
    problems = []
    meta = json.loads((mod / ".metadata" / "metadata.json").read_text(encoding="utf-8"))
    replaced = set(meta["game_custom_data"]["replace_paths"])

    cultures = keys_in("cultures")
    religions = keys_in("religions")
    laws = keys_in("laws")
    buildings = keys_in("buildings")
    effects = keys_in("scripted_effects")
    states = set()
    for path in (mod / "map_data" / "state_regions").glob("*.txt"):
        states |= {k for k, _, _ in pdx.parse_file(path).pairs()}
    tags = {k for k, _, _ in pdx.parse_file(mod / "common" / "country_definitions" / "aoe_countries.txt").pairs()}

    # Country definitions
    for tag, _, b in pdx.parse_file(mod / "common" / "country_definitions" / "aoe_countries.txt").pairs():
        for c in b.get("cultures").values():
            if c not in cultures:
                problems.append(f"country {tag}: unknown culture {c}")
        if b.get("religion") and b.get("religion") not in religions:
            problems.append(f"country {tag}: unknown religion {b.get('religion')}")
        if b.get("capital") not in states:
            problems.append(f"country {tag}: capital {b.get('capital')} is not a state")

    # History text: scan references with regexes (cheaper than walking every effect).
    hist = mod / "common" / "history"
    text = {p: p.read_text(encoding="utf-8-sig") for p in hist.rglob("*.txt")}
    owned_capitals = {}
    for path, t in text.items():
        for m in re.finditer(r"\bc:([A-Z0-9]{3})\b", t):
            if m.group(1) not in tags:
                problems.append(f"{path.name}: unknown country c:{m.group(1)}")
        for m in re.finditer(r"\bs:(STATE_\w+)", t):
            if m.group(1) not in states:
                problems.append(f"{path.name}: unknown state {m.group(1)}")
        for m in re.finditer(r"\bcu:(\w+)|\bculture = (\w+)", t):
            c = m.group(1) or m.group(2)
            if c not in cultures:
                problems.append(f"{path.name}: unknown culture {c}")
        for m in re.finditer(r"\breligion = (\w+)", t):
            if m.group(1) not in religions:
                problems.append(f"{path.name}: unknown religion {m.group(1)}")
        for m in re.finditer(r"law_type:(\w+)", t):
            if m.group(1) not in laws:
                problems.append(f"{path.name}: unknown law {m.group(1)}")
        for m in re.finditer(r"^\s*(effect_\w+) = yes", t, re.MULTILINE):
            if m.group(1) not in effects:
                problems.append(f"{path.name}: unknown scripted effect {m.group(1)}")
        for m in re.finditer(r'building = "(\w+)"', t):
            if m.group(1) not in buildings:
                problems.append(f"{path.name}: unknown building {m.group(1)}")

    # State keys anywhere else in the mod's common/ files (strategic and geographic regions...).
    for path in (mod / "common").rglob("*.txt"):
        if hist in path.parents:
            continue
        for key in set(re.findall(r"\b(STATE_[A-Z0-9_]+)", path.read_text(encoding="utf-8-sig"))):
            if key not in states:
                problems.append(f"{path.name}: unknown state {key}")
    in_regions = Counter()
    for path in (mod / "common" / "strategic_regions").glob("*.txt"):
        for _, _, rb in pdx.parse_file(path).pairs():
            in_regions.update(rb.get("states", pdx.Block()).values())
    problems += [f"{s} is in {in_regions[s]} strategic regions" for s in sorted(states) if in_regions[s] != 1]

    # Every country owns land in its capital state.
    owned = Counter()
    states_hist = pdx.parse_file(hist / "states" / "aoe_states.txt")
    owner_states = {}
    for _, _, root in states_hist.pairs():
        for skey, _, sb in root.pairs():
            for cs in sb.get_all("create_state"):
                owner_states.setdefault(cs.get("country").removeprefix("c:"), set()).add(skey.removeprefix("s:"))
    for tag, _, b in pdx.parse_file(mod / "common" / "country_definitions" / "aoe_countries.txt").pairs():
        if b.get("capital") not in owner_states.get(tag, set()):
            problems.append(f"country {tag}: doesn't own its capital {b.get('capital')}")

    # Flag textures exist.
    emblems = mod / "gfx" / "coat_of_arms" / "textured_emblems"
    for m in re.finditer(r'texture = "([^"]+\.tga)"', (mod / "common" / "coat_of_arms" / "coat_of_arms" / "aoe_flags.txt").read_text(encoding="utf-8-sig")):
        if not (emblems / m.group(1)).exists():
            problems.append(f"missing flag texture {m.group(1)}")

    # Localisation keys for every state and country.
    loc = (mod / "localization" / "english" / "aoe_countries_l_english.yml").read_text(encoding="utf-8-sig")
    loc += (mod / "localization" / "english" / "replace" / "aoe_states_l_english.yml").read_text(encoding="utf-8-sig")
    loc_keys = set(re.findall(r"^\s*([\w.]+):\d*\s", loc, re.MULTILINE))
    problems += [f"no localisation for {k}" for k in sorted(states | tags) if k not in loc_keys]

    counts = Counter(p.split(":")[0] for p in problems)
    for p in problems[:60]:
        print(p)
    print(f"{len(problems)} problems" + (f" ({dict(counts)})" if problems else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
