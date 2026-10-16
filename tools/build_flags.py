"""Use AoE's Vic2 flags as Vic3 coats of arms.

Each AoE flag TGA becomes a textured emblem filling the whole flag (the same
approach the Paradox Game Converters use). Vic2 picks TAG_<type>.tga by the
government's flag type and falls back to TAG.tga; the Vic3 flag definitions
reproduce that with the monarchy/republic triggers and, for AoE's dominion
flags, the colonial administration law.

Generated native nations have no AoE flag and get a simple two-colour emblem.

Reads mod/common/country_definitions/aoe_countries.txt (from build_history.py).
"""
import re
import shutil

import numpy as np

import config
import pdx

FLAG_DIR = "aoe"
VARIANTS = [  # (file suffix, priority, trigger)
    ("", 1, None),
    ("_monarchy", 10, "coa_def_monarchy_flag_trigger = yes"),
    ("_republic", 10, "coa_def_republic_flag_trigger = yes"),
    ("_dominion", 20, "scope:actor ?= { has_law_or_variant = law_type:law_colonial_administration }"),
]
NATIVE_EMBLEMS = ["ce_sun.dds", "ce_spearhead.dds", "ce_horse_tribal.dds", "ce_comanche_sun.dds",
                  "ce_native_american_pattern_diamond.dds", "ce_circle.dds"]


def main():
    defs = pdx.parse_file(config.MOD / "common" / "country_definitions" / "aoe_countries.txt")
    flags = {p.stem.lower(): p for p in (config.AOE / "gfx" / "flags").glob("*.tga")}
    emblem_dir = config.MOD / "gfx" / "coat_of_arms" / "textured_emblems" / FLAG_DIR
    if emblem_dir.exists():
        shutil.rmtree(emblem_dir)
    emblem_dir.mkdir(parents=True)

    coas, flag_defs, missing = [], [], []
    for tag, _, block in defs.pairs():
        variants = []
        for suffix, priority, trigger in VARIANTS:
            src = flags.get(f"{tag}{suffix}".lower())
            if src is None:
                continue
            name = f"aoe_{tag}{suffix}"
            shutil.copyfile(src, emblem_dir / f"{tag}{suffix}.tga")
            coas.append(f'{name} = {{ textured_emblem = {{ texture = "{FLAG_DIR}/{tag}{suffix}.tga" }} }}')
            variants.append((name, priority, trigger))
        if not any(p == 1 for _, p, _ in variants):
            # No base flag: use the first variant as the default, or build one.
            if variants:
                variants.insert(0, (variants[0][0], 1, None))
            else:
                if not re.fullmatch(r"N[0-9A-Z]{2}", tag):
                    missing.append(tag)
                color = block.get("color").values()
                rng = np.random.default_rng(sum(map(ord, tag)))
                emblem = NATIVE_EMBLEMS[int(rng.integers(len(NATIVE_EMBLEMS)))]
                coas.append(f'aoe_{tag} = {{\n\tpattern = "pattern_solid.tga"\n'
                            f'\tcolor1 = rgb {{ {" ".join(color)} }}\n\tcolor2 = "white"\n'
                            f'\tcolored_emblem = {{\n\t\ttexture = "{emblem}"\n\t\tcolor1 = "white"\n'
                            f'\t\tcolor2 = "black"\n\t\tinstance = {{ scale = {{ 0.6 0.6 }} }}\n\t}}\n}}')
                variants.append((f"aoe_{tag}", 1, None))

        lines = [f"{tag} = {{"]
        for name, priority, trigger in variants:
            lines += ["\tflag_definition = {", f"\t\tcoa = {name}", f"\t\tsubject_canton = {name}",
                      "\t\tallow_overlord_canton = yes", f"\t\tpriority = {priority}"]
            if trigger:
                lines.append(f"\t\ttrigger = {{ {trigger} }}")
            lines.append("\t}")
        lines.append("}")
        flag_defs.append("\n".join(lines))

    coa_path = config.MOD / "common" / "coat_of_arms" / "coat_of_arms" / "aoe_flags.txt"
    coa_path.parent.mkdir(parents=True, exist_ok=True)
    coa_path.write_text("\n".join(coas) + "\n", encoding="utf-8-sig")
    fd_path = config.MOD / "common" / "flag_definitions" / "aoe_flag_definitions.txt"
    fd_path.parent.mkdir(parents=True, exist_ok=True)
    fd_path.write_text("\n\n".join(flag_defs) + "\n", encoding="utf-8-sig")
    print(f"{len(flag_defs)} flag definitions, {len(list(emblem_dir.glob('*.tga')))} AoE flag images")
    if missing:
        print(f"AoE countries with no flag file: {' '.join(missing)}")


if __name__ == "__main__":
    main()
