# Age of Enlightenment — Victoria 3

A Victoria 3 total conversion of [AoE Remake](https://github.com/MaTi5893/AoE-remake) (multiplayer version, `main` branch),
a Victoria 2 mod set in 1700. The map, states, countries, cultures, religions, pops and flags are generated
from the AoE source by the scripts in `tools/`; the result is in `mod/`.

## Building

Requires Python 3.12 with `numpy`, `pillow` and `scipy` (`pip install -r requirements.txt`), and local installs
of Victoria 2 and Victoria 3 (paths in `tools/config.py`, or set `VIC2_DIR` / `VIC3_DIR`).

```
cd tools
python fetch_sources.py          # AoE Remake + converter province mappings -> sources/
python build_mod.py              # everything -> mod/ (about 5 minutes)
python build_mod.py --skip-map   # reuse the province alignment and redraw (about 1 minute)
python validate_mod.py           # check references before launching the game
python render_political.py       # build/political_1700.png
python compare_region.py NAME AOE_PROVINCE...   # AoE (Vic2) vs. generated Vic3, side by side
python check_topology.py AOE_PROVINCE...        # which provinces border which, AoE vs. Vic3
```

`build_province_map.py --reuse-warp` skips the slow warp fit when only the assignment step changed; run
`redraw_provinces.py` after it, since the redraw rewrites `build/province_map.csv` for the redrawn map.
`fetch_sources.py --checkout` re-applies the sparse checkout of `sources/AoE-main` without fetching.

To play, link `mod/` into `Documents/Paradox Interactive/Victoria 3/mod/` (a directory junction works).

## How the conversion works

| Step | Script | What it does |
|---|---|---|
| Province alignment | `build_province_map.py` | Warps the Vic2 map onto Vic3's projection (spline fitted on EU4→Vic2 / EU4→Vic3 converter mappings, then refined so the converters' matching province groups line up) and assigns each Vic3 province to an AoE province: majority vote, then every AoE province gets land, then a local search restores AoE's province adjacency (enclaves, which provinces border which) with Vic3's own state borders breaking near-ties. |
| Province redraw | `redraw_provinces.py` | Moves Vic3 province lines onto AoE's 1700 country borders where a province straddles one, and adds new provinces where a piece is too big to give to a neighbour. Coastlines, sea, impassable/passable land and the provinces in `adjacencies.csv` don't change; provinces named in vanilla scripts can give land to a neighbour but are never split into new provinces. Writes `map_data/provinces.png` and `province_terrains.txt`; new provinces join their parent's state, impassable and prime-land lists. The redrawn provinces are then assigned to AoE provinces again, without crossing an owner border. |
| States | `build_states.py` | Regroups Vic3 provinces into AoE's states. Resources, arable land and traits are split from the overlapping vanilla states by area; hubs, locators, strategic and geographic regions are regenerated. |
| History | `build_history.py` | 1700 ownership, homelands, claims, pops (Vic2 sizes × 4), laws, technology tiers, subjects. Unowned land becomes decentralized native nations. AoE cultures missing from Vic3 copy the vanilla culture living in the same place. |
| Flags | `build_flags.py` | AoE's TGA flags become textured coats of arms, with monarchy/republic/dominion variants. |

Debug output (alignment images, province and state tables) goes to `build/`.
