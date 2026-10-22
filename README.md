# Age of Enlightenment — Victoria 3

A Victoria 3 total conversion of [AoE Remake](https://github.com/MaTi5893/AoE-remake) (multiplayer version, `main` branch),
a Victoria 2 mod set in 1700. The map, states, countries, cultures, religions, pops and flags are generated
from the AoE source by the scripts in `tools/`; the result is written to
`Documents/Paradox Interactive/Victoria 3/mod/Age of Enlightenment/` (set `MOD_DIR` to build elsewhere).

## Building

Requires Python 3.12 with `numpy`, `pillow` and `scipy` (`pip install -r requirements.txt`), and local installs
of Victoria 2 and Victoria 3 (paths in `tools/config.py`, or set `VIC2_DIR` / `VIC3_DIR`).

```
cd tools
python fetch_sources.py          # AoE Remake + converter province mappings -> sources/
python build_mod.py              # everything -> the game's mod folder (about 5 minutes)
python build_mod.py --skip-map   # reuse the province alignment and redraw (about 1 minute)
python validate_mod.py           # check references before launching the game
python render_political.py       # build/political_1700.png
python compare_region.py NAME AOE_PROVINCE...   # AoE (Vic2) vs. generated Vic3, side by side
python check_topology.py AOE_PROVINCE...        # which provinces border which, AoE vs. Vic3
```

`build_province_map.py --reuse-warp` skips the slow warp fit when only the assignment step changed; run
`redraw_provinces.py` after it, since the redraw rewrites `build/province_map.csv` for the redrawn map.
`fetch_sources.py --checkout` re-applies the sparse checkout of `sources/AoE-main` without fetching.

The build writes straight into the game's mod folder, so the mod is ready to enable in the launcher. It only
replaces the folders it generates (`.metadata`, `common`, `gfx`, `localization`, `map_data`).

## How the conversion works

| Step | Script | What it does |
|---|---|---|
| Province alignment | `build_province_map.py` | Warps the Vic2 map onto Vic3's projection (spline fitted on EU4→Vic2 / EU4→Vic3 converter mappings, then refined so coastlines line up; the interior keeps the spline, since forcing the converters' province groups to match pulled land out of place) and assigns each Vic3 province to an AoE province: majority vote (each Vic3 landmass only counts AoE land of its own landmass, so Alaska doesn't spill onto Chukotka's tip, which AoE's map cuts at 180°; land under AoE sea takes the nearest AoE land on its landmass, across the map's edge too), then every AoE province gets land, then a local search restores AoE's province adjacency (enclaves, which provinces border which) with Vic3's own state borders breaking near-ties. |
| Province redraw | `redraw_provinces.py` | Snaps Vic3 province lines to AoE's state and 1700 country borders, overlaid with the alignment warp. Each piece of a province on the wrong side of such a border first joins a neighbouring province on the right side; a new province is added only where no such neighbour exists or it would grow past twice its size. Coastlines, sea, impassable/passable land and the provinces in `adjacencies.csv` don't change; provinces named in vanilla scripts can give land to a neighbour but are never split into new provinces, except that a state/owner piece left without a province (trading posts and small enclaves, mostly in India) gets one carved where the warp puts it, on the coast if it is coastal in AoE (taking a few pixels of an `adjacencies.csv` province if need be, away from the crossing's ends). Writes `map_data/provinces.png` and `province_terrains.txt`; new provinces join their parent's state, impassable and prime-land lists. The redrawn provinces are then assigned to AoE provinces again, without crossing a state or country border. |
| States | `build_states.py` | Regroups Vic3 provinces into AoE's states. Resources, arable land and traits are split from the overlapping vanilla states by area; hubs, locators, strategic and geographic regions are regenerated. |
| History | `build_history.py` | 1700 ownership, homelands, claims, pops (Vic2 sizes × 4), laws, technology tiers, subjects. Unowned land becomes decentralized native nations. AoE cultures missing from Vic3 copy the vanilla culture living in the same place. |
| Flags | `build_flags.py` | AoE's TGA flags become textured coats of arms, with monarchy/republic/dominion variants. |

Debug output (alignment images, province and state tables) goes to `build/`.
