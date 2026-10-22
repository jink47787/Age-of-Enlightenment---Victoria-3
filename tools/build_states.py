"""Generate Vic3 state regions from AoE's Vic2 states.

Reads build/province_map.csv and the redrawn province map (build_province_map.py, redraw_provinces.py)
and writes:
  mod/map_data/state_regions/*.txt          one file per vanilla file name, replacing vanilla
  mod/common/strategic_regions/*.txt        land strategic regions listing the new states
  mod/gfx/map/map_object_data/generated_map_object_locators_*.txt   hub locators
  mod/localization/<lang>/replace/aoe_states_l_<lang>.yml           state and hub names
  build/state_map.csv                       AoE state -> Vic3 state key / id / hubs

Arable land, resources, traits and subsistence buildings come from the vanilla
Vic3 states each new state overlaps, split in proportion to overlap so world
totals stay the same.

Hub names and places are period-corrected (1700-1821) by hub_names.py: global renames
in PERIOD_NAMES and per-state renames and moves in HUB_OVERRIDES.
"""
import csv
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

import config
import hub_names
import pdx
from aoedata import read_localisation, read_positions, read_province_map, read_region_comments
from aoehistory import read_pops
from build_province_map import GridWarp
from mapdata import province_borders, read_definition, read_vic3_map

HUB_TYPES = ["city", "port", "farm", "mine", "wood"]
COASTAL_BUILDINGS = {"building_fishing_wharf", "building_whaling_station"}
COASTAL_TRAIT = re.compile(r"harbo|port_mahon|coastal")
AMOUNT_FIELDS = ("amount", "undiscovered_amount", "discovered_amount", "depleted_amount")
LANGUAGES = ["english", "braz_por", "french", "german", "japanese", "korean", "polish", "russian",
             "simp_chinese", "spanish", "turkish"]


def norm_hex(p):
    return "x" + p[1:].upper()


def apportion(total, weights):
    """Split an integer total by weights using largest remainders."""
    weights = {k: w for k, w in weights.items() if w > 0}
    s = sum(weights.values())
    if total <= 0 or s <= 0:
        return {}
    quotas = {k: total * w / s for k, w in weights.items()}
    out = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: quotas[k] - out[k], reverse=True)[:total - sum(out.values())]:
        out[k] += 1
    return {k: v for k, v in out.items() if v > 0}


_TRANSLIT = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss",
                           "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "þ": "th", "Þ": "TH", "ı": "i"})


def make_key(name, taken):
    ascii_name = unicodedata.normalize("NFKD", name.translate(_TRANSLIT)).encode("ascii", "ignore").decode()
    base = "STATE_" + (re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_").upper() or "UNNAMED")
    key, n = base, 2
    while key in taken:
        key, n = f"{base}_{n}", n + 1
    taken.add(key)
    return key


def read_vanilla():
    """Vanilla land/sea state blocks, which file each lives in, strategic regions and hub names."""
    land, sea, file_of = {}, {}, {}
    for path in sorted((config.VIC3 / "map_data" / "state_regions").glob("*.txt")):
        for key, _, block in pdx.parse_file(path).pairs():
            if isinstance(block, pdx.Block):
                (sea if path.name.startswith("99_") else land)[key] = block
                file_of[key] = path.name
    regions = {}  # region key -> (file name, block)
    for path in sorted((config.VIC3 / "common" / "strategic_regions").glob("*.txt")):
        for key, _, block in pdx.parse_file(path).pairs():
            if isinstance(block, pdx.Block):
                regions[key] = (path.name, block)
    hub_names = {}
    hub_loc = pdx.read_text(config.VIC3 / "localization" / "english" / "hub_names_l_english.yml")
    for m in re.finditer(r'^\s*(HUB_NAME_\w+?)_(city|port|farm|mine|wood):\d*\s*"(.*)"', hub_loc, re.MULTILINE):
        hub_names[(m.group(1)[len("HUB_NAME_"):], m.group(2))] = m.group(3)
    return land, sea, file_of, regions, hub_names


def read_vanilla_locators(land):
    """{(vanilla state key, hub type): (x, z, rotation text)}: vanilla's hand-placed hub locators."""
    key_of = {int(b.get("id")): k for k, b in land.items()}
    out = {}
    for t in HUB_TYPES:
        text = pdx.read_text(config.VIC3 / "gfx" / "map" / "map_object_data" / f"generated_map_object_locators_{t}.txt")
        for m in re.finditer(r"id=(\d+)\s*position=\{\s*([-\d.]+)\s+[-\d.]+\s+([-\d.]+)\s*\}\s*rotation=\{([^}]*)\}", text):
            if int(m.group(1)) in key_of:
                out[(key_of[int(m.group(1))], t)] = (float(m.group(2)), float(m.group(3)), m.group(4).strip())
    return out


def place_key(name):
    """Name normalised for matching: no accents, case, punctuation or Saint/St difference."""
    ascii_name = unicodedata.normalize("NFKD", name.translate(_TRANSLIT)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\b(saint|sankt|san|sao|santo|santa)\b", "st", ascii_name))


def wrap_dist(x1, y1, x2, y2, width):
    dx = (x1 - x2 + width / 2) % width - width / 2
    return np.hypot(dx, y1 - y2)


class StateBuilder:
    def __init__(self):
        self.rows = read_province_map()
        self.v3 = read_vic3_map(config.VIC3, config.REDRAW)
        _, self.sea, self.file_of, self.regions, self.vanilla_hub_names = read_vanilla()
        self.land = self.v3.land_states     # vanilla blocks, with the redrawn map's new provinces added
        self.loc = read_localisation()
        self.region_names = read_region_comments()
        self.positions = read_positions()
        _, self.aoe_def_names = read_definition(config.AOE / "map" / "definition.csv")
        self.aoe_regions = [(k, [int(v) for v in b.values() if v.isdigit()])
                            for k, _, b in pdx.parse_file(config.AOE / "map" / "region.txt").pairs()
                            if isinstance(b, pdx.Block)]

        self.index = {p: i for i, p in enumerate(self.v3.provinces)}
        self.row_of = {self.index[r["vic3_province"]]: r for r in self.rows}
        self.impassable = {self.index[norm_hex(p)] for b in self.land.values()
                           for p in b.get("impassable", pdx.Block()).values()}
        self.prime_land = {self.index[norm_hex(p)] for b in self.land.values()
                           for p in b.get("prime_land", pdx.Block()).values()}
        self.sea_state_of = {self.index[norm_hex(p)]: k for k, b in self.sea.items()
                             for p in b.get("provinces").values()}
        self.vanilla_region = {s: r for r, (_, b) in self.regions.items() for s in b.get("states", pdx.Block()).values()}
        self.vanilla_locators = read_vanilla_locators(self.land)
        self.warnings = []
        self.population = {p: sum(pop.size for pop in pops) for p, pops in read_pops().items()}
        g = np.load(config.BUILD / "warp_grid.npz")
        self.warp = GridWarp(g["x"], g["y"], float(g["step"]), float(g["origin"]), self.v3.width, int(g["w2"]))

        print("computing province borders")
        self.sea_borders = defaultdict(list)  # land province -> [(sea state, length, (x, y))]
        for (a, b), (length, mx, my) in province_borders(self.v3.labels).items():
            for land_p, sea_p in ((a, b), (b, a)):
                if land_p in self.row_of and sea_p in self.sea_state_of:
                    self.sea_borders[land_p].append((self.sea_state_of[sea_p], length, (mx, my)))

        order = np.argsort(self.v3.labels.ravel(), kind="stable")
        starts = np.concatenate([[0], np.cumsum(np.bincount(self.v3.labels.ravel(), minlength=len(self.v3.provinces)))])
        self.pixel_order, self.pixel_starts = order, starts

    # -------------------------------------------------------------- helpers ----

    def province_name(self, aoe_prov):
        return self.loc.get(f"PROV{aoe_prov}") or self.aoe_def_names.get(aoe_prov) or f"Province {aoe_prov}"

    def inside_point(self, i):
        """The pixel of province i closest to its centroid, as Vic3 pixel coordinates."""
        r = self.row_of[i]
        flat = self.pixel_order[self.pixel_starts[i]:self.pixel_starts[i + 1]]
        ys, xs = np.divmod(flat, self.v3.width)
        k = np.argmin(wrap_dist(xs, ys, r["vic3_x"], r["vic3_y"], self.v3.width))
        return float(xs[k]), float(ys[k])

    # ---------------------------------------------------------------- build ----

    def build(self):
        by_state = defaultdict(list)
        for i, r in self.row_of.items():
            by_state[r["aoe_state"]].append(i)

        taken, states = set(), []
        for aoe_key, aoe_provs in self.aoe_regions:
            provs = sorted(by_state.get(aoe_key, []), key=lambda i: self.v3.provinces[i])
            if not provs:
                continue
            present = {self.row_of[i]["aoe_province"] for i in provs}
            capital = next((p for p in aoe_provs if p in present), next(iter(present)))
            name = self.loc.get(aoe_key) or self.region_names.get(aoe_key) or self.province_name(capital)
            states.append(dict(aoe=aoe_key, key=make_key(name, taken), name=name, provs=provs,
                               capital=capital, aoe_provs=aoe_provs))
        for n, s in enumerate(states, start=1):
            s["id"] = n

        # Overlap with vanilla states, in pixels.
        area_v = Counter()
        for s in states:
            s["overlap"] = Counter()
            for i in s["provs"]:
                r = self.row_of[i]
                s["overlap"][r["vanilla_vic3_state"]] += r["pixels"]
                area_v[r["vanilla_vic3_state"]] += r["pixels"]
            s["area"] = sum(s["overlap"].values())
            s["main_vanilla"] = s["overlap"].most_common(1)[0][0]
            s["coastal"] = any(self.sea_borders.get(i) for i in s["provs"])
        self.area_v = area_v

        self.assign_resources(states)
        print("choosing hubs")
        for s in states:
            self.choose_hubs(s)
        for w in self.warnings:
            print("warning:", w)
        self.assign_strategic_regions(states)
        return states

    def assign_resources(self, states):
        by_vanilla = defaultdict(list)
        for s in states:
            for v in s["overlap"]:
                by_vanilla[v].append(s)
        for s in states:
            s.update(arable_land=0, capped=Counter(), resources={}, traits=[], arable_resources=[])

        for v, members in by_vanilla.items():
            block = self.land[v]
            area_w = {s["id"]: s["overlap"][v] for s in members}
            coast_w = {s["id"]: sum(length for i in s["provs"] if self.row_of[i]["vanilla_vic3_state"] == v
                                    for _, length, _ in self.sea_borders.get(i, ()))
                       for s in members}
            sid = {s["id"]: s for s in members}

            for k, n in apportion(int(block.get("arable_land", 0)), area_w).items():
                sid[k]["arable_land"] += n
            for building, _, amount in block.get("capped_resources", pdx.Block()).pairs():
                weights = coast_w if building in COASTAL_BUILDINGS else area_w
                for k, n in apportion(int(amount), weights).items():
                    sid[k]["capped"][building] += n
            for res in block.get_all("resource"):
                rtype = res.get("type")
                weights = coast_w if rtype in COASTAL_BUILDINGS else area_w
                for field in AMOUNT_FIELDS:
                    for k, n in apportion(int(res.get(field, 0)), weights).items():
                        entry = sid[k]["resources"].setdefault(rtype, {"extra": [(f, o, val) for f, o, val in res.pairs()
                                                                                   if f not in AMOUNT_FIELDS and f != "type"]})
                        entry[field] = entry.get(field, 0) + n

        for s in states:
            ranked = [v for v, _ in s["overlap"].most_common()]
            main = self.land[ranked[0]]
            s["subsistence"] = main.get("subsistence_building")
            for v in ranked:
                share_s, share_v = s["overlap"][v] / s["area"], s["overlap"][v] / self.area_v[v]
                block = self.land[v]
                if v == ranked[0] or share_s >= 0.2:
                    for b in block.get("arable_resources", pdx.Block()).values():
                        if b not in s["arable_resources"]:
                            s["arable_resources"].append(b)
                if share_s >= 0.3 or share_v >= 0.5:
                    for t in block.get("traits", pdx.Block()).values():
                        if t not in s["traits"] and (s["coastal"] or not COASTAL_TRAIT.search(t)):
                            s["traits"].append(t)
            s["impassable"] = [i for i in s["provs"] if i in self.impassable]
            s["prime_land"] = [i for i in s["provs"] if i in self.prime_land]

    def aoe_town_point(self, aoe_prov, provs):
        """(province, x, y): where the warp puts AoE's city position of aoe_prov, searched in the provinces of
        provs that belong to aoe_prov (all of provs if none does). None if AoE has no position for it."""
        if aoe_prov not in self.positions or not provs:
            return None
        own = [i for i in provs if self.row_of[i]["aoe_province"] == aoe_prov] or list(provs)
        flat = np.concatenate([self.pixel_order[self.pixel_starts[i]:self.pixel_starts[i + 1]] for i in own])
        ys, xs = np.divmod(flat, self.v3.width)
        x2, y2 = self.warp(xs.astype(float), ys.astype(float))
        tx, ty = self.positions[aoe_prov]
        w2 = self.warp.w2
        k = int(np.argmin(np.hypot((x2 - tx + w2 / 2) % w2 - w2 / 2, y2 - ty)))
        return int(self.v3.labels[ys[k], xs[k]]), float(xs[k]), float(ys[k])

    def vanilla_point(self, vkey, t, provs):
        """(province, x, y, rotation) of vanilla's locator for that hub, if it lies in one of provs."""
        loc = self.vanilla_locators.get((vkey, t))
        if not loc:
            return None
        x, z, rot = loc
        px, py = int(x) % self.v3.width, min(max(int(self.v3.height - z), 0), self.v3.height - 1)
        i = int(self.v3.labels[py, px])
        return (i, x - 0.5, self.v3.height - z - 0.5, rot) if i in provs else None

    def gazetteer(self, s, provs):
        """{place key: (province, x, y, rotation, hub type)} for places this state's provinces hold: vanilla hubs
        (any vanilla state) whose locator lies in provs, then AoE towns, under every name they go by."""
        out = {}
        for (vkey, t), name in self.vanilla_hub_names.items():
            pt = self.vanilla_point(vkey, t, provs)
            if pt:
                for n in hub_names.names_of(name):
                    out.setdefault(place_key(n), (*pt, t))
        for p in {self.row_of[i]["aoe_province"] for i in provs}:
            for n in hub_names.names_of(self.province_name(p)):
                k = place_key(n)
                if k not in out:
                    pt = self.aoe_town_point(p, provs)
                    if pt:
                        out[k] = (*pt, None, None)
        return out

    def set_port(self, s, i, vanilla_state=None):
        """Make province i the port, with vanilla_state's naval exit if given, else the sea it borders most."""
        borders = self.sea_borders[i]
        if vanilla_state:
            s["naval_exit"] = int(self.land[vanilla_state].get("naval_exit_id"))
            exit_key = next((k for k, b in self.sea.items() if int(b.get("id")) == s["naval_exit"]), None)
            borders = [b for b in borders if b[0] == exit_key] or borders
        else:
            s["naval_exit"] = int(self.sea[max(borders, key=lambda b: b[1])[0]].get("id"))
        s["hubs"]["port"] = i
        s["hub_pos"]["port"] = max(borders, key=lambda b: b[1])[2]

    def choose_hubs(self, s):
        """Hub provinces, names and locator positions. A hub at a vanilla hub keeps vanilla's name for it (in its
        1700-1821 form) and vanilla's locator; any other hub is an AoE town, placed where the warp puts AoE's city
        position. hub_names.HUB_OVERRIDES then renames and moves hubs by hand."""
        provs = set(s["provs"])
        passable = [i for i in s["provs"] if i not in self.impassable] or s["provs"]
        coastal = [i for i in s["provs"] if self.sea_borders.get(i)]
        ranked = [v for v, _ in s["overlap"].most_common()]
        vanilla_hub = {t: [(v, norm_hex(self.land[v].get(t))) for v in ranked if self.land[v].get(t)
                           and self.index.get(norm_hex(self.land[v].get(t))) in provs]
                       for t in HUB_TYPES}
        order = {p: n for n, p in enumerate(s["aoe_provs"])}
        towns = sorted({self.row_of[i]["aoe_province"] for i in passable},
                       key=lambda p: (-self.population.get(p, 0), order.get(p, len(order))))
        gaz = self.gazetteer(s, set(passable))
        s["hubs"], s["hub_pos"], s["hub_rot"], s["hub_names"] = {}, {}, {}, {}
        hubs, pos, names = s["hubs"], s["hub_pos"], s["hub_names"]
        used, used_towns = set(), set()

        def put(t, name, pt):
            hubs[t], pos[t], names[t] = pt[0], (pt[1], pt[2]), name
            if len(pt) > 3 and pt[3]:
                s["hub_rot"][t] = pt[3]
            used.add(pt[0])

        def town(t, candidates):
            """The most populous AoE town not yet named, placed in candidates; None if there is none."""
            for p in towns:
                if p in used_towns or hub_names.period_name(self.province_name(p)) in names.values():
                    continue
                pt = self.aoe_town_point(p, [i for i in candidates if self.row_of[i]["aoe_province"] == p])
                if pt:
                    used_towns.add(p)
                    return hub_names.period_name(self.province_name(p)), pt
            return None

        # City: AoE's first-listed province of the state (usually its main town), at vanilla's locator for that
        # town if it has one. Other hubs without a vanilla hub take the most populous towns left.
        main = s["capital"]
        name = hub_names.period_name(self.province_name(main))
        pt = gaz.get(place_key(name)) or self.aoe_town_point(main, passable) or \
            (passable[0], *self.inside_point(passable[0]))
        used_towns.add(main)
        put("city", name, pt)

        for t in ("farm", "mine", "wood"):
            pick = next(((v, self.index[h]) for v, h in vanilla_hub[t] if self.index[h] not in used), None)
            vname = hub_names.period_name(self.vanilla_hub_names.get((pick[0], t), "")) if pick else ""
            if pick and vname and vname not in names.values():
                put(t, vname, self.vanilla_point(pick[0], t, provs) or (pick[1], *self.inside_point(pick[1])))
                continue
            found = town(t, [i for i in passable if i not in used]) or town(t, passable)
            if found:
                put(t, *found)
            else:
                spare = [i for i in passable if i not in used] or [hubs["city"]]
                i = max(spare, key=lambda i: self.row_of[i]["pixels"])
                put(t, hub_names.period_name(self.province_name(self.row_of[i]["aoe_province"])),
                    (i, *self.inside_point(i)))

        if coastal:
            pick = next(((v, self.index[h]) for v, h in vanilla_hub["port"] if self.index[h] in coastal), None)
            vname = hub_names.period_name(self.vanilla_hub_names.get((pick[0], "port"), "")) if pick else ""
            if pick:
                self.set_port(s, pick[1], pick[0])
                pt = self.vanilla_point(pick[0], "port", provs)
                if pt and pt[0] == pick[1]:
                    pos["port"], s["hub_rot"]["port"] = (pt[1], pt[2]), pt[3]
                names["port"] = vname if vname and vname not in names.values() else hub_names.period_name(
                    self.province_name(self.row_of[pick[1]]["aoe_province"]))
            else:
                found = town("port", coastal)
                c = pos["city"]
                i = found[1][0] if found else min(coastal, key=lambda i: wrap_dist(
                    self.row_of[i]["vic3_x"], self.row_of[i]["vic3_y"], c[0], c[1], self.v3.width))
                self.set_port(s, i)
                names["port"] = found[0] if found else hub_names.period_name(
                    self.province_name(self.row_of[i]["aoe_province"]))

        self.apply_overrides(s, gaz, passable, coastal)
        dups = [n for n, c in Counter(place_key(n) for n in names.values()).items() if c > 1]
        if dups and len(towns) > 1:
            self.warnings.append(f"{s['aoe']} {s['key']}: hub names repeat: "
                                 + ", ".join(f"{t}={n}" for t, n in names.items()))

    def apply_overrides(self, s, gaz, passable, coastal):
        """hub_names.HUB_OVERRIDES: (AoE state, hub type) -> name, or (name, place). The hub moves to place (an AoE
        province id, another name to look up, or (name, name, f): f of the way from one to the other) or else to
        the place its new name is found at in this state."""
        for t in HUB_TYPES:
            spec = hub_names.HUB_OVERRIDES.get((s["aoe"], t))
            if spec is None or (t == "port" and not coastal):
                if spec is not None:
                    self.warnings.append(f"{s['aoe']}: port override for an inland state")
                continue
            name, place = (spec, None) if isinstance(spec, str) else spec
            pool = coastal if t == "port" else passable
            if isinstance(place, int):
                own = [i for i in pool if self.row_of[i]["aoe_province"] == place]
                pt = self.aoe_town_point(place, own)
                # several hubs in one AoE province: the city keeps the town, the others take its other provinces
                others = {i for u, i in s["hubs"].items() if u != t}
                free = [i for i in own if i not in others]
                if pt and pt[0] in others and free and t not in ("city", "port"):
                    i = max(free, key=lambda i: self.row_of[i]["pixels"])
                    pt = (i, *self.inside_point(i))
                pt = pt and (*pt, None, None)
            elif isinstance(place, tuple):
                a, b, f = gaz.get(place_key(place[0])), gaz.get(place_key(place[1])), place[2]
                pt = None
                if a and b:
                    x, y = a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])
                    i = int(self.v3.labels[int(y), int(x)])
                    pt = (i, x, y, None, None) if i in pool else None
            else:
                pt = gaz.get(place_key(place or name))
            s["hub_names"][t] = name
            if not pt:
                if place is not None:
                    self.warnings.append(f"{s['aoe']} {t}: place {place!r} not found in the state")
                continue
            if t == "port":
                if pt[0] not in coastal:
                    self.warnings.append(f"{s['aoe']} port: {place or name} is not coastal")
                    continue
                self.set_port(s, pt[0])
                s["hub_rot"].pop("port", None)
                if pt[4] == "port":
                    s["hub_pos"]["port"], s["hub_rot"]["port"] = (pt[1], pt[2]), pt[3]
            else:
                s["hubs"][t], s["hub_pos"][t] = pt[0], (pt[1], pt[2])
                s["hub_rot"].pop(t, None)
                if pt[3] and pt[4] == t:
                    s["hub_rot"][t] = pt[3]

    def assign_strategic_regions(self, states):
        for s in states:
            votes = Counter()
            for v, px in s["overlap"].items():
                votes[self.vanilla_region.get(v)] += px
            votes.pop(None, None)
            s["region"] = votes.most_common(1)[0][0]

    # ---------------------------------------------------------------- write ----

    def write(self, states):
        mod = config.MOD
        sr_dir = mod / "map_data" / "state_regions"
        sr_dir.mkdir(parents=True, exist_ok=True)
        land_files = sorted({f for k, f in self.file_of.items() if k in self.land})
        per_file = defaultdict(list)
        for s in states:
            per_file[self.file_of[s["main_vanilla"]]].append(s)
        for fname in land_files:
            text = "\n".join(self.format_state(s) for s in per_file.get(fname, []))
            (sr_dir / fname).write_text(text, encoding="utf-8-sig")

        self.write_strategic_regions(states)
        self.write_geographic_regions(states)
        self.write_locators(states)
        self.write_localisation(states)

        with open(config.BUILD / "state_map.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["aoe_state", "vic3_state", "id", "name", "file", "strategic_region", "capital_aoe_province",
                        "provinces", "coastal", "naval_exit", *HUB_TYPES])
            for s in states:
                w.writerow([s["aoe"], s["key"], s["id"], s["name"], self.file_of[s["main_vanilla"]], s["region"],
                            s["capital"], len(s["provs"]), int(s["coastal"]), s.get("naval_exit", ""),
                            *[self.v3.provinces[s["hubs"][t]] if t in s["hubs"] else "" for t in HUB_TYPES]])

    def format_state(self, s):
        p = self.v3.provinces
        q = lambda ids: " ".join(f'"{p[i]}"' for i in ids)
        lines = [f"{s['key']} = {{", f"    id = {s['id']}"]
        if s["subsistence"]:
            lines.append(f'    subsistence_building = "{s["subsistence"]}"')
        lines.append(f"    provinces = {{ {q(s['provs'])} }}")
        if s["impassable"]:
            lines.append(f"    impassable = {{ {q(s['impassable'])} }}")
        if s["prime_land"]:
            lines.append(f"    prime_land = {{ {q(s['prime_land'])} }}")
        if s["traits"]:
            lines.append("    traits = { " + " ".join(f'"{t}"' for t in s["traits"]) + " }")
        for t in HUB_TYPES:
            if t in s["hubs"]:
                lines.append(f'    {t} = "{p[s["hubs"][t]]}"')
        lines.append(f"    arable_land = {s['arable_land']}")
        lines.append("    arable_resources = { " + " ".join(f'"{b}"' for b in s["arable_resources"]) + " }")
        if s["capped"]:
            lines.append("    capped_resources = {")
            lines += [f"        {b} = {n}" for b, n in sorted(s["capped"].items())]
            lines.append("    }")
        for rtype, entry in sorted(s["resources"].items()):
            lines += ["    resource = {", f'        type = "{rtype}"']
            lines += [f"        {f} = {pdx.format_scalar(v)}" for f, _, v in entry["extra"]]
            lines += [f"        {f} = {entry[f]}" for f in AMOUNT_FIELDS if entry.get(f)]
            lines.append("    }")
        if s["coastal"]:
            lines.append(f"    naval_exit_id = {s['naval_exit']}")
        lines.append("}\n")
        return "\n".join(lines)

    def write_strategic_regions(self, states):
        out_dir = config.MOD / "common" / "strategic_regions"
        out_dir.mkdir(parents=True, exist_ok=True)
        members = defaultdict(list)
        for s in states:
            members[s["region"]].append(s)
        city_of = {}
        for s in states:
            city_of[self.v3.provinces[s["hubs"]["city"]]] = s
        per_file = defaultdict(list)
        for rkey, (fname, block) in self.regions.items():
            if fname.startswith("water"):
                continue
            per_file[fname]  # keep the file even if all its regions end up empty
            mine = members.get(rkey)
            if not mine:
                continue
            prov_keys = {self.v3.provinces[i] for s in mine for i in s["provs"]}
            capital = block.get("capital_province")
            if not capital or norm_hex(capital) not in prov_keys:
                capital = self.v3.provinces[max(mine, key=lambda s: s["area"])["hubs"]["city"]]
            out = pdx.Block()
            for k, op, v in block.pairs():
                if k == "states":
                    v = pdx.Block(s["key"] for s in mine)
                elif k == "capital_province":
                    v = capital
                out.append((k, op, v))
            per_file[fname].append((rkey, out))
        for fname, regions in per_file.items():
            text = "\n\n".join(f"{rkey} = {{\n{pdx.dumps(block, 1)}\n}}" for rkey, block in regions)
            (out_dir / fname).write_text(text + "\n", encoding="utf-8-sig")

    def translate_states(self, states, vanilla_keys):
        """New states lying mostly (by area) inside the given vanilla states."""
        vanilla_keys = set(vanilla_keys)
        share = {s["key"]: sum(px for v, px in s["overlap"].items() if v in vanilla_keys) / s["area"] for s in states}
        out = [k for k, f in share.items() if f >= 0.5]
        if not out and any(share.values()):
            out = [max(share, key=share.get)]
        return out

    def write_geographic_regions(self, states):
        """Rewrite vanilla geographic regions that list state regions in terms of the new states."""
        out_dir = config.MOD / "common" / "geographic_regions"
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted((config.VIC3 / "common" / "geographic_regions").glob("*.txt")):
            root = pdx.parse_file(path)
            if not any(isinstance(b, pdx.Block) and b.get("state_regions") is not None for _, _, b in root.pairs()):
                continue
            regions = []
            for key, _, block in root.pairs():
                if not isinstance(block, pdx.Block):
                    continue
                out = pdx.Block()
                for k, op, v in block.pairs():
                    if k == "state_regions":
                        v = pdx.Block(self.translate_states(states, v.values()))
                    out.append((k, op, v))
                regions.append(f"{key} = {{\n{pdx.dumps(out, 1)}\n}}")
            (out_dir / path.name).write_text("\n\n".join(regions) + "\n", encoding="utf-8-sig")

    def write_locators(self, states):
        out_dir = config.MOD / "gfx" / "map" / "map_object_data"
        out_dir.mkdir(parents=True, exist_ok=True)
        h = self.v3.height
        for t in HUB_TYPES:
            vanilla = pdx.read_text(config.VIC3 / "gfx" / "map" / "map_object_data" / f"generated_map_object_locators_{t}.txt")
            header = vanilla[:vanilla.index("instances")]
            lines = [header.rstrip() + "\n\tinstances={"]
            for s in states:
                if t not in s["hubs"]:
                    continue
                x, y = s["hub_pos"][t]
                rot = s["hub_rot"].get(t, "-0.000000 0.000000 -0.000000 1.000000")
                lines += ["\t\t{", f"\t\t\tid={s['id']}",
                          f"\t\t\tposition={{ {x + 0.5:.6f} 0.000000 {h - (y + 0.5):.6f} }}",
                          f"\t\t\trotation={{ {rot} }}",
                          "\t\t\tscale={ 1.000000 1.000000 1.000000 }", "\t\t}"]
            lines += ["\t}", "}"]
            (out_dir / f"generated_map_object_locators_{t}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    def write_localisation(self, states):
        def esc(text):
            return text.replace('"', '\\"')
        for lang in LANGUAGES:
            out_dir = config.MOD / "localization" / lang / "replace"
            out_dir.mkdir(parents=True, exist_ok=True)
            lines = [f"l_{lang}:"]
            for s in states:
                lines.append(f' {s["key"]}:0 "{esc(s["name"])}"')
                for t, n in s["hub_names"].items():
                    lines.append(f' HUB_NAME_{s["key"]}_{t}:0 "{esc(n)}"')
            (out_dir / f"aoe_states_l_{lang}.yml").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main():
    builder = StateBuilder()
    states = builder.build()
    builder.write(states)
    coastal = sum(s["coastal"] for s in states)
    print(f"wrote {len(states)} states ({coastal} coastal) into {config.MOD}")
    totals = Counter()
    for s in states:
        totals["arable_land"] += s["arable_land"]
    vanilla_arable = sum(int(b.get("arable_land", 0)) for b in builder.land.values())
    print(f"arable land: {totals['arable_land']} (vanilla {vanilla_arable})")
    regions = Counter(s["region"] for s in states)
    print(f"strategic regions used: {len(regions)}")


if __name__ == "__main__":
    main()
