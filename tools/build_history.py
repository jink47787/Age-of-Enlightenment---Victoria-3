"""Convert AoE's 1700 setup (countries, cultures, religions, ownership, pops) to Vic3.

Needs build/province_map.csv and build/state_map.csv (run build_province_map.py
and build_states.py first). Writes into mod/:

  common/country_definitions/aoe_countries.txt
  common/cultures/aoe_cultures.txt            cultures Vic3 doesn't already have
  common/religions/aoe_religions.txt          religions Vic3 doesn't already have
  common/history/countries/aoe_countries.txt  laws, technology
  common/history/population/aoe_population.txt
  common/history/states/aoe_states.txt        ownership, homelands, claims
  common/history/pops/aoe_pops.txt
  common/history/buildings/aoe_buildings.txt  capital administration and barracks
  common/history/diplomacy/aoe_diplomacy.txt  subjects and alliances
  localization/<lang>/aoe_countries_l_<lang>.yml

Unowned AoE land becomes decentralized native nations, grouped by the
dominant culture of their pops.
"""
import csv
import math
from collections import Counter, defaultdict

import numpy as np

import config
import pdx
from aoedata import read_localisation, read_province_map
from aoehistory import read_countries, read_cultures, read_diplomacy, read_pops, read_provinces, read_religions
from mapdata import province_borders, read_definition, read_vic3_map

POP_SCALE = 4            # Vic2 pop sizes count roughly a quarter of the real population
NATIVE_MAX_STATES = 4    # largest native nation, in AoE states
LANGUAGES = ["english", "braz_por", "french", "german", "japanese", "korean", "polish", "russian",
             "simp_chinese", "spanish", "turkish"]

# AoE religion -> existing Vic3 religion, where one fits.
RELIGION_MAP = {"coptic": "oriental_orthodox"}
# Template for each new religion (icon and heritage).
RELIGION_TEMPLATE = {"mormon": "protestant", "druze": "shiite", "yazidi": "sunni", "taoist": "confucian",
                     "tengri": "animist", "zoroastrian": "shiite"}

# AoE government -> (governance law, distribution of power law, vanilla politics effect)
GOVERNMENTS = {
    "absolute_monarchy": ("law_monarchy", "law_autocracy", "traditional"),
    "khanate": ("law_monarchy", "law_autocracy", "traditional"),
    "theocracy": ("law_theocracy", "law_autocracy", "traditional"),
    "elective_monarchy_oligarchy": ("law_monarchy", "law_oligarchy", "traditional"),
    "prussian_constitutionalism": ("law_monarchy", "law_landed_voting", "conservative"),
    "hms_government": ("law_monarchy", "law_wealth_voting", "conservative"),
    "oligarchic_republic": ("law_parliamentary_republic", "law_oligarchy", "conservative"),
    "democracy": ("law_parliamentary_republic", "law_wealth_voting", "liberal"),
    "presidential_dictatorship": ("law_presidential_republic", "law_autocracy", "conservative"),
    "bourgeois_dictatorship": ("law_presidential_republic", "law_oligarchy", "conservative"),
    "tribal_democracy": ("law_chiefdom", "law_elder_council", "traditional"),
    "colonial_company": ("law_colonial_administration", "law_oligarchy", "traditional"),
    "dominion": ("law_colonial_administration", "law_oligarchy", "conservative"),
}
VOTING = {"landed_voting": "law_landed_voting", "wealth_voting": "law_wealth_voting",
          "wealth_weighted_voting": "law_wealth_voting", "universal_weighted_voting": "law_census_voting",
          "universal_voting": "law_universal_suffrage"}
PRESS = {"state_press": "law_outlawed_dissent", "censored_press": "law_censorship", "free_press": "law_right_of_assembly"}
COLONIAL = {"settlement": "law_colonial_resettlement", "exploitation": "law_colonial_exploitation",
            "civilizing": "law_frontier_colonization"}
SCHOOLS = {"low_schools": "law_religious_schools", "acceptable_schools": "law_private_schools",
           "good_schools": "law_private_schools"}
SUBJECTS = {"vassal": "vassal", "substate": "puppet", "union": "personal_union"}


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")


def loc_escape(text):
    return text.replace('"', '\\"')


class HistoryBuilder:
    def __init__(self):
        self.rows = read_province_map()
        with open(config.BUILD / "state_map.csv", encoding="utf-8") as f:
            self.state_rows = list(csv.DictReader(f))
        self.vic3_state = {r["aoe_state"]: r["vic3_state"] for r in self.state_rows}
        self.state_name = {r["vic3_state"]: r["name"] for r in self.state_rows}
        self.coastal = {r["vic3_state"]: r["coastal"] == "1" for r in self.state_rows}

        self.provs = read_provinces()
        self.countries = read_countries()
        self.pops = read_pops()
        self.diplomacy = read_diplomacy()
        self.aoe_cultures, _ = read_cultures()
        self.aoe_religions = read_religions()
        self.loc = read_localisation()

        self.v3_cultures = self.read_vic3_blocks("cultures")
        self.v3_religions = self.read_vic3_blocks("religions")

        self.v3_of_aoe = defaultdict(list)       # AoE province -> Vic3 provinces
        self.aoe_state = {}                      # AoE province -> AoE state
        for r in self.rows:
            self.v3_of_aoe[r["aoe_province"]].append(r["vic3_province"])
            self.aoe_state[r["aoe_province"]] = r["aoe_state"]
        self.warnings = []

    @staticmethod
    def read_vic3_blocks(folder):
        out = {}
        for path in sorted((config.VIC3 / "common" / folder).glob("*.txt")):
            for k, _, b in pdx.parse_file(path).pairs():
                if isinstance(b, pdx.Block):
                    out[k] = b
        return out

    # ------------------------------------------------------- religions ----

    def religion(self, aoe_religion):
        r = RELIGION_MAP.get(aoe_religion, aoe_religion)
        if r in self.v3_religions or r in RELIGION_TEMPLATE:
            return r
        self.warnings.append(f"unknown religion {aoe_religion!r}, using animist")
        return "animist"

    def build_religions(self):
        blocks, loc = [], {}
        for key, template in RELIGION_TEMPLATE.items():
            if key not in self.aoe_religions:
                continue
            _, aoe_block = self.aoe_religions[key]
            out = pdx.Block()
            for k, op, v in self.v3_religions[template].pairs():
                if k == "color":
                    rgb = aoe_block.get("color")
                    if isinstance(rgb, pdx.Block):
                        v = pdx.Block(rgb.values())
                out.append((k, op, v))
            blocks.append(f"{key} = {{\n{pdx.dumps(out, 1)}\n}}")
            loc[key] = self.loc.get(key, key.replace("_", " ").title())
        write_text(config.MOD / "common" / "religions" / "aoe_religions.txt", "\n\n".join(blocks) + "\n")
        return loc

    # -------------------------------------------------------- cultures ----

    def culture(self, aoe_culture):
        if aoe_culture in self.aoe_cultures or aoe_culture in self.v3_cultures:
            return aoe_culture
        return None

    def vanilla_culture_by_state(self):
        """{vanilla Vic3 state: Counter(culture -> pop size)} from vanilla pop history."""
        out = defaultdict(Counter)
        for path in sorted((config.VIC3 / "common" / "history" / "pops").glob("*.txt")):
            for _, _, root in pdx.parse_file(path).pairs():
                if not isinstance(root, pdx.Block):
                    continue
                for skey, _, sblock in root.pairs():
                    if not isinstance(sblock, pdx.Block):
                        continue
                    for _, _, region_state in sblock.pairs():
                        if isinstance(region_state, pdx.Block):
                            for pop in region_state.get_all("create_pop"):
                                if pop.get("culture"):
                                    out[skey.removeprefix("s:")][pop.get("culture")] += int(float(pop.get("size", 0)))
        return out

    def build_cultures(self):
        """Create Vic3 cultures for AoE cultures Vic3 lacks, copying the vanilla culture
        that lives where AoE puts that culture's pops."""
        prov_vanilla = defaultdict(Counter)
        for r in self.rows:
            prov_vanilla[r["aoe_province"]][r["vanilla_vic3_state"]] += r["pixels"]
        by_state = self.vanilla_culture_by_state()

        culture_pops = defaultdict(Counter)   # AoE culture -> Counter(vanilla Vic3 culture)
        culture_religion = defaultdict(Counter)
        for prov, pops in self.pops.items():
            states = prov_vanilla.get(prov)
            for pop in pops:
                culture_religion[pop.culture][pop.religion] += pop.size
                if not states:
                    continue
                total = sum(states.values())
                for vstate, px in states.items():
                    vc = by_state.get(vstate)
                    if not vc:
                        continue
                    vtotal = sum(vc.values())
                    for vculture, n in vc.items():
                        culture_pops[pop.culture][vculture] += pop.size * (px / total) * (n / vtotal)

        blocks, loc, templates = [], {}, {}
        for key, (group, aoe_block) in sorted(self.aoe_cultures.items()):
            if key in self.v3_cultures:
                continue
            candidates = [c for c, _ in culture_pops[key].most_common() if c in self.v3_cultures]
            if not candidates:
                # No pops on the map: borrow from another culture of the same AoE group.
                siblings = [c for c, (g, _) in self.aoe_cultures.items() if g == group and c in self.v3_cultures]
                candidates = siblings or ["british"]
            template = candidates[0]
            templates[key] = template
            religion = culture_religion[key].most_common(1)[0][0] if culture_religion[key] else None
            out = pdx.Block()
            for k, op, v in self.v3_cultures[template].pairs():
                if k == "color":
                    rgb = aoe_block.get("color")
                    v = pdx.Block(rgb.values()) if isinstance(rgb, pdx.Block) else v
                elif k == "religion" and religion:
                    v = self.religion(religion)
                out.append((k, op, v))
            blocks.append(f"{key} = {{\n{pdx.dumps(out, 1)}\n}}")
            loc[key] = self.loc.get(key, key.replace("_", " ").title())
        write_text(config.MOD / "common" / "cultures" / "aoe_cultures.txt",
                   "# Generated by tools/build_history.py: AoE cultures missing from Vic3, each copied from\n"
                   "# the vanilla culture living where AoE places it.\n\n" + "\n\n".join(blocks) + "\n")
        with open(config.BUILD / "culture_templates.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["aoe_culture", "vic3_template"])
            w.writerows(sorted(templates.items()))
        return loc

    # --------------------------------------------------------- natives ----

    def make_natives(self):
        """Group unowned AoE land into decentralized nations by dominant culture."""
        unowned = [p for p in self.v3_of_aoe if not (self.provs.get(p) and self.provs[p].owner)]
        units = defaultdict(list)
        for p in unowned:
            units[self.aoe_state[p]].append(p)

        def top_culture(provs):
            c = Counter()
            for p in provs:
                for pop in self.pops.get(p, []):
                    if self.culture(pop.culture):
                        c[pop.culture] += pop.size
            return c.most_common(1)[0][0] if c else None

        culture_of = {s: top_culture(ps) for s, ps in units.items()}

        # Adjacency between AoE states, through Vic3 province borders.
        v3 = read_vic3_map(config.VIC3, config.REDRAW)
        state_of_v3 = {}
        index = {p: i for i, p in enumerate(v3.provinces)}
        for r in self.rows:
            state_of_v3[index[r["vic3_province"]]] = r["aoe_state"]
        adj = defaultdict(set)
        for a, b in province_borders(v3.labels):
            sa, sb = state_of_v3.get(a), state_of_v3.get(b)
            if sa and sb and sa != sb:
                adj[sa].add(sb)
                adj[sb].add(sa)
        self.v3_map = v3

        # Units without pops take the culture of the nearest unit that has one.
        fallback = Counter(c for c in culture_of.values() if c).most_common(1)
        for s in sorted(u for u, c in culture_of.items() if c is None):
            seen, frontier = {s}, [s]
            found = None
            while frontier and not found:
                nxt = []
                for x in frontier:
                    for y in sorted(adj[x]):
                        if y in seen:
                            continue
                        seen.add(y)
                        if culture_of.get(y):
                            found = culture_of[y]
                            break
                        nxt.append(y)
                    if found:
                        break
                frontier = nxt
            culture_of[s] = found or (fallback[0][0] if fallback else "native_american_minor")

        # Merge adjacent units of the same culture, up to NATIVE_MAX_STATES each.
        nations, assigned = [], {}
        for s in sorted(units, key=lambda s: -sum(x.size for p in units[s] for x in self.pops.get(p, []))):
            if s in assigned:
                continue
            group, frontier = [s], [s]
            assigned[s] = len(nations)
            while frontier and len(group) < NATIVE_MAX_STATES:
                x = frontier.pop(0)
                for y in sorted(adj[x]):
                    if y in units and y not in assigned and culture_of[y] == culture_of[s] and len(group) < NATIVE_MAX_STATES:
                        assigned[y] = len(nations)
                        group.append(y)
                        frontier.append(y)
            nations.append(group)

        taken = set(self.countries)
        alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        tags = (f"N{a}{b}" for a in alphabet for b in alphabet)
        culture_count = Counter(culture_of[g[0]] for g in nations)
        self.natives = {}
        for group in nations:
            tag = next(t for t in tags if t not in taken)
            taken.add(tag)
            culture = culture_of[group[0]]
            cname = self.loc.get(culture, culture.replace("_", " ").title())
            capital_state = group[0]
            name = cname if culture_count[culture] == 1 else \
                f"{cname} of {self.state_name.get(self.vic3_state[capital_state], capital_state)}"
            religion = Counter(pop.religion for s in group for p in units[s] for pop in self.pops.get(p, [])
                               if pop.culture == culture).most_common(1)
            self.natives[tag] = dict(culture=culture, name=name, adjective=cname, states=group,
                                     religion=self.religion(religion[0][0]) if religion else None)
            for s in group:
                for p in units[s]:
                    self.native_owner[p] = tag
        print(f"created {len(self.natives)} native nations for {len(unowned)} unowned AoE provinces")

    # ------------------------------------------------------- ownership ----

    def owner(self, aoe_prov):
        p = self.provs.get(aoe_prov)
        return p.owner if p and p.owner else self.native_owner.get(aoe_prov)

    def build_states_and_pops(self):
        """Per Vic3 state: {owner: [Vic3 provinces]}, and pops per (state, owner)."""
        parts = defaultdict(lambda: defaultdict(list))
        colonial = defaultdict(lambda: defaultdict(list))
        for aoe_p, v3_provs in self.v3_of_aoe.items():
            state = self.vic3_state[self.aoe_state[aoe_p]]
            owner = self.owner(aoe_p)
            parts[state][owner] += v3_provs
            colonial[state][owner].append(self.provs[aoe_p].colonial if aoe_p in self.provs else 0)

        # AoE provinces too small to get a Vic3 province: send their pops to the
        # Vic3 province covering most of their pixels.
        target = {p: (self.vic3_state[self.aoe_state[p]], self.owner(p)) for p in self.v3_of_aoe}
        lost = [p for p in self.pops if p not in target]
        if lost:
            aoe_at = np.load(config.BUILD / "aoe_at_vic3.npy")
            labels = self.v3_map.labels
            owner_of_v3 = {}
            index = {p: i for i, p in enumerate(self.v3_map.provinces)}
            for r in self.rows:
                owner_of_v3[index[r["vic3_province"]]] = (self.vic3_state[r["aoe_state"]], self.owner(r["aoe_province"]))
            lost_set = np.zeros(int(aoe_at.max()) + 1, bool)
            lost_set[[p for p in lost if p < len(lost_set)]] = True
            sel = lost_set[aoe_at]
            keys, counts = np.unique(aoe_at[sel].astype(np.int64) * 100000 + labels[sel], return_counts=True)
            best = {}
            for k, n in zip(keys, counts):
                a, v = int(k // 100000), int(k % 100000)
                if v in owner_of_v3 and n > best.get(a, (0, None))[0]:
                    best[a] = (n, owner_of_v3[v])
            # Failing that (an island the warp puts in open sea), to their own state and owner's land, if it has
            # any. Provinces missing from AoE's definition.csv aren't on its map, so the game never uses their pops.
            on_map = read_definition(config.AOE / "map" / "definition.csv")[1]
            state_of = {int(v): s for s, _, provs in pdx.parse_file(config.AOE / "map" / "region.txt").pairs()
                        if isinstance(provs, pdx.Block) for v in provs.values() if v.isdigit()}
            native_of = {s: tag for tag, n in self.natives.items() for s in n["states"]}
            for p in lost:
                if p in best:
                    target[p] = best[p][1]
                elif p in on_map and state_of.get(p) in self.vic3_state:
                    key = (self.vic3_state[state_of[p]], self.owner(p) or native_of.get(state_of[p]))
                    if key[1] in parts.get(key[0], {}):
                        target[p] = key
            missing = [p for p in lost if p not in target]
            if missing:
                self.warnings.append(f"pops of {len(missing)} AoE provinces dropped (not on AoE's map, or no Vic3 "
                                     f"land nearby or in their state under their owner): {missing[:10]}")

        pops = defaultdict(Counter)  # (state, owner) -> Counter((culture, religion, slave) -> size)
        for p, plist in self.pops.items():
            if p not in target:
                continue
            for pop in plist:
                culture = self.culture(pop.culture)
                if not culture:
                    self.warnings.append(f"province {p}: unknown culture {pop.culture!r}")
                    continue
                pops[target[p]][(culture, self.religion(pop.religion), pop.type == "slaves")] += pop.size * POP_SCALE
        self.parts, self.colonial, self.state_pops = parts, colonial, pops

    # ------------------------------------------------------- countries ----

    def country_list(self):
        """Every tag that owns land (AoE countries and natives)."""
        owners = Counter()
        for state, by_owner in self.parts.items():
            for owner in by_owner:
                owners[owner] += 1
        return owners

    def capital_state(self, tag):
        c = self.countries.get(tag)
        owned = {s for s, by_owner in self.parts.items() if tag in by_owner}
        if c and c.capital in self.aoe_state:
            s = self.vic3_state[self.aoe_state[c.capital]]
            if not owned or s in owned:
                return s
        if tag in self.natives:
            return self.vic3_state[self.natives[tag]["states"][0]]
        if owned:
            return max(owned, key=lambda s: sum(self.state_pops[(s, tag)].values()))
        if c and c.capital in self.aoe_state:
            return self.vic3_state[self.aoe_state[c.capital]]
        return None

    def country_population(self, tag):
        return sum(sum(v.values()) for (s, o), v in self.state_pops.items() if o == tag)

    def tier(self, tag, n_states, population):
        if tag in self.natives:
            return "principality"
        if n_states >= 20 or population >= 20_000_000:
            return "empire"
        if n_states >= 6 or population >= 4_000_000:
            return "kingdom"
        if n_states >= 3 or population >= 1_000_000:
            return "grand_principality"
        if n_states == 1 and population < 150_000:
            return "city_state"
        return "principality"

    def country_cultures(self, tag):
        if tag in self.natives:
            return [self.natives[tag]["culture"]]
        c = self.countries[tag]
        out = [x for x in [c.primary_culture, *c.accepted] if x and self.culture(x)]
        if not out:
            # Fall back on the largest culture among the country's pops.
            counts = Counter()
            for (s, o), v in self.state_pops.items():
                if o == tag:
                    for (culture, _, _), n in v.items():
                        counts[culture] += n
            out = [counts.most_common(1)[0][0]] if counts else ["british"]
            self.warnings.append(f"{tag}: no valid primary culture, using {out[0]}")
        return list(dict.fromkeys(out))

    def laws(self, tag):
        c = self.countries[tag]
        governance, power, politics = GOVERNMENTS.get(c.government, ("law_monarchy", "law_autocracy", "traditional"))
        r = c.reforms
        laws = [governance, power]
        if power not in ("law_autocracy", "law_elder_council") and r.get("vote_franschise") in VOTING:
            laws[1] = VOTING[r["vote_franschise"]]
        has_colonies = any(self.provs[p].colonial for p in self.provs if self.provs[p].owner == tag)
        slavery = r.get("slavery")
        if slavery == "yes_slavery":
            laws.append("law_colonial_slavery" if c.civilized and has_colonies else "law_slave_trade")
        elif slavery == "freedom_of_womb":
            laws.append("law_legacy_slavery")
        elif slavery == "no_slavery":
            laws.append("law_slavery_banned")
        if r.get("press_rights") in PRESS:
            laws.append(PRESS[r["press_rights"]])
        if r.get("debt_law") == "serfdom":
            laws.append("law_serfdom")
        elif r.get("debt_law"):
            laws.append("law_tenant_farmers")
        if r.get("colonial_policy") in COLONIAL:
            laws.append(COLONIAL[r["colonial_policy"]])
        elif has_colonies:
            laws.append("law_frontier_colonization")
        if r.get("school_reforms") in SCHOOLS:
            laws.append(SCHOOLS[r["school_reforms"]])
        if c.civilized:
            laws += ["law_mercantilism", "law_professional_army"]
        return politics, laws

    def tech_tier(self, tag, tech_rank):
        if tag in self.natives:
            return 7
        c = self.countries[tag]
        if c.civilized:
            return 3 if tech_rank >= 0.9 else 4
        return 5 if len(c.techs) > 0 else 6

    @staticmethod
    def literacy_effect(literacy):
        for threshold, name in ((0.5, "high"), (0.3, "middling"), (0.15, "baseline"), (0.05, "low")):
            if literacy >= threshold:
                return f"effect_starting_pop_literacy_{name}"
        return "effect_starting_pop_literacy_very_low"

    # ----------------------------------------------------------- write ----

    def write(self, culture_loc, religion_loc):
        owners = self.country_list()
        civ_tech = sorted(len(self.countries[t].techs) for t in owners if t in self.countries and self.countries[t].civilized)

        definitions, histories, population, loc = [], [], [], {}
        for tag in sorted(owners):
            native = self.natives.get(tag)
            c = self.countries.get(tag)
            capital = self.capital_state(tag)
            pop_total = self.country_population(tag)
            if native:
                color = tuple(int(x) for x in np.random.default_rng(sum(map(ord, tag))).integers(90, 200, 3))
                ctype, religion = "decentralized", native["religion"]
            else:
                color = c.color
                ctype = "recognized" if c.civilized else "unrecognized"
                religion = self.religion(c.religion) if c.religion else None
            lines = [f"{tag} = {{", f"\tcolor = {{ {color[0]} {color[1]} {color[2]} }}", f"\tcountry_type = {ctype}",
                     f"\ttier = {self.tier(tag, owners[tag], pop_total)}",
                     f"\tcultures = {{ {' '.join(self.country_cultures(tag))} }}"]
            if religion:
                lines.append(f"\treligion = {religion}")
            if capital:
                lines.append(f"\tcapital = {capital}")
            lines.append("}")
            definitions.append("\n".join(lines))

            if native:
                loc[tag], loc[f"{tag}_ADJ"] = native["name"], native["adjective"]
                histories.append(f"\tc:{tag} ?= {{\n\t\teffect_starting_technology_tier_7_tech = yes\n"
                                 f"\t\teffect_starting_politics_traditional = yes\n\t}}")
                population.append(f"\tc:{tag} ?= {{\n\t\teffect_starting_pop_wealth_low = yes\n"
                                   f"\t\teffect_starting_pop_literacy_very_low = yes\n\t}}")
                continue

            loc[tag] = self.loc.get(tag, tag)
            loc[f"{tag}_ADJ"] = self.loc.get(f"{tag}_ADJ", loc[tag])
            rank = (sum(1 for x in civ_tech if x <= len(c.techs)) / len(civ_tech)) if civ_tech else 0
            politics, laws = self.laws(tag)
            body = [f"effect_starting_technology_tier_{self.tech_tier(tag, rank)}_tech = yes",
                    f"effect_starting_politics_{politics} = yes"]
            body += [f"activate_law = law_type:{law}" for law in laws]
            histories.append(f"\tc:{tag} ?= {{\n" + "\n".join(f"\t\t{x}" for x in body) + "\n\t}")
            wealth = "medium" if c.civilized and c.prestige >= 50 else "low"
            population.append(f"\tc:{tag} ?= {{\n\t\teffect_starting_pop_wealth_{wealth} = yes\n"
                              f"\t\t{self.literacy_effect(c.literacy)} = yes\n\t}}")

        mod = config.MOD
        write_text(mod / "common" / "country_definitions" / "aoe_countries.txt", "\n\n".join(definitions) + "\n")
        write_text(mod / "common" / "history" / "countries" / "aoe_countries.txt",
                   "COUNTRIES = {\n" + "\n".join(histories) + "\n}\n")
        write_text(mod / "common" / "history" / "population" / "aoe_population.txt",
                   "POPULATION = {\n" + "\n".join(population) + "\n}\n")
        self.write_states()
        self.write_pops()
        self.write_buildings(owners)
        self.write_diplomacy(owners)

        all_loc = {**loc, **culture_loc, **religion_loc}
        for lang in LANGUAGES:
            body = "\n".join(f' {k}:0 "{loc_escape(v)}"' for k, v in all_loc.items())
            write_text(mod / "localization" / lang / f"aoe_countries_l_{lang}.yml", f"l_{lang}:\n{body}\n")
        return owners

    def write_states(self):
        out = ["STATES = {"]
        for state in sorted(self.parts):
            out.append(f"\ts:{state} = {{")
            for owner, provs in sorted(self.parts[state].items()):
                incorporated = owner in self.natives or not all(x > 0 for x in self.colonial[state][owner])
                out += ["\t\tcreate_state = {", f"\t\t\tcountry = c:{owner}",
                        f"\t\t\towned_provinces = {{ {' '.join(sorted(provs))} }}"]
                if not incorporated:
                    out.append("\t\t\tstate_type = unincorporated")
                out.append("\t\t}")
            # Homelands: every culture with at least 10% of the state's people, and the largest.
            cultures = Counter()
            for owner in self.parts[state]:
                for (culture, _, _), n in self.state_pops[(state, owner)].items():
                    cultures[culture] += n
            total = sum(cultures.values())
            homelands = [c for c, n in cultures.most_common() if n >= 0.1 * total][:4] or \
                        [c for c, _ in cultures.most_common(1)]
            out += [f"\t\tadd_homeland = cu:{c}" for c in homelands]
            # Claims: AoE cores of other existing countries on land they don't own.
            claims = set()
            for aoe_p in {p for p, s in self.aoe_state.items() if self.vic3_state[s] == state}:
                if aoe_p in self.provs:
                    claims |= {t for t in self.provs[aoe_p].cores
                               if t in self.existing and t != self.owner(aoe_p)}
            out += [f"\t\tadd_claim = c:{t}" for t in sorted(claims)]
            out.append("\t}")
        out.append("}")
        write_text(config.MOD / "common" / "history" / "states" / "aoe_states.txt", "\n".join(out) + "\n")

    def write_pops(self):
        by_state = defaultdict(dict)
        for (state, owner), pops in self.state_pops.items():
            by_state[state][owner] = pops
        out = ["POPS = {"]
        for state in sorted(by_state):
            out.append(f"\ts:{state} = {{")
            for owner, pops in sorted(by_state[state].items()):
                out.append(f"\t\tregion_state:{owner} = {{")
                for (culture, religion, slave), size in sorted(pops.items(), key=lambda x: -x[1]):
                    fields = [f"culture = {culture}", f"religion = {religion}", f"size = {int(size)}"]
                    if slave:
                        fields.insert(0, "pop_type = slaves")
                    out.append("\t\t\tcreate_pop = { " + " ".join(fields) + " }")
                out.append("\t\t}")
            out.append("\t}")
        out.append("}")
        write_text(config.MOD / "common" / "history" / "pops" / "aoe_pops.txt", "\n".join(out) + "\n")

    def write_buildings(self, owners):
        """Minimal starting buildings: administration and barracks in each capital."""
        out = ["BUILDINGS = {"]
        for tag in sorted(owners):
            if tag in self.natives:
                continue
            capital = self.capital_state(tag)
            if not capital or tag not in self.parts.get(capital, {}):
                continue
            population = self.country_population(tag)
            admin = max(1, min(10, round(population / 3_000_000)))
            barracks = max(1, min(30, round(population / 1_000_000)))
            out += [f"\ts:{capital} = {{", f"\t\tregion_state:{tag} = {{"]
            for building, levels in (("building_government_administration", admin), ("building_barrack", barracks)):
                out += ["\t\t\tcreate_building = {", f'\t\t\t\tbuilding = "{building}"',
                        "\t\t\t\tadd_ownership = {", f'\t\t\t\t\tcountry = {{ country = "c:{tag}" levels = {levels} }}',
                        "\t\t\t\t}", "\t\t\t\treserves = 1", "\t\t\t}"]
            out += ["\t\t}", "\t}"]
        out.append("}")
        write_text(config.MOD / "common" / "history" / "buildings" / "aoe_buildings.txt", "\n".join(out) + "\n")

    def write_diplomacy(self, owners):
        pacts, relations = defaultdict(list), defaultdict(list)
        for kind, first, second in self.diplomacy:
            if first not in owners or second not in owners:
                continue
            if kind in SUBJECTS:
                pacts[first].append(f"\t\tcreate_diplomatic_pact = {{ country = c:{second} type = {SUBJECTS[kind]} }}")
            elif kind == "alliance":
                relations[first].append(f"\t\tset_relations = {{ country = c:{second} value = 50 }}")
        out = ["DIPLOMACY = {"]
        for tag in sorted(set(pacts) | set(relations)):
            out += [f"\tc:{tag} ?= {{", *pacts[tag], *relations[tag], "\t}"]
        out.append("}")
        write_text(config.MOD / "common" / "history" / "diplomacy" / "aoe_diplomacy.txt", "\n".join(out) + "\n")

    # ------------------------------------------------------------- run ----

    def run(self):
        self.native_owner = {}
        self.make_natives()
        self.build_states_and_pops()
        self.existing = set(self.country_list())
        religion_loc = self.build_religions()
        culture_loc = self.build_cultures()
        owners = self.write(culture_loc, religion_loc)
        total = sum(sum(v.values()) for v in self.state_pops.values())
        print(f"{len(owners)} countries ({len(self.natives)} native), population {total:,}")
        for w in Counter(self.warnings).most_common(25):
            print("warning:", w[0], f"(x{w[1]})" if w[1] > 1 else "")


if __name__ == "__main__":
    HistoryBuilder().run()
