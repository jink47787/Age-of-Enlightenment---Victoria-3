"""AoE (Vic2) history flattened to the start date: provinces, countries, pops, diplomacy.

Vic2 applies a history file's undated entries, then every dated block up to
and including the start date, in date order. Later blocks are ignored.
"""
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import config
import pdx

# AoE's bookmark date. Its province history also has a block dated 1700.01.02
# (French Canada, Thirteen Colonies, ...) that Vic2 does not apply at this start.
START_DATE = (1700, 1, 1)

_DATE = re.compile(r"^(\d{1,4})\.(\d{1,2})\.(\d{1,2})$")


def parse_date(text):
    m = _DATE.match(text)
    return tuple(int(x) for x in m.groups()) if m else None


def num(text, default=0.0):
    """Lenient number parsing: AoE files contain values like '0,1'."""
    try:
        return float(str(text).replace(",", "."))
    except ValueError:
        return default


def flatten(block, date=START_DATE):
    """[(key, value)] in the order Vic2 applies them at `date`."""
    entries = [(k, v) for k, _, v in block.pairs() if parse_date(k) is None]
    dated = [(parse_date(k), v) for k, _, v in block.pairs() if parse_date(k) is not None]
    for d, b in sorted(dated, key=lambda x: x[0]):
        if d <= date and isinstance(b, pdx.Block):
            entries += [(k, v) for k, _, v in b.pairs()]
    return entries


@dataclass
class Province:
    id: int
    owner: str = None
    controller: str = None
    cores: set = field(default_factory=set)
    trade_goods: str = None
    life_rating: int = None
    colonial: int = 0
    fort: int = 0
    naval_base: int = 0
    factories: list = field(default_factory=list)


@dataclass
class Country:
    tag: str
    color: tuple = (128, 128, 128)
    graphical_culture: str = None
    capital: int = None
    primary_culture: str = None
    accepted: list = field(default_factory=list)
    religion: str = None
    government: str = None
    civilized: bool = False
    literacy: float = 0.0
    prestige: float = 0.0
    reforms: dict = field(default_factory=dict)
    techs: set = field(default_factory=set)
    flags: set = field(default_factory=set)


@dataclass
class Pop:
    type: str
    culture: str
    religion: str
    size: int


REFORM_KEYS = {"slavery", "vote_franschise", "upper_house_composition", "public_meetings", "press_rights",
               "trade_unions", "political_parties", "debt_law", "penal_system", "citizens_rights", "centralization",
               "colonial_policy", "school_reforms", "wage_reform", "work_hours", "safety_regulations",
               "unemployment_subsidies", "pensions", "health_care"}
NON_TECH_KEYS = REFORM_KEYS | {"capital", "primary_culture", "culture", "religion", "government", "plurality",
                               "nationalvalue", "literacy", "non_state_culture_literacy", "civilized", "prestige",
                               "ruling_party", "upper_house", "last_election", "consciousness",
                               "nonstate_consciousness", "schools", "foreign_investment", "set_country_flag",
                               "clr_country_flag", "decision", "remove_accepted_culture", "oob", "is_releasable_vassal",
                               "govt_flag", "colonial_points", "set_global_flag", "election", "badboy"}


def read_provinces(date=START_DATE):
    out = {}
    for path in sorted((config.AOE / "history" / "provinces").glob("*/*.txt")):
        m = re.match(r"\s*(\d+)", path.name)
        if not m:
            continue
        p = Province(int(m.group(1)))
        for k, v in flatten(pdx.parse_file(path), date):
            if k == "owner":
                p.owner = v
            elif k == "controller":
                p.controller = v
            elif k == "add_core":
                p.cores.add(v)
            elif k == "remove_core":
                p.cores.discard(v)
            elif k == "trade_goods":
                p.trade_goods = v
            elif k == "life_rating":
                p.life_rating = int(num(v))
            elif k == "colonial":
                p.colonial = int(num(v))
            elif k in ("fort", "naval_base"):
                setattr(p, k, int(num(v)))
            elif k == "state_building" and isinstance(v, pdx.Block):
                p.factories.append((v.get("building"), int(num(v.get("level", 1)))))
        out[p.id] = p
    return out


def read_country_files():
    """{tag: path of common/countries/*.txt} from common/countries.txt, in file order."""
    out = {}
    for tag, _, path in pdx.parse_file(config.AOE / "common" / "countries.txt").pairs():
        if isinstance(path, str) and re.fullmatch(r"[A-Z][A-Z0-9]{2}", tag):
            out[tag] = config.AOE / "common" / path
    return out


def read_countries(date=START_DATE):
    out = {}
    for tag, path in read_country_files().items():
        c = Country(tag)
        if path.exists():
            block = pdx.parse_file(path)
            color = block.get("color")
            if isinstance(color, pdx.Block) and len(color.values()) >= 3:
                c.color = tuple(int(num(x)) for x in color.values()[:3])
            c.graphical_culture = block.get("graphical_culture")
        out[tag] = c

    history = defaultdict(list)
    for path in sorted((config.AOE / "history" / "countries").glob("*.txt"), key=lambda p: p.name.lower()):
        history[path.name[:3].upper()].append(path)
    for tag, c in out.items():
        if not history.get(tag):
            continue
        for k, v in flatten(pdx.parse_file(history[tag][0]), date):
            if isinstance(v, pdx.Block):
                continue
            if k == "capital":
                c.capital = int(num(v))
            elif k == "primary_culture":
                c.primary_culture = v
            elif k == "culture":
                if v not in c.accepted:
                    c.accepted.append(v)
            elif k == "remove_accepted_culture" and v in c.accepted:
                c.accepted.remove(v)
            elif k == "religion":
                c.religion = v
            elif k == "government":
                c.government = v
            elif k == "civilized":
                c.civilized = v == "yes"
            elif k == "literacy":
                c.literacy = num(v)
            elif k == "prestige":
                c.prestige = num(v)
            elif k == "set_country_flag":
                c.flags.add(v)
            elif k in REFORM_KEYS:
                c.reforms[k] = v
            elif k not in NON_TECH_KEYS and v in ("1", "yes"):
                c.techs.add(k)
    return out


def read_pops():
    """{province id: [Pop]} from history/pops (Vic2 reads the 1836.1.1 folder for every start)."""
    out = defaultdict(list)
    for path in sorted((config.AOE / "history" / "pops").glob("*/*.txt")):
        for prov, _, block in pdx.parse_file(path).pairs():
            if not prov.isdigit() or not isinstance(block, pdx.Block):
                continue
            for ptype, _, pb in block.pairs():
                if isinstance(pb, pdx.Block) and pb.get("culture") and pb.get("size"):
                    out[int(prov)].append(Pop(ptype, pb.get("culture"), pb.get("religion"), int(num(pb.get("size")))))
    return out


def read_diplomacy(date=START_DATE):
    """[(kind, first, second)] for relationships active on `date`."""
    out = []
    for path in sorted((config.AOE / "history" / "diplomacy").glob("*.txt")):
        for kind, _, block in pdx.parse_file(path).pairs():
            if not isinstance(block, pdx.Block):
                continue
            start = parse_date(block.get("start_date", "1.1.1")) or (1, 1, 1)
            end = parse_date(block.get("end_date", "9999.1.1")) or (9999, 1, 1)
            if start <= date < end and block.get("first") and block.get("second"):
                out.append((kind, block.get("first"), block.get("second")))
    return out


def read_cultures():
    """{culture: (group, block)} and {group: block}."""
    cultures, groups = {}, {}
    for gk, _, gb in pdx.parse_file(config.AOE / "common" / "cultures.txt").pairs():
        if not isinstance(gb, pdx.Block):
            continue
        groups[gk] = gb
        for ck, _, cb in gb.pairs():
            if isinstance(cb, pdx.Block) and cb.get("color") is not None:
                cultures[ck] = (gk, cb)
    return cultures, groups


def read_religions():
    """{religion: (group, block)}."""
    out = {}
    for gk, _, gb in pdx.parse_file(config.AOE / "common" / "religion.txt").pairs():
        if isinstance(gb, pdx.Block):
            for rk, _, rb in gb.pairs():
                if isinstance(rb, pdx.Block):
                    out[rk] = (gk, rb)
    return out
