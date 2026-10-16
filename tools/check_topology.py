"""Compare which provinces border each other in AoE vs. the generated Vic3 map.

    python check_topology.py AOE_PROVINCE [AOE_PROVINCE ...]

For each AoE province, lists AoE neighbours that no longer border it in Vic3
("missing") and Vic3 neighbours it doesn't have in AoE ("extra"). Reads
build/province_map.csv.
"""
import sys

import numpy as np

import config
from aoedata import read_province_map
from mapdata import province_borders, read_vic2_map, read_vic3_map


def main(provinces):
    aoe = read_vic2_map(config.AOE / "map")
    rows = read_province_map()
    v3 = read_vic3_map(config.VIC3)
    index = {p: i for i, p in enumerate(v3.provinces)}
    aoe_of = {index[r["vic3_province"]]: r["aoe_province"] for r in rows}

    ys, xs = np.nonzero(np.isin(aoe.labels, provinces))
    window = aoe.labels[max(ys.min() - 40, 0):ys.max() + 40, max(xs.min() - 40, 0):xs.max() + 40]
    aoe_pairs = set(province_borders(window))
    ours = {tuple(sorted((aoe_of[a], aoe_of[b]))) for a, b in province_borders(v3.labels)
            if a in aoe_of and b in aoe_of and aoe_of[a] != aoe_of[b]}
    problems = 0
    for p in provinces:
        theirs = {q for pair in aoe_pairs if p in pair for q in pair if q != p and q and q not in aoe.sea}
        mine = {q for pair in ours if p in pair for q in pair if q != p}
        missing, extra = sorted(theirs - mine), sorted(mine - theirs)
        problems += len(missing) + len(extra)
        print(f"{p} {aoe.names.get(p, '')}: missing {missing} extra {extra}")
    print(f"{problems} differences")


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
