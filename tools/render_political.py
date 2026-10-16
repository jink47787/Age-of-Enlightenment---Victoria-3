"""Render the generated 1700 political map (from mod/ files) to build/political_1700.png.

Each Vic3 province is coloured by its owner's country colour, with state
borders in grey and country borders in black. Useful for checking the result
against AoE in Vic2 without launching Vic3.
"""
import numpy as np
from PIL import Image

import config
import pdx
from mapdata import read_vic3_map


def main():
    v3 = read_vic3_map(config.VIC3)
    index = {p: i for i, p in enumerate(v3.provinces)}
    colors = {}
    for tag, _, b in pdx.parse_file(config.MOD / "common" / "country_definitions" / "aoe_countries.txt").pairs():
        colors[tag] = [int(x) for x in b.get("color").values()]

    n = len(v3.provinces)
    owner = np.full(n, -1)
    state = np.full(n, -1)
    rgb = np.zeros((n, 3), np.uint8)
    rgb[:] = (30, 50, 90)
    rgb[[index[p] for p in v3.lakes if p in index]] = (70, 110, 160)
    tag_ids = {t: i for i, t in enumerate(sorted(colors))}
    root = pdx.parse_file(config.MOD / "common" / "history" / "states" / "aoe_states.txt")
    for _, _, states in root.pairs():
        for s_i, (skey, _, sb) in enumerate(states.pairs()):
            for cs in sb.get_all("create_state"):
                tag = cs.get("country").removeprefix("c:")
                for p in cs.get("owned_provinces").values():
                    i = index["x" + p[1:].upper()]
                    owner[i], state[i] = tag_ids[tag], s_i
                    rgb[i] = colors[tag]

    lab = v3.labels
    img = rgb[lab]
    o, s = owner[lab], state[lab]
    for a, b in ((o, s),):
        pass
    state_border = np.zeros(lab.shape, bool)
    country_border = np.zeros(lab.shape, bool)
    for arr, mask in ((s, state_border), (o, country_border)):
        mask[:, :-1] |= arr[:, :-1] != arr[:, 1:]
        mask[:-1, :] |= arr[:-1, :] != arr[1:, :]
    land = o >= 0
    img[state_border & land] = (img[state_border & land] * 0.6).astype(np.uint8)
    img[country_border & land] = (15, 15, 15)
    Image.fromarray(img).save(config.BUILD / "political_1700.png", optimize=True)
    print(f"wrote {config.BUILD / 'political_1700.png'}")


if __name__ == "__main__":
    main()
