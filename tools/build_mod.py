"""Build the whole mod into config.MOD (the game's mod folder).

    python build_mod.py              run every step
    python build_mod.py --skip-map   reuse build/province_map.csv and build/redraw/ (alignment and redraw)

Steps: province alignment -> province redraw -> state regions -> history -> flags -> static files.
"""
import json
import shutil
import sys

import config
import build_flags
import build_history
import build_province_map
import build_states
import redraw_provinces

MOD_NAME = "Age of Enlightenment"
MOD_VERSION = "0.1"
GAME_VERSION = "1.13.*"
START_DATE = "1700.1.1"
END_DATE = "1821.12.31"
GENERATED = [".metadata", "common", "gfx", "localization", "map_data"]  # top-level folders the build writes

# Vanilla folders the mod replaces wholesale: everything that sets up the
# 1836 world for specific countries and states.
REPLACE_PATHS = [
    "common/country_definitions",
    "common/flag_definitions",
    *[f"common/history/{folder}" for folder in (
        "ai", "buildings", "characters", "countries", "diplomacy", "diplomatic_plays", "global", "governments",
        "interests", "lobbies", "military_deployments", "military_formations", "political_movements", "pops",
        "population", "power_blocs", "production_methods", "states", "trade", "treaties")],
]

# The generic parts of vanilla's common/history/global/00_global.txt.
GLOBAL_HISTORY = """GLOBAL = {
	every_country = {
		limit = { NOT = { has_law_or_variant = law_type:law_slavery_banned } }
		if = {
			limit = {
				has_law_or_variant = law_type:law_colonial_slavery
			}
			ig:ig_landowners ?= { add_ideology = ideology_pro_slavery_colonial }
		}
		else = {
			ig:ig_landowners ?= { add_ideology = ideology_pro_slavery }
		}
	}

	every_country = {
		limit = {
			has_law = law_type:law_monarchy
			has_law_or_variant = law_type:law_autocracy
		}
		set_variable = king_in_parliament_possible
	}

	every_country = { # Mercantile ideology
		limit = {
			OR = {
				has_law_or_variant = law_type:law_traditionalism
				has_law_or_variant = law_type:law_isolationism
			}
		}
		ig:ig_petty_bourgeoisie = {
			add_ideology = ideology_mercantile
		}
	}
}
"""


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")


def write_static_files():
    mod = config.MOD
    metadata = {
        "name": MOD_NAME,
        "id": "aoe_vic3",
        "version": MOD_VERSION,
        "supported_game_version": GAME_VERSION,
        "short_description": "Age of Enlightenment: the AoE Remake (multiplayer) world of 1700, ported from Victoria 2.",
        "tags": ["Total Conversion", "Map", "Historical"],
        "relationships": [],
        "game_custom_data": {"multiplayer_synchronized": True, "replace_paths": REPLACE_PATHS},
    }
    (mod / ".metadata").mkdir(parents=True, exist_ok=True)
    (mod / ".metadata" / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write(mod / "common" / "defines" / "aoe_defines.txt",
          f'NGame = {{\n\tSTART_DATE = "{START_DATE}"\n\tEND_DATE = "{END_DATE}"\n}}\n')
    write(mod / "common" / "history" / "global" / "aoe_global.txt", GLOBAL_HISTORY)


def main():
    # Start clean so files from earlier runs can't linger. Only the folders the
    # build writes: the mod folder can hold other things, like a checkout of this repo.
    for name in GENERATED:
        if (config.MOD / name).exists():
            shutil.rmtree(config.MOD / name)
    if "--skip-map" not in sys.argv:
        build_province_map.main()
        redraw_provinces.main()
    redraw_provinces.write_mod_files()
    builder = build_states.StateBuilder()
    builder.write(builder.build())
    build_history.HistoryBuilder().run()
    build_flags.main()
    write_static_files()
    print(f"mod written to {config.MOD}")


if __name__ == "__main__":
    main()
