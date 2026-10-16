"""Paths shared by the conversion tools.

Game install locations can be overridden with the VIC2_DIR / VIC3_DIR
environment variables.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "sources"
BUILD = ROOT / "build"
MOD = ROOT / "mod"

AOE_REPO_URL = "https://github.com/MaTi5893/AoE-remake"
AOE_BRANCH = "SP-version"
AOE_REPO = SOURCES / "AoE-remake"
AOE = AOE_REPO / "AoE Remake"

CONVERTER_MAPPINGS = SOURCES / "converter_mappings"
EU4_TO_VIC2 = CONVERTER_MAPPINGS / "eu4_to_vic2_province_mappings.txt"
EU4_TO_VIC3 = CONVERTER_MAPPINGS / "eu4_to_vic3_province_mappings.txt"

_STEAM = Path(r"C:\Program Files (x86)\Steam\steamapps\common")
VIC2 = Path(os.environ.get("VIC2_DIR", _STEAM / "Victoria 2"))
VIC3 = Path(os.environ.get("VIC3_DIR", _STEAM / "Victoria 3" / "game"))
