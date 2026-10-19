"""Download the inputs the conversion needs into sources/.

- The AoE Remake mod (multiplayer version, main branch), only the folders we convert.
- The EU4->Vic2 and EU4->Vic3 province mappings from the Paradox Game
  Converters project (MIT licensed), used to align the two maps.

Re-running pulls the latest AoE commit.
    python fetch_sources.py --checkout    # only re-apply the sparse checkout, at the current commit
"""
import subprocess
import sys
import urllib.request

from config import AOE, AOE_BRANCH, AOE_REPO, AOE_REPO_URL, CONVERTER_MAPPINGS, EU4_TO_VIC2, EU4_TO_VIC3, SOURCES

AOE_FOLDERS = ["map", "history", "common", "localisation", "gfx/flags", "decisions", "events"]

CONVERTER_FILES = {
    EU4_TO_VIC2: "https://raw.githubusercontent.com/ParadoxGameConverters/EU4ToVic2/master/EU4ToVic2/Data_Files/configurables/province_mappings.txt",
    EU4_TO_VIC3: "https://raw.githubusercontent.com/ParadoxGameConverters/EU4ToVic3/master/EU4ToVic3/Data_Files/configurables/province_mappings.txt",
}


def git(*args, cwd=None):
    subprocess.run(["git", "-c", "core.longpaths=true", *args], cwd=cwd, check=True)


def fetch_aoe():
    if not (AOE_REPO / ".git").exists():
        git("clone", "--filter=blob:none", "--no-checkout", "--single-branch",
            "--branch", AOE_BRANCH, AOE_REPO_URL, str(AOE_REPO))
    else:
        git("fetch", "origin", AOE_BRANCH, cwd=AOE_REPO)
        git("reset", "--soft", f"origin/{AOE_BRANCH}", cwd=AOE_REPO)
    checkout_aoe()


def checkout_aoe():
    """Check out only AOE_FOLDERS at the current commit."""
    # Write sparse patterns directly: `git sparse-checkout set` mangles
    # leading slashes into Windows paths under Git Bash.
    patterns = [f"/{AOE.name}/{folder}/" for folder in AOE_FOLDERS] + [f"/{AOE.name}.mod"]
    git("config", "core.sparseCheckout", "true", cwd=AOE_REPO)
    # .git is a file in a worktree; ask git where this checkout's sparse file lives
    path = subprocess.run(["git", "rev-parse", "--git-path", "info/sparse-checkout"], cwd=AOE_REPO,
                          check=True, capture_output=True, text=True).stdout.strip()
    sparse = AOE_REPO / path
    sparse.parent.mkdir(parents=True, exist_ok=True)
    sparse.write_text("\n".join(patterns) + "\n")
    git("read-tree", "-mu", "HEAD", cwd=AOE_REPO)
    git("log", "-1", f"--format={AOE.name} at %h (%ad): %s", "--date=short", cwd=AOE_REPO)


def fetch_converter_mappings():
    CONVERTER_MAPPINGS.mkdir(parents=True, exist_ok=True)
    for path, url in CONVERTER_FILES.items():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)


if __name__ == "__main__":
    SOURCES.mkdir(exist_ok=True)
    if "--checkout" in sys.argv:
        checkout_aoe()
    else:
        fetch_aoe()
        fetch_converter_mappings()
