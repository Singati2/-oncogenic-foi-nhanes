"""Download NHANES public-use XPT files from CDC.

Reads the canonical file list from manifest.yaml; downloads each file once,
caches under data/raw/, and re-uses on subsequent runs (idempotent).

Run directly:
    python -m analysis.data_pull.download              # all cycles
    python -m analysis.data_pull.download C F          # subset of cycles
"""

from __future__ import annotations

import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).parent / "manifest.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_manifest() -> dict:
    """Load and return the cycle/file manifest."""
    with MANIFEST.open() as f:
        return yaml.safe_load(f)


def download_one(url: str, target: Path, *, timeout: int = 60) -> Path:
    """Download a single file. Skip if already on disk and non-empty (cache)."""
    if target.exists() and target.stat().st_size > 0:
        logger.info("cached  %s", target.relative_to(PROJECT_ROOT))
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("fetch   %s", url)
    req = urllib.request.Request(
        url, headers={"User-Agent": "oncogenic-foi-nhanes/0.0.1 (research)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            target.write_bytes(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"HTTP {e.code} fetching {url} — manifest may need update"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error fetching {url}: {e}") from e

    if target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Empty download from {url}")

    logger.info("saved   %s (%d bytes)", target.relative_to(PROJECT_ROOT), target.stat().st_size)
    return target


def download_all(*, cycles: list[str] | None = None) -> dict[str, dict[str, Path]]:
    """Download every (cycle, pathogen) file listed in the manifest.

    Returns a nested mapping: cycle_code -> pathogen_key -> local Path.
    """
    manifest = load_manifest()
    base = manifest["base_url"].rstrip("/")
    selected = cycles or list(manifest["cycles"].keys())

    paths: dict[str, dict[str, Path]] = {}
    for cycle_code in selected:
        if cycle_code not in manifest["cycles"]:
            raise KeyError(f"Unknown cycle code: {cycle_code!r}")
        cycle = manifest["cycles"][cycle_code]
        cycle_paths: dict[str, Path] = {}
        for pathogen, spec in cycle["files"].items():
            url = f"{base}/{spec['relpath']}"
            target = RAW_DIR / cycle["years"] / f"{spec['file']}.XPT"
            cycle_paths[pathogen] = download_one(url, target)
        paths[cycle_code] = cycle_paths
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    requested = sys.argv[1:] or None
    download_all(cycles=requested)


if __name__ == "__main__":
    main()
