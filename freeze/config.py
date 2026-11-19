"""Load config.yaml, apply test mode, and create the cache/output folders.

Every other module receives the `cfg` dict returned by load_config(), so all
settings come from one place.
"""
from pathlib import Path

import yaml


def load_config(path="config.yaml", test=False):
    """Read the YAML config and add derived values (active resolution, folders)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    run = cfg["run"]
    # Test mode = coarse pixels, so downloads are small and runs are fast.
    res = run["test_resolution_m"] if test else run["resolution_m"]
    run["res"] = res
    run["test"] = test

    # Separate folders per resolution, so a test run never overwrites the 10 m results.
    root = Path(path).resolve().parent
    cache = root / run["data_dir"] / f"{res}m"
    out = root / run["output_dir"] / f"{res}m"
    run["cache"] = cache
    run["out"] = out
    for sub in ["rasters", "vectors", "maps", "tables"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    # Pixel area in hectares (1 ha = 10 000 m2), used for all area statistics.
    run["pixel_ha"] = res * res / 10000
    return cfg
