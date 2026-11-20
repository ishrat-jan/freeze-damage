"""Memory-safe loading: process one date at a time and keep only what is needed.

At 10 m one date-band is ~70 MB as float32, so holding every band of every date
as reflectance would need ~8-12 GB. Instead, for each date we:
  1. load its digital numbers (from the per-date cache, or download),
  2. convert to reflectance, build the clear mask, add a row to the clear-fraction table,
  3. keep its NDVI only if that date is used later (baseline, damage, noise, sensitivity),
  4. keep the true-colour bands only for the RGB map date, and SCL only for the water date,
  5. drop everything else before the next date.
Every calculation is exactly the same as on the full stack, just done per date.
"""
import gc

import pandas as pd
import xarray as xr

from freeze import masks, ndvi, sentinel


def load_series(cfg, items, cache_dir, keep_dates, rgb_date=None, water_date=None, table_name=None):
    """Return dict: ndvi (time, y, x float32 for keep_dates), clear_tab, rgb, water."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    days = sorted({sentinel._day(it) for it in items})
    keep = set(keep_dates)
    nd_list, rows, rgb, water = [], [], None, None
    for day in days:
        refl = sentinel.to_reflectance(sentinel.load_dn_date(cfg, items, day, cache_dir), cfg)
        clear = masks.clear_mask(refl, cfg)
        rows.append(masks.clear_fraction_table(refl, clear, cfg, save=False))
        if day in keep:
            nd_list.append(ndvi.compute_ndvi(refl, clear))
        if day == rgb_date:
            rgb = refl[["red", "green", "blue"]]
        if day == water_date:
            water = masks.water_mask(refl, cfg)
        del refl, clear
        gc.collect()                                     # free this date before the next one

    missing = keep - set(days)
    if missing:
        raise RuntimeError(f"Dates in config.yaml without a Sentinel-2 scene: {sorted(missing)}")
    clear_tab = pd.concat(rows, ignore_index=True)
    if table_name:
        clear_tab.to_csv(cfg["run"]["out"] / "tables" / table_name, index=False)
    return {"ndvi": xr.concat(nd_list, dim="time"), "clear_tab": clear_tab, "rgb": rgb, "water": water}


def needed_dates(cfg):
    """2026 dates whose NDVI is used: baseline, damage dates, noise pair, sensitivity variants."""
    d = cfg["dates"]
    days = set(d["baseline"]) | {d["damage_early"], d["damage_late"]} | set(d["noise_pair"])
    for v in cfg["baseline_sensitivity"]["variants"].values():
        days |= set(v["dates"])
    return sorted(days)
