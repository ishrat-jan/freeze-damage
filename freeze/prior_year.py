"""Prior-year (2025) NDVI change, computed exactly like 2026.

Loads the prior-year scenes listed in config.yaml (only those dates, to keep
the download small), on the same grid, then applies the same cloud mask,
baseline median, vegetation threshold and damage thresholds. The 2025 citrus
mask is the same CDL layer as for 2026.
"""
from freeze import damage, masks, ndvi, seasonal, sentinel


def run_prior_year(cfg, cit, cdl):
    """Return (stats table, clear-fraction table, baseline count) for the prior year."""
    p = cfg["prior_year"]
    dates = p["baseline"] + [p["damage_early"], p["damage_late"]]
    items = sentinel.search_scenes(cfg, p["start_date"], p["end_date"], dates=dates)
    refl = sentinel.to_reflectance(sentinel.load_dn(cfg, items, cache_file=p["cache_file"]), cfg)

    clear = masks.clear_mask(refl, cfg)
    clear_tab = masks.clear_fraction_table(refl, clear, cfg, save=False)
    clear_tab.to_csv(cfg["run"]["out"] / "tables" / f"clear_fraction_by_date_{p['label']}.csv", index=False)
    nd = ndvi.compute_ndvi(refl, clear)
    base, count = ndvi.baseline(nd, p["baseline"])
    veg = masks.vegetation_mask(base, cfg)

    changes = {d: ndvi.change(nd, base, d) for d in [p["damage_early"], p["damage_late"]]}
    damages = {d: damage.classify_damage(changes[d], veg, cfg) for d in changes}
    stats = seasonal.group_stats(p["label"], {"early": p["damage_early"], "late": p["damage_late"]},
                                 changes, damages, veg, cit, cdl, cfg)
    return stats, clear_tab, count
