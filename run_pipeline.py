"""Run the whole freeze-damage pipeline end to end.

    python run_pipeline.py            # full 10 m run (use on Kaggle)
    python run_pipeline.py --test     # quick 60 m run (small download)

All settings are in config.yaml. Results go to outputs/<resolution>m/.
"""
import argparse
import gc
import os
import sys
import traceback
import time
import warnings

from freeze import (citrus, damage, maps, masks, ndvi, prior_year, report, seasonal, sensitivity, sentinel,
                    stack, tables, terrain, zones)
from freeze.config import load_config
from freeze.damage import DAMAGE_NAMES, NODATA, RECOVERY_NAMES
from freeze.io_utils import save_geotiff


def memory_gb():
    """Peak memory (RAM) used so far by this process, in GB (Linux/macOS; None elsewhere)."""
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1024**2 if sys.platform != "darwin" else kb / 1024**3   # macOS reports bytes
    except ImportError:
        return None


def step(msg):
    mem = memory_gb()
    mem_txt = f"  (peak RAM so far {mem:.1f} GB)" if mem is not None else ""
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}{mem_txt}", flush=True)


def main(config_path="config.yaml", test=False):
    cfg = load_config(config_path, test=test)
    d, out = cfg["dates"], cfg["run"]["out"]
    early, late = d["damage_early"], d["damage_late"]
    print(f"Resolution {cfg['run']['res']} m -> {out}")

    step("1. Search Sentinel-2 scenes")
    items = sentinel.search_scenes(cfg)
    inventory = sentinel.scene_inventory(items, cfg)
    print(f"  {len(inventory)} dates found")

    step("2-3. Load bands date by date (cached), cloud masks and NDVI")
    series = stack.load_series(cfg, items, cfg["run"]["cache"] / cfg["sentinel2"]["cache_dir"],
                               stack.needed_dates(cfg), rgb_date=cfg["maps"]["rgb_date"],
                               water_date=cfg["sentinel2"]["water_reference_date"],
                               table_name="clear_fraction_by_date.csv")
    nd, clear_tab, rgb, water = series["ndvi"], series["clear_tab"], series["rgb"], series["water"]
    del series

    step("4. Baseline, vegetation mask, noise check")
    base, base_count = ndvi.baseline(nd, d["baseline"])
    veg = masks.vegetation_mask(base, cfg)
    noise = ndvi.noise_check(nd, veg, cfg)
    n = noise.iloc[0]
    print(f"  noise: mean {n['mean']:.3f}, std {n['std']:.3f}, p01 {n['p01']:.3f}")

    step("5. NDVI change and damage classes")
    changes = {day: ndvi.change(nd, base, day) for day in [early, late]}
    damages = {day: damage.classify_damage(changes[day], veg, cfg) for day in changes}
    recovery = damage.classify_recovery(changes[early], changes[late], damages[early], damages[late], cfg)

    step("6. Citrus mask (USDA CDL)")
    cdl = citrus.cdl_on_grid(base, cfg)
    cit = citrus.citrus_mask(cdl, cfg)

    step("7. Kill zones")
    kz_ids = zones.label_zones(zones.kill_zone_mask(damages[early], damages[late], cit, cfg), base, cfg)
    kz = zones.zone_polygons(kz_ids, changes[early], changes[late], damages[late], cfg)
    kz.to_file(out / "vectors" / "kill_zones.gpkg", driver="GPKG")
    kz.drop(columns="geometry").to_csv(out / "tables" / "kill_zones.csv", index=False)

    step("8. Seasonal checks: citrus vs controls, and 2026 vs prior year")
    stats_2026 = seasonal.group_stats("2026", {"early": early, "late": late}, changes, damages, veg, cit, cdl, cfg)
    season = seasonal.control_table(stats_2026, cfg)
    stats_prior, clear_prior, _ = prior_year.run_prior_year(cfg, cit, cdl)
    prior = seasonal.prior_year_table(stats_2026, stats_prior, cfg)

    step("8b. Baseline sensitivity (A/B/C baseline dates)")
    sens = sensitivity.baseline_sensitivity(nd, cit, cfg)
    print(sens[["variant", "date", "assessed_ha", "damaged_pct", "shared_ha", "shared_damaged_pct"]].to_string(index=False))
    del nd                         # the NDVI time series is not needed after this point
    gc.collect()

    step("9. Frost-prone areas (DEM)")
    dem = terrain.load_dem(base, cfg)
    frost_date = early if cfg["terrain"]["damage_date"] == "early" else late
    by_band, by_low, by_frost, frost, lowlying = terrain.frost_analysis(dem, damages[frost_date], kz_ids, cit, water, cfg)

    step("10. Tables")
    dmg_tab = tables.damage_summary(damages, veg, cit, cfg)
    rec_tab = tables.recovery_summary(recovery, veg, cit, cfg)
    area_tab = tables.citrus_area(cit, base_count, veg, damages, cfg)
    cover_tab = tables.baseline_coverage(base_count, cit, water, cfg)

    step("11. GeoTIFFs")
    r = out / "rasters"
    save_geotiff(base, r / "ndvi_baseline.tif")
    save_geotiff(base_count, r / "baseline_clear_count.tif", nodata=NODATA)
    for day in changes:
        save_geotiff(changes[day], r / f"ndvi_change_{day}.tif")
        save_geotiff(damages[day], r / f"damage_class_{day}.tif", nodata=NODATA)
    save_geotiff(recovery, r / "recovery_class.tif", nodata=NODATA)
    save_geotiff(kz_ids.astype("uint32"), r / "kill_zone_id.tif", nodata=0)
    save_geotiff(cit.astype("uint8"), r / "citrus_mask.tif", nodata=NODATA)
    save_geotiff(dem, r / "dem.tif")
    save_geotiff(lowlying, r / "lowlying_index.tif")
    save_geotiff(frost, r / "frost_prone_class.tif", nodata=NODATA)

    step("12. Maps")
    c, lim = cfg["maps"]["colors"], cfg["maps"]["change_limit"]
    land = ~water
    maps.rgb_map(rgb, cfg["maps"]["rgb_date"], cfg)
    base_days = ", ".join(x[5:] for x in d["baseline"])
    maps.continuous_map(base, "ndvi_baseline", f"Baseline NDVI (median of {base_days})", "NDVI", cfg,
                        cmap="Greens", vmin=0, vmax=1)
    n_base = len(d["baseline"])
    maps.class_map(base_count.where(land, NODATA), [str(k) for k in range(1, n_base + 1)],
                   c["baseline_count"][-n_base:], "baseline_count",
                   "Number of clear baseline dates per pixel (land only)", cfg, land=land,
                   legend_title="clear dates", zero=("0 (no baseline)", c["baseline_none"]))
    dmg_labels = [DAMAGE_NAMES[k] for k in (1, 2, 3)]
    for day in changes:
        maps.continuous_map(changes[day].where(veg), f"ndvi_change_{day}",
                            f"NDVI change {day} vs baseline (vegetated pixels)", "NDVI change", cfg,
                            cmap=maps.change_cmap(cfg), vmin=-lim, vmax=lim)
        maps.class_map(damages[day], dmg_labels, c["damage"], f"damage_all_vegetation_{day}",
                       f"Damage class {day}, all vegetation", cfg, land=land, legend_title="damage")
        maps.class_map(damages[day].where(cit, NODATA), dmg_labels, c["damage"], f"damage_citrus_{day}",
                       f"Damage class {day}, citrus only (grey = other land)", cfg, land=land,
                       legend_title="citrus damage", zero=("none", c["citrus_ok"]))
    rec_labels = [RECOVERY_NAMES[k] for k in (1, 2, 3, 4)]
    maps.class_map(recovery.where(cit, NODATA), rec_labels, c["recovery"], "recovery_citrus",
                   f"Citrus recovery: {late} vs {early} (grey = other land)", cfg, land=land,
                   legend_title="recovery", zero=("undamaged", c["citrus_ok"]))
    maps.kill_zone_map(rgb, kz, cfg)
    damages[late].attrs["date"] = late
    maps.kill_zone_zoom(damages[late], cit, kz, land, cfg)
    t = cfg["terrain"]
    maps.class_map(frost, [f"hollow ({t['deep_hollow_m']} to {t['hollow_m']} m)",
                           f"deep hollow (< {t['deep_hollow_m']} m)"], c["frost"],
                   "frost_prone", "Frost-prone areas: low-lying hollows (land only)", cfg, land=land,
                   legend_title="below surroundings")
    maps.lowlying_chart(by_low, cfg)
    maps.elevation_chart(by_band, cfg)
    maps.seasonal_chart(season, cfg)
    maps.prior_year_chart(prior, cfg)

    step("13. Report (Markdown + PDF)")
    results = locals()
    md_path, pdf_path = report.write_report(results, cfg)
    print(f"  {md_path}\n  {pdf_path}")

    print(f"\nDone. Results in {out}  (peak RAM {memory_gb() or float('nan'):.1f} GB)")
    return results


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # e.g. median of all-NaN pixels
    p = argparse.ArgumentParser(description="Freeze damage assessment pipeline")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--test", action="store_true", help="coarse 60 m run for quick testing")
    a = p.parse_args()
    try:
        main(a.config, test=a.test)
    except Exception:
        # Print the error and exit at once: after a failed download, background
        # download threads can otherwise keep the program hanging instead of exiting.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
