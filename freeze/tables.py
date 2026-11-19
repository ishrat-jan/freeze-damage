"""Summary tables (CSV) of areas in hectares and percentages.

- damage_summary.csv   : damage class areas per date, for citrus and all vegetation
- recovery_summary.csv : recovery class areas (citrus and all vegetation)
- citrus_area.csv      : how much CDL citrus could actually be assessed, and why not
- baseline_coverage.csv: number of clear baseline dates per pixel
"""
import numpy as np
import pandas as pd

from freeze.damage import DAMAGE_NAMES, NODATA, RECOVERY_NAMES


def _class_areas(classes, mask, names, ha):
    """Area and % of each class inside `mask` (nodata excluded from the %)."""
    v = classes.values[mask.values if hasattr(mask, "values") else mask]
    v = v[v != NODATA]
    total = max(len(v), 1)
    return [{"class": name, "area_ha": round((v == k).sum() * ha, 2),
             "pct": round(100 * (v == k).sum() / total, 1)} for k, name in names.items()]


def damage_summary(damages, veg, citrus, cfg):
    """damages: dict {date: damage-class DataArray}."""
    ha = cfg["run"]["pixel_ha"]
    rows = []
    for day, dmg in damages.items():
        for group, m in [("citrus", citrus), ("all vegetation", veg)]:
            for r in _class_areas(dmg, m, DAMAGE_NAMES, ha):
                rows.append({"date": day, "group": group, **r})
    df = pd.DataFrame(rows)
    # "damaged" = mild + moderate + severe, the headline number
    dmg_rows = df[df["class"] != "none"].groupby(["date", "group"], as_index=False)[["area_ha", "pct"]].sum()
    dmg_rows["class"] = "damaged (mild or worse)"
    df = pd.concat([df, dmg_rows]).sort_values(["date", "group"], kind="stable")
    df.to_csv(cfg["run"]["out"] / "tables" / "damage_summary.csv", index=False)
    return df


def recovery_summary(recovery, veg, citrus, cfg):
    ha = cfg["run"]["pixel_ha"]
    rows = []
    for group, m in [("citrus", citrus), ("all vegetation", veg)]:
        for r in _class_areas(recovery, m, RECOVERY_NAMES, ha):
            rows.append({"group": group, **r})
    df = pd.DataFrame(rows)
    df.to_csv(cfg["run"]["out"] / "tables" / "recovery_summary.csv", index=False)
    return df


def citrus_area(citrus, base_count, veg, damages, cfg):
    """Explain the step from all CDL citrus to the citrus we could assess.

    Each CDL citrus pixel falls in exactly one row:
      - no clear baseline date          (cloud/cirrus on every baseline date)
      - baseline NDVI below threshold   (sparse canopy: young/reset or abandoned grove, bare ground)
      - no clear data on a damage date
      - assessed
    """
    ha = cfg["run"]["pixel_ha"]
    c = citrus.values
    no_base = c & (base_count.values == 0)
    low_ndvi = c & ~no_base & ~veg.values
    clear_all = np.ones_like(c)
    for dmg in damages.values():
        clear_all &= dmg.values != NODATA
    vegetated = c & veg.values
    no_damage_data = vegetated & ~clear_all
    assessed = vegetated & clear_all
    rows = [
        ("CDL citrus in AOI (total)", c),
        ("excluded: no clear baseline date", no_base),
        (f"excluded: baseline NDVI < {cfg['thresholds']['vegetation_ndvi']} (sparse/young/abandoned)", low_ndvi),
        ("excluded: not clear on a damage date", no_damage_data),
        ("assessed citrus", assessed),
    ]
    df = pd.DataFrame([{"category": s, "area_ha": round(m.sum() * ha, 2),
                        "pct_of_cdl": round(100 * m.sum() / max(c.sum(), 1), 1)} for s, m in rows])
    df.to_csv(cfg["run"]["out"] / "tables" / "citrus_area.csv", index=False)
    return df


def baseline_coverage(base_count, citrus, water, cfg):
    """How many clear baseline dates each pixel has (citrus and all land)."""
    ha = cfg["run"]["pixel_ha"]
    n_dates = len(cfg["dates"]["baseline"])
    groups = {"citrus": citrus.values, "all land": ~water.values}
    rows = []
    for k in range(n_dates + 1):
        row = {"clear_baseline_dates": k}
        for name, m in groups.items():
            v = base_count.values[m]
            row[f"{name}_ha"] = round((v == k).sum() * ha, 2)
            row[f"{name}_pct"] = round(100 * float(np.mean(v == k)), 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(cfg["run"]["out"] / "tables" / "baseline_coverage.csv", index=False)
    return df
