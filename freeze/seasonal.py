"""Seasonal checks: how much of the NDVI change is normal for the time of year?

Two comparisons, both using the same statistics (group_stats):

1. Citrus vs control vegetation in 2026. The controls (other vegetation,
   evergreen forest) also experienced the freeze, so they are NOT subtracted
   from citrus. They show whether citrus reacted more strongly than the rest.

2. 2026 vs the prior year (2025): the same baseline -> 6 Feb / 21 Feb NDVI
   change, same AOI, masks and thresholds. The 2025 change shows how much NDVI
   normally changes between mid/late January and February without this freeze.

For each group and period:
    median_change : median NDVI change vs baseline
    damaged_pct   : share of pixels that would be classed as damaged (< -0.10)
"""
import numpy as np
import pandas as pd

from freeze.damage import NODATA


def _groups(veg, citrus, cdl, cfg):
    """Boolean masks for citrus and the control groups."""
    g = {"citrus": citrus.values & veg.values,
         "other vegetation": ~citrus.values & veg.values}
    for name, classes in cfg["seasonal_check"]["controls"].items():
        g[name] = np.isin(cdl.values, classes) & veg.values
    return g


def group_stats(year, dates, changes, damages, veg, citrus, cdl, cfg):
    """One row per period (early/late) and group.

    dates: {"early": "YYYY-MM-DD", "late": ...}; changes/damages are keyed by these dates.
    """
    rows = []
    for period, day in dates.items():
        ch, dmg = changes[day].values, damages[day].values
        for name, m in _groups(veg, citrus, cdl, cfg).items():
            valid = m & ~np.isnan(ch)
            damaged = (dmg >= 1) & (dmg != NODATA)
            rows.append({
                "year": year,
                "period": period,
                "date": day,
                "group": name,
                "n_pixels": int(valid.sum()),
                "median_change": round(float(np.median(ch[valid])), 4) if valid.any() else np.nan,
                "damaged_pct": round(100 * float(damaged[valid].mean()), 1) if valid.any() else np.nan,
            })
    return pd.DataFrame(rows)


def control_table(stats_2026, cfg):
    """2026 citrus vs controls. 'citrus_minus_group' is for comparison only."""
    df = stats_2026.copy()
    citrus_med = df[df.group == "citrus"].set_index("period").median_change
    df["citrus_minus_group"] = (df.period.map(citrus_med) - df.median_change).round(4)
    df.loc[df.group == "citrus", "citrus_minus_group"] = np.nan
    df.to_csv(cfg["run"]["out"] / "tables" / "seasonal_check.csv", index=False)
    return df


def prior_year_table(stats_2026, stats_prior, cfg):
    """Side by side: 2026 vs prior year, per period and group."""
    key = ["period", "group"]
    a = stats_2026.set_index(key)[["date", "median_change", "damaged_pct"]]
    b = stats_prior.set_index(key)[["date", "median_change", "damaged_pct"]]
    y = cfg["prior_year"]["label"]
    df = a.join(b, lsuffix="_2026", rsuffix=f"_{y}").reset_index()
    # How much more NDVI was lost in 2026 than in the prior year (negative = more loss)
    df["change_2026_minus_prior"] = (df.median_change_2026 - df[f"median_change_{y}"]).round(4)
    df["damaged_pct_2026_minus_prior"] = (df.damaged_pct_2026 - df[f"damaged_pct_{y}"]).round(1)
    df.to_csv(cfg["run"]["out"] / "tables" / "prior_year_comparison.csv", index=False)
    return df
