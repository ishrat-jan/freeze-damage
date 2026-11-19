"""NDVI, baseline, noise check and change maps.

- compute_ndvi(): NDVI = (nir - red) / (nir + red), NaN where the pixel is not clear.
- baseline(): per-pixel median NDVI of the pre-freeze dates (each date
  cloud-masked first), plus how many clear dates went into it
  (pixels with 0 have no baseline).
- noise_check(): NDVI difference between two pre-freeze dates on vegetated
  pixels. This shows how much NDVI changes with no freeze at all, which justifies
  the -0.10 "no damage" threshold.
- change(): NDVI(date) - baseline. Negative = vegetation got less green.
"""
import pandas as pd

from freeze.sentinel import select_date


def compute_ndvi(refl, clear):
    """Masked NDVI time series (float32)."""
    ndvi = (refl.nir - refl.red) / (refl.nir + refl.red)
    return ndvi.where(clear).astype("float32").rename("ndvi")


def baseline(ndvi, days):
    """Median NDVI of the baseline dates, and the number of clear baseline dates.

    days: list of 'YYYY-MM-DD' (2026 or prior-year baseline from config.yaml).
    """
    stack = ndvi.sel(time=pd.to_datetime(days))
    base = stack.median("time", skipna=True).rename("ndvi_baseline")
    count = stack.notnull().sum("time").astype("uint8").rename("baseline_count")
    return base, count


def noise_check(ndvi, veg, cfg):
    """Statistics of NDVI(later) - NDVI(earlier) for the two pre-freeze dates."""
    early, late = cfg["dates"]["noise_pair"]
    diff = select_date(ndvi, late) - select_date(ndvi, early)
    vals = diff.where(veg).values
    vals = vals[~pd.isna(vals)]
    s = pd.Series(vals.astype("float64"))   # float64 so rounding prints cleanly
    t = cfg["thresholds"]
    df = pd.DataFrame([{
        "pair": f"{late} minus {early}",
        "n_pixels": len(s),
        "mean": round(s.mean(), 4),
        "std": round(s.std(), 4),
        "p01": round(s.quantile(0.01), 4),
        "p05": round(s.quantile(0.05), 4),
        "p50": round(s.quantile(0.50), 4),
        "p95": round(s.quantile(0.95), 4),
        "p99": round(s.quantile(0.99), 4),
        # share of stable pixels that would wrongly be called "damaged"
        "pct_below_mild_threshold": round(100 * (s < t["mild"]).mean(), 2),
    }])
    df.to_csv(cfg["run"]["out"] / "tables" / "noise_check.csv", index=False)
    return df


def change(ndvi, base, day):
    """NDVI change of one date against the baseline."""
    return (select_date(ndvi, day) - base).rename(f"ndvi_change_{day}")
