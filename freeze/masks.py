"""Masks: which pixels are usable on each date, and which are vegetated.

- clear_mask(): True where the pixel is cloud-free (SCL class 4 or 5) AND has
  valid reflectance (nir + red > 0). Everything else (cloud, shadow, cirrus,
  water, nodata) is excluded.
- clear_fraction_table(): share of clear pixels per date in the AOI -> CSV.
  This documents why some dates are used and others are not.
- vegetation_mask(): True where baseline NDVI >= threshold.
"""
import pandas as pd


def clear_mask(refl, cfg):
    """Per-date boolean mask of usable pixels."""
    classes = cfg["sentinel2"]["clear_scl_classes"]
    scl_ok = refl.scl.isin(classes)
    valid = (refl.nir + refl.red) > 0      # NaN (nodata) also gives False here
    return scl_ok & valid


def clear_fraction_table(refl, mask, cfg, save=True):
    """Percentage of AOI pixels that are clear on each date."""
    has_data = refl.scl > 0                # SCL 0 = no data (outside the swath)
    n_data = has_data.sum(["y", "x"])
    n_clear = mask.sum(["y", "x"])
    df = pd.DataFrame({
        "date": pd.to_datetime(mask.time.values).strftime("%Y-%m-%d"),
        "pixels_with_data": n_data.values,
        "clear_pixels": n_clear.values,
    })
    df["clear_pct"] = (100 * df.clear_pixels / df.pixels_with_data.clip(lower=1)).round(1)
    # Why pixels were rejected: SCL groups as % of pixels with data
    groups = cfg["sentinel2"]["scl_report_groups"]
    for name, classes in groups.items():
        n = refl.scl.isin(classes).sum(["y", "x"]).values
        df[f"{name}_pct"] = (100 * n / df.pixels_with_data.clip(lower=1)).round(1)
    # BOA offset actually applied (decided from the data, see sentinel.py) and the Earth Search flag
    df["boa_offset_used"] = refl.boa_offset.values.astype(int)
    df["es_offset_flag"] = refl.boa_flag.values
    if save:
        df.to_csv(cfg["run"]["out"] / "tables" / "clear_fraction_by_date.csv", index=False)
    return df


def water_mask(refl, cfg):
    """Water pixels (SCL class 6) on a cloud-free reference date."""
    day = pd.Timestamp(cfg["sentinel2"]["water_reference_date"])
    return (refl.scl.sel(time=day) == cfg["sentinel2"]["water_scl_class"]).rename("water")


def vegetation_mask(baseline_ndvi, cfg):
    """Vegetated pixels = baseline NDVI at or above the threshold."""
    return baseline_ndvi >= cfg["thresholds"]["vegetation_ndvi"]
