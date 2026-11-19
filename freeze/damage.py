"""Damage and recovery classes from NDVI change.

Damage classes (per date, vegetated pixels only):
    0 none      change >= mild threshold (-0.10)
    1 mild      -0.10 > change >= -0.20
    2 moderate  -0.20 > change >= -0.30
    3 severe    change < -0.30
    255         no data (not vegetated, cloudy, or no baseline)

Recovery classes (late date vs early date):
    0 undamaged        not damaged on either date
    1 recovered        damaged early, no damage on the late date
    2 partly recovered damaged on both dates, NDVI rose by more than the tolerance
    3 same or worse    damaged on both dates, NDVI did not rise by more than the tolerance
    4 new damage       not damaged early, damaged on the late date
    255                no data on either date
"""
import numpy as np
import xarray as xr

NODATA = 255
DAMAGE_NAMES = {0: "none", 1: "mild", 2: "moderate", 3: "severe"}
RECOVERY_NAMES = {0: "undamaged", 1: "recovered", 2: "partly recovered",
                  3: "same or worse", 4: "new damage"}


def classify_damage(change, veg, cfg):
    """Turn an NDVI change map into damage classes (uint8)."""
    t = cfg["thresholds"]
    c = change.values
    classes = np.select(
        [c < t["severe"], c < t["moderate"], c < t["mild"], c >= t["mild"]],
        [3, 2, 1, 0],
        default=NODATA,              # NaN fails every comparison -> nodata
    ).astype("uint8")
    classes[~veg.values] = NODATA
    return xr.DataArray(classes, coords=change.coords, dims=change.dims, name="damage_class")


def classify_recovery(change_early, change_late, dmg_early, dmg_late, cfg):
    """Compare the late date with the early date for each pixel (uint8 classes)."""
    tol = cfg["thresholds"]["recovery_tolerance"]
    e, l = dmg_early.values, dmg_late.values
    rise = (change_late - change_early).values          # = NDVI(late) - NDVI(early)
    valid = (e != NODATA) & (l != NODATA)
    hit_e, hit_l = e >= 1, l >= 1

    out = np.full(e.shape, NODATA, dtype="uint8")
    out[valid & ~hit_e & ~hit_l] = 0
    out[valid & hit_e & ~hit_l] = 1
    out[valid & hit_e & hit_l & (rise > tol)] = 2
    out[valid & hit_e & hit_l & (rise <= tol)] = 3
    out[valid & ~hit_e & hit_l] = 4
    return xr.DataArray(out, coords=dmg_early.coords, dims=dmg_early.dims, name="recovery_class")
