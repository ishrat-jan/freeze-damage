"""Kill zones inside citrus orchards (for insurance claims).

A kill-zone pixel is citrus that is
    - moderate or worse on the late date (21 Feb), AND
    - already damaged (mild or worse) on the early date (6 Feb),
i.e. damage that appeared right after the freeze and has not recovered.

Neighbouring kill-zone pixels are joined into patches (connected components).
Patches smaller than the minimum area are dropped. Each remaining patch gets an
ID, statistics, and a polygon outline.
"""
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape

from freeze.damage import NODATA


def kill_zone_mask(dmg_early, dmg_late, citrus, cfg):
    """Boolean mask of kill-zone pixels (before the size filter)."""
    k = cfg["kill_zones"]
    e, l = dmg_early.values, dmg_late.values
    return (l != NODATA) & (l >= k["late_min_class"]) & \
           (e != NODATA) & (e >= k["early_min_class"]) & citrus.values


def label_zones(mask, template, cfg):
    """Label connected patches, drop small ones, return a zone-ID raster (0 = none)."""
    k = cfg["kill_zones"]
    structure = np.ones((3, 3)) if k["connectivity"] == 8 else None   # None = 4-connected
    labels, n = ndimage.label(mask, structure=structure)

    # Keep patches with at least this many pixels
    min_px = max(1, math.ceil(k["min_area_ha"] / cfg["run"]["pixel_ha"]))
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_px
    keep[0] = False                                  # label 0 = background
    # Renumber kept patches 1..N
    new_id = np.zeros(len(sizes), dtype="uint32")
    new_id[keep] = np.arange(1, keep.sum() + 1)
    zone_ids = new_id[labels]
    print(f"  kill zones: {n} patches found, {keep.sum()} kept (>= {min_px} px)")
    return xr.DataArray(zone_ids, coords=template.coords, dims=template.dims, name="kill_zone_id")


def zone_polygons(zone_ids, change_early, change_late, dmg_late, cfg):
    """Polygons (GeoDataFrame) with per-zone statistics."""
    ids = zone_ids.values
    crs = zone_ids.odc.crs
    transform = zone_ids.odc.geobox.affine
    if ids.max() == 0:
        return gpd.GeoDataFrame(columns=["zone_id", "geometry"], geometry="geometry", crs=str(crs))

    # Raster -> polygons (one or more shapes per zone ID; merged below)
    geoms = [(int(v), shape(g)) for g, v in shapes(ids.astype("int32"), mask=ids > 0, transform=transform)]
    gdf = gpd.GeoDataFrame(geoms, columns=["zone_id", "geometry"], crs=str(crs)).dissolve("zone_id").reset_index()

    # Per-zone statistics with scipy (fast, no loop over pixels)
    index = np.arange(1, ids.max() + 1)
    n_px = ndimage.sum(np.ones_like(ids), ids, index)
    stats = pd.DataFrame({
        "zone_id": index,
        "n_pixels": n_px.astype(int),
        "area_ha": (n_px * cfg["run"]["pixel_ha"]).round(2),
        "mean_change_early": ndimage.mean(np.nan_to_num(change_early.values), ids, index).round(3),
        "mean_change_late": ndimage.mean(np.nan_to_num(change_late.values), ids, index).round(3),
        "min_change_late": ndimage.minimum(np.nan_to_num(change_late.values), ids, index).round(3),
        "severe_share": (ndimage.mean((dmg_late.values == 3).astype(float), ids, index)).round(2),
    })
    gdf = gdf.merge(stats, on="zone_id")
    # Centre point in lon/lat, so a field team can find each zone
    centroids = gdf.geometry.centroid.to_crs("EPSG:4326")
    gdf["lon"], gdf["lat"] = centroids.x.round(5), centroids.y.round(5)
    return gdf.sort_values("area_ha", ascending=False)
