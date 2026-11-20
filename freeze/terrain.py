"""Frost-prone areas from terrain (Copernicus DEM GLO-30).

Cold air is heavy: on calm freezing nights it drains downhill and pools in
low spots.

MAIN RESULT - low-lying index = elevation - mean elevation within a radius
(300 m). Negative values = lower than the surroundings (a hollow / frost pocket).
We report the citrus damage rate per index bin, and map hollows.

Frost-prone map classes (land pixels, useful for planning new plantings):
    0 not low-lying
    1 hollow       (deep_hollow_m <= index < hollow_m, e.g. 0-4 m below surroundings)
    2 deep hollow  (index < deep_hollow_m, e.g. more than 4 m below surroundings)
    255 no data or water

SUPPLEMENTARY - damage rate per absolute elevation band (table/chart only).
"""
import numpy as np
import pandas as pd
import xarray as xr
from odc.stac import configure_rio, load as stac_load
from pystac_client import Client
from scipy import ndimage

from freeze.damage import NODATA

FROST_NAMES = ["not low-lying", "hollow", "deep hollow"]   # frost-prone classes 0, 1, 2


def load_dem(template, cfg):
    """DEM (metres) on the NDVI grid, cached as NetCDF."""
    cache = cfg["run"]["cache"] / cfg["terrain"]["cache_file"]
    if cache.exists():
        print(f"  using cached DEM: {cache}")
        return xr.open_dataarray(cache, engine="h5netcdf").load()

    client = Client.open(cfg["sentinel2"]["stac_url"])
    items = list(client.search(collections=[cfg["terrain"]["dem_collection"]],
                               bbox=cfg["aoi"]["bbox"]).items())
    print(f"  downloading DEM ({len(items)} tiles) ...")
    # DEM files are s3:// links to a public bucket: read them without AWS credentials.
    s2 = cfg["sentinel2"]   # same web-read retry settings as for Sentinel-2
    configure_rio(cloud_defaults=True, aws={"aws_unsigned": True},
                  GDAL_HTTP_MAX_RETRY=str(s2["gdal_http_retries"]),
                  GDAL_HTTP_RETRY_DELAY=str(s2["gdal_http_retry_delay_s"]))
    ds = stac_load(items, bands=["data"], geobox=template.odc.geobox, resampling="bilinear")
    dem = ds["data"].squeeze("time", drop=True).astype("float32").rename("elevation")
    tmp = cache.with_suffix(".tmp")
    dem.to_netcdf(tmp, engine="h5netcdf")
    tmp.rename(cache)
    return dem


def lowlying_index(dem, cfg):
    """Elevation minus the local mean elevation (square window of +/- radius)."""
    px = int(round(cfg["terrain"]["lowlying_radius_m"] / cfg["run"]["res"]))
    local_mean = ndimage.uniform_filter(dem.values, size=2 * px + 1, mode="nearest")
    return xr.DataArray(dem.values - local_mean, coords=dem.coords, dims=dem.dims, name="lowlying_index")


def elevation_bands(dem, cfg):
    """Lower edge of each pixel's elevation band (e.g. 30, 35, 40 ... m)."""
    w = cfg["terrain"]["elevation_band_m"]
    return (np.floor(dem / w) * w).rename("elevation_band")


def _rate_table(groups, damaged, in_kill, valid, ha, mean_rate):
    """Citrus damage and kill-zone rates for each group label."""
    df = pd.DataFrame({"group": groups[valid], "damaged": damaged[valid], "kill": in_kill[valid]})
    t = df.groupby("group", sort=False).agg(citrus_pixels=("damaged", "size"),
                                            damage_rate=("damaged", "mean"),
                                            kill_zone_rate=("kill", "mean"))
    t["citrus_ha"] = (t.citrus_pixels * ha).round(2)
    t["ratio_to_mean"] = (t.damage_rate / mean_rate).round(2) if mean_rate > 0 else np.nan
    t[["damage_rate", "kill_zone_rate"]] = t[["damage_rate", "kill_zone_rate"]].round(3)
    return t.reset_index()


def lowlying_labels(index, bins):
    """Bin labels like '< -4 m', '-4 to -2 m', ..., '>= 2 m' for each pixel."""
    edges = [-np.inf] + list(bins) + [np.inf]
    names = [f"< {bins[0]} m"] + [f"{a} to {b} m" for a, b in zip(bins[:-1], bins[1:])] + [f">= {bins[-1]} m"]
    idx = np.digitize(index, bins)                    # 0 .. len(bins)
    return np.array(names, dtype=object)[idx], names, edges


def frost_analysis(dem, dmg, kill, citrus, water, cfg):
    """Damage by low-lying index (main), by elevation band (supplementary),
    and the frost-prone class raster.

    dmg: damage classes on the chosen date; kill: kill-zone ID raster;
    water: boolean water mask (lakes are excluded from the frost-prone map).
    """
    t = cfg["terrain"]
    ha = cfg["run"]["pixel_ha"]
    low = lowlying_index(dem, cfg)
    li = low.values

    valid = (dmg.values != NODATA) & citrus.values & ~np.isnan(li)
    damaged = (dmg.values >= 1) & (dmg.values != NODATA)
    in_kill = kill.values > 0
    mean_rate = damaged[valid].mean() if valid.any() else 0.0

    # --- main table: damage rate per low-lying index bin (ordered low -> high)
    labels, names, _ = lowlying_labels(li, t["lowlying_bins"])
    by_low = _rate_table(labels, damaged, in_kill, valid, ha, mean_rate)
    by_low = by_low.rename(columns={"group": "lowlying_index"})
    by_low["order"] = by_low.lowlying_index.map({n: i for i, n in enumerate(names)})
    by_low = by_low.sort_values("order").drop(columns="order")

    # --- supplementary table: damage rate per elevation band
    bands = elevation_bands(dem, cfg).values
    by_band = _rate_table(bands, damaged, in_kill, valid, ha, mean_rate)
    by_band = by_band.rename(columns={"group": "band_from_m"}).sort_values("band_from_m")
    by_band["band_to_m"] = by_band.band_from_m + t["elevation_band_m"]

    # --- frost-prone raster from the low-lying index only
    fp = np.zeros(li.shape, dtype="uint8")
    fp[(li < t["hollow_m"]) & (li >= t["deep_hollow_m"])] = 1
    fp[li < t["deep_hollow_m"]] = 2
    fp[np.isnan(li) | water.values] = NODATA
    frost = xr.DataArray(fp, coords=dem.coords, dims=dem.dims, name="frost_prone")

    # --- damage rate per frost-prone class (same classes as the map)
    names = np.array(FROST_NAMES + ["no data"], dtype=object)
    by_class = _rate_table(names[np.minimum(fp, len(FROST_NAMES))], damaged, in_kill, valid, ha, mean_rate)
    by_class = by_class.rename(columns={"group": "frost_class"})
    by_class["order"] = by_class.frost_class.map({n: i for i, n in enumerate(FROST_NAMES)})
    by_class = by_class.dropna(subset=["order"]).sort_values("order").drop(columns="order")

    out = cfg["run"]["out"] / "tables"
    by_low.to_csv(out / "damage_by_lowlying.csv", index=False)
    by_class.to_csv(out / "damage_by_frost_class.csv", index=False)
    by_band.to_csv(out / "damage_by_elevation_band.csv", index=False)
    return by_band, by_low, by_class, frost, low
