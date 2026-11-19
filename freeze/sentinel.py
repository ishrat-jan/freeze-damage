"""Find and load Sentinel-2 L2A scenes from the Earth Search STAC catalogue.

Steps:
1. search_scenes(): list all scenes for tile 17RMM in a date window.
2. scene_inventory(): one row per date (cloud %, our clear/cloudy label) -> CSV.
3. load_dn(): download the bands for the AOI on our grid (cached as NetCDF).
4. to_reflectance(): DN -> surface reflectance.

BOA offset: since processing baseline 04.00, L2A digital numbers can carry a
+1000 offset. Earth Search has a property "earthsearch:boa_offset_applied", but
it is NOT reliable: several 2025 scenes say False although their DNs clearly
have no offset (most dark pixels are below 1000, which would mean negative
reflectance). So we decide from the data, per date:
    offset present  -> almost no clear-land red DN below 1000 -> (DN - 1000) / 10000
    offset absent   -> DN / 10000
The flag is still recorded, and a warning is printed when data and flag disagree.
"""
import numpy as np
import pandas as pd
import xarray as xr
from odc.stac import load as stac_load
from pystac_client import Client


def search_scenes(cfg, start=None, end=None, dates=None):
    """STAC items on tile 17RMM covering the AOI.

    start/end: date window (defaults to the main window in config.yaml).
    dates: optional list of 'YYYY-MM-DD'; keep only scenes from these days.
    """
    s2 = cfg["sentinel2"]
    start, end = start or s2["start_date"], end or s2["end_date"]
    client = Client.open(s2["stac_url"])
    search = client.search(
        collections=[s2["collection"]],
        bbox=cfg["aoi"]["bbox"],
        datetime=f"{start}/{end}",
        query={"grid:code": {"eq": s2["mgrs_tile"]}},
    )
    items = list(search.items())
    if dates is not None:
        items = [it for it in items if _day(it) in set(dates)]
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes found for {start}..{end} - check config.yaml")
    return items


def _day(item):
    return item.datetime.strftime("%Y-%m-%d")


def flags_by_date(items, cfg):
    """Earth Search offset flag per date (True only if every scene of that day says True)."""
    flags = {}
    for it in items:
        flag = bool(it.properties.get(cfg["sentinel2"]["offset_flag"], False))
        flags[_day(it)] = flags.get(_day(it), True) and flag
    return flags


def detect_offsets(ds, cfg):
    """Per-date BOA offset decided from the data (see module docstring)."""
    s2 = cfg["sentinel2"]
    clear = ds.scl.isin(s2["clear_scl_classes"]) & (ds.red > 0)
    below = ((ds.red < -s2["boa_offset"]) & clear).sum(["y", "x"]) / clear.sum(["y", "x"]).clip(min=1)
    has_offset = below.values < s2["offset_check_max_below"]
    offsets = np.where(has_offset, s2["boa_offset"], 0).astype("int16")
    for day, off, flag in zip(pd.to_datetime(ds.time.values).strftime("%Y-%m-%d"), offsets, ds.boa_flag.values):
        if (off == 0) != bool(flag):
            print(f"  WARNING {day}: flag {s2['offset_flag']}={bool(flag)}, but data says "
                  f"offset {'absent' if off == 0 else 'present'} -> using the data")
    return offsets


def scene_inventory(items, cfg):
    """Table of acquisitions: date, cloud cover, and how the pipeline uses each date."""
    d = cfg["dates"]
    use = {day: "baseline" for day in d["baseline"]}
    use[d["damage_early"]] = "damage (early)"
    use[d["damage_late"]] = "damage (late)"
    for day in d["clear_not_used"]:
        use[day] = "not used (clear, during freeze)"
    for day in d["cloudy"]:
        use[day] = "not used (cloudy)"

    rows = []
    for it in items:
        rows.append({
            "date": it.datetime.strftime("%Y-%m-%d"),
            "item_id": it.id,
            "tile_cloud_pct": round(it.properties.get("eo:cloud_cover", float("nan")), 1),
            "tile_nodata_pct": round(it.properties.get("s2:nodata_pixel_percentage", float("nan")), 1),
            "es_offset_flag": bool(it.properties.get(cfg["sentinel2"]["offset_flag"], False)),
        })
    df = pd.DataFrame(rows).sort_values(["date", "item_id"])
    # Several items can exist for the same day (reprocessing); show one row per day.
    df = df.groupby("date", as_index=False).agg(
        n_items=("item_id", "count"),
        tile_cloud_pct=("tile_cloud_pct", "min"),
        tile_nodata_pct=("tile_nodata_pct", "min"),
        es_offset_flag=("es_offset_flag", "min"),
    )
    df["use"] = df["date"].map(use).fillna("not used")
    out = cfg["run"]["out"] / "tables" / "scene_inventory.csv"
    df.to_csv(out, index=False)
    return df


def load_dn(cfg, items=None, cache_file=None):
    """Load raw digital numbers for all dates on the analysis grid.

    The result is saved as compressed NetCDF in the cache folder; if that file
    exists it is reused, so the (slow) download happens only once per resolution.
    The Earth Search offset flag of each date is stored with it (coordinate "boa_flag").
    """
    s2 = cfg["sentinel2"]
    cache = cfg["run"]["cache"] / (cache_file or s2["cache_file"])
    if cache.exists():
        print(f"  using cached Sentinel-2 data: {cache}")
        return xr.open_dataset(cache, engine="h5netcdf", decode_coords="all").load()

    if items is None:
        items = search_scenes(cfg)
    resampling = {b: s2["resampling_reflectance"] for b in s2["bands"]}
    resampling["scl"] = s2["resampling_scl"]

    ds = stac_load(
        items,
        bands=s2["bands"],
        bbox=cfg["aoi"]["bbox"],
        crs=cfg["run"]["crs"],
        resolution=cfg["run"]["res"],
        groupby="solar_day",          # merge tiles/duplicates from the same day
        chunks={"x": s2["chunk_size"], "y": s2["chunk_size"]},
        resampling=resampling,
        fail_on_error=True,
    )
    print(f"  downloading {len(ds.time)} dates, grid {ds.sizes['y']} x {ds.sizes['x']} pixels ...")
    ds = ds.compute()
    # Use plain dates (YYYY-MM-DD) as the time labels, matching config.yaml.
    ds = ds.assign_coords(time=pd.to_datetime(ds.time.values).normalize())
    flags = flags_by_date(items, cfg)
    days = pd.to_datetime(ds.time.values).strftime("%Y-%m-%d")
    ds = ds.assign_coords(boa_flag=("time", np.array([flags[d] for d in days])))

    encoding = {b: {"zlib": True, "complevel": 4} for b in s2["bands"]}
    # Write to a temp file first, so a failed save never leaves a broken cache.
    tmp = cache.with_suffix(".tmp")
    ds.to_netcdf(tmp, engine="h5netcdf", encoding=encoding)
    tmp.rename(cache)
    print(f"  saved cache: {cache}")
    return ds


def to_reflectance(ds, cfg):
    """Convert DN to reflectance for the optical bands. DN 0 = nodata -> NaN."""
    s2 = cfg["sentinel2"]
    offset = xr.DataArray(detect_offsets(ds, cfg).astype("float32"), dims="time")   # 0 or -1000
    out = xr.Dataset(coords=ds.coords, attrs=ds.attrs)
    out = out.assign_coords(boa_offset=offset)
    for b in s2["bands"]:
        if b == "scl":
            out[b] = ds[b]
        else:
            # float32 halves memory vs the default float64 (matters at 10 m)
            band = ds[b].astype("float32")
            out[b] = (band.where(band > 0) + offset) / np.float32(s2["scale"])
    return out


def select_date(da, day):
    """Pick one date (string 'YYYY-MM-DD') from a time series."""
    return da.sel(time=pd.Timestamp(day))
