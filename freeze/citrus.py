"""Citrus mask from the USDA Cropland Data Layer (CDL), via the CropScape API.

1. Convert the AOI (lon/lat) to the CDL projection (EPSG:5070). All 4 corners
   are transformed, because a lon/lat rectangle is not a rectangle in Albers.
2. Ask GetCDLFile for that box. The server answers with XML containing a
   temporary download link, which expires quickly, so we download it at once.
3. Cache the GeoTIFF, reproject it onto the NDVI grid (nearest neighbour, so
   class codes are never mixed), and keep the citrus class.
"""
import re
import time

import requests
import rioxarray
from odc.geo.xr import xr_reproject
from pyproj import Transformer


def aoi_bbox_in_cdl_crs(cfg):
    """AOI bbox in EPSG:5070, built from all four transformed corners."""
    w, s, e, n = cfg["aoi"]["bbox"]
    tr = Transformer.from_crs("EPSG:4326", cfg["citrus"]["cdl_crs"], always_xy=True)
    xs, ys = tr.transform([w, e, e, w], [s, s, n, n])
    return min(xs), min(ys), max(xs), max(ys)


def _get_with_retries(url, cfg, **kwargs):
    """GET with timeout and retries (the CropScape server is slow and flaky)."""
    c = cfg["citrus"]
    for attempt in range(1, c["retries"] + 1):
        try:
            r = requests.get(url, timeout=c["timeout_s"], **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException as err:
            print(f"  attempt {attempt}/{c['retries']} failed: {err}")
            if attempt == c["retries"]:
                raise
            time.sleep(c["retry_wait_s"])


def download_cdl(cfg):
    """Download the CDL clip for the AOI (cached). Returns the local file path."""
    c = cfg["citrus"]
    path = cfg["run"]["cache"].parent / f"cdl_{c['cdl_year']}.tif"   # same file for all resolutions
    if path.exists():
        print(f"  using cached CDL: {path}")
        return path

    bbox = ",".join(f"{v:.0f}" for v in aoi_bbox_in_cdl_crs(cfg))
    url = f"{c['api_url']}?year={c['cdl_year']}&bbox={bbox}"
    for attempt in range(1, c["retries"] + 1):
        xml = _get_with_retries(url, cfg).text
        match = re.search(r"<returnURL>(.*?)</returnURL>", xml)
        if not match:
            raise RuntimeError(f"CropScape gave no download link. Response:\n{xml[:500]}")
        try:
            # Download straight away: the link expires within minutes.
            data = _get_with_retries(match.group(1), cfg).content
            break
        except requests.RequestException:
            # Link may have expired while retrying -> ask for a fresh one.
            if attempt == c["retries"]:
                raise
            print("  download link failed, requesting a new one ...")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.rename(path)
    print(f"  saved CDL: {path} ({len(data) / 1e6:.1f} MB)")
    return path


def cdl_on_grid(template, cfg):
    """All CDL crop classes on the same grid as `template` (an NDVI DataArray)."""
    cdl = rioxarray.open_rasterio(download_cdl(cfg)).squeeze("band", drop=True)
    return xr_reproject(cdl, template.odc.geobox, resampling="nearest").rename("cdl")


def citrus_mask(cdl, cfg):
    """Boolean citrus mask from the gridded CDL."""
    return (cdl == cfg["citrus"]["cdl_class"]).rename("citrus")
