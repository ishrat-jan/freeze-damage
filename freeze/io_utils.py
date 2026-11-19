"""Small helpers for saving results (GeoTIFFs) with correct georeferencing."""
import numpy as np
import rioxarray  # noqa: F401  (adds the .rio accessor to xarray)


def save_geotiff(da, path, nodata=None):
    """Save a 2-D DataArray as a compressed GeoTIFF.

    Float rasters use NaN as nodata; integer rasters need an explicit nodata value.
    """
    da = da.squeeze(drop=True).copy()
    # Drop leftover NetCDF settings (e.g. _FillValue from the cache file)
    da.encoding = {}
    da.attrs = {k: v for k, v in da.attrs.items() if k not in ("_FillValue", "nodata")}
    if "spatial_ref" in da.coords:
        crs = da.spatial_ref.attrs.get("crs_wkt")
    else:
        crs = da.odc.crs.to_wkt()
    da = da.rio.write_crs(crs)
    if nodata is None and np.issubdtype(da.dtype, np.floating):
        nodata = np.nan
    if nodata is not None:
        da = da.rio.write_nodata(nodata)
    da.rio.to_raster(path, compress="deflate")
    return path
