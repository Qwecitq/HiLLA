import re, os
from datetime import datetime
import s3fs
import rioxarray as rxr
import xarray as xr
import numpy as np
from tqdm import tqdm

# --- helper to parse time from filename ---
_ts_patterns = [
    re.compile(r"(?P<year>19\d{2}|20\d{2})[._\-]?(?P<month>0[1-9]|1[0-2])$"),
    re.compile(r"(?P<year>19\d{2}|20\d{2})[._\-](?P<month>0[1-9]|1[0-2])"),
    re.compile(r"(?P<year>19\d{2}|20\d{2})(?P<month>0[1-9]|1[0-2])"),
]

def _extract_time_from_name(fname):
    name = os.path.basename(fname).replace(".tif", "").replace(".TIF", "")
    for pat in _ts_patterns:
        m = pat.search(name)
        if m:
            y, mo = int(m.group("year")), int(m.group("month"))
            try:
                return np.datetime64(datetime(y, mo, 1))
            except Exception:
                pass
    # fallback: try any YYYY and MM pair
    m_y = re.search(r"(19|20)\d{2}", name)
    m_m = re.search(r"(0[1-9]|1[0-2])", name)
    if m_y and m_m:
        return np.datetime64(datetime(int(m_y.group(0)), int(m_m.group(0)), 1))
    return None

# --- main function ---
def open_chirps_s3(bucket_prefix="deafrica-input-datasets/rainfall_chirps_monthly/",
                    s3_region="af-south-1",
                    anon=True,
                    bbox=None,             # (lon_min, lon_max, lat_min, lat_max) or None
                    max_files=None,        # limit number for quick tests
                    chunks="auto"):        # pass to rioxarray.open_rasterio for dask chunks
    """
    Open CHIRPS GeoTIFFs from a public S3 prefix and return an xarray.Dataset (lazy, dask-backed).

    Returns
    -------
    xr.Dataset
        DataArray variable named 'precip' (or original name) with dims (time, y, x) or (time, latitude, longitude).
    """
    fs = s3fs.S3FileSystem(anon=anon, client_kwargs={"region_name": s3_region})
    prefix = bucket_prefix.rstrip("/") + "/"
    objs = fs.ls(prefix)
    tif_objs = [o for o in objs if o.lower().endswith(".tif")]
    if not tif_objs:
        raise RuntimeError(f"No .tif files found under s3://{prefix}")

    # build full s3 paths and extract times
    path_time = []
    for o in tif_objs:
        p = "s3://" + o if not o.startswith("s3://") else o
        t = _extract_time_from_name(o)
        if t is None:
            continue
        path_time.append((p, t))
    if not path_time:
        raise RuntimeError("No tif files with parseable timestamps found.")

    # sort by time
    path_time.sort(key=lambda x: x[1])

    data_arrays = []
    for p, t in tqdm(path_time[:max_files] if max_files else path_time, desc="opening tifs lazily"):
        try:
            da = rxr.open_rasterio(p, masked=True, chunks=chunks)  # lazy
            # squeeze single band
            if "band" in da.dims and da.sizes["band"] == 1:
                da = da.squeeze("band", drop=True)
            # assign time coordinate
            da = da.assign_coords(time=[t])
            # optionally subset bbox (best-effort; assumes x/y are lon/lat or coords named lon/latitude)
            if bbox is not None:
                lon_min, lon_max, lat_min, lat_max = bbox
                # common coord names: x/y or longitude/latitude
                if ("x" in da.coords) and ("y" in da.coords):
                    da = da.sel(x=slice(lon_min, lon_max), y=slice(lat_max, lat_min))
                elif ("longitude" in da.coords) and ("latitude" in da.coords):
                    da = da.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
            # ensure a sensible name
            if da.name is None:
                da = da.rename("precip")
            data_arrays.append(da)
        except Exception as e:
            # skip problematic files but report
            print(f"Skipping {p}: {e}")

    if not data_arrays:
        raise RuntimeError("No DataArrays could be opened successfully.")

    combined = xr.concat(data_arrays, dim="time", coords="minimal", data_vars="minimal", compat="override")
    # standardize spatial dim names if possible
    rename_map = {}
    if "x" in combined.dims and "longitude" not in combined.dims:
        rename_map["x"] = "longitude"
    if "y" in combined.dims and "latitude" not in combined.dims:
        rename_map["y"] = "latitude"
    if rename_map:
        combined = combined.rename(rename_map)

    # return as Dataset (DataArray -> Dataset for consistency)
    ds = combined.to_dataset(name=combined.name if combined.name else "precip")
    return ds
