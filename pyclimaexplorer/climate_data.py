

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import streamlit as st



VARIABLE_META = {
    "t2m": {
        "label":    "2m Temperature",
        "unit":     "°C",
        "unit_raw": "K",
        "colorscale": "RdBu_r",
        "convert":  lambda x: x - 273.15,
        "icon":     "🌡️",
    },
    "tp": {
        "label":    "Total Precipitation",
        "unit":     "mm/day",
        "unit_raw": "mm/d",
        "colorscale": "Blues",
        "convert":  lambda x: x,
        "icon":     "🌧️",
    },
    "u10": {
        "label":    "10m Wind Speed",
        "unit":     "m/s",
        "unit_raw": "m/s",
        "colorscale": "Viridis",
        "convert":  lambda x: x,
        "icon":     "💨",
    },
}


STORY_PRESETS = {
    "🌍 Overview – Modern Warming": {
        "desc": "Global surface temperature anomaly over the full record. "
                "Notice the accelerating warming trend post-2000.",
        "variable": "t2m",
        "year": 2023,
        "month": 7,
        "lat": 0.0,
        "lon": 0.0,
        "anomaly": True,
    },
    "🔥 1998 El Niño – Record Heat": {
        "desc": "The 1997-98 El Niño was one of the strongest on record. "
                "July 1998 shows dramatic warming across the tropical Pacific.",
        "variable": "t2m",
        "year": 1998,
        "month": 7,
        "lat": 0.0,
        "lon": -140.0,
        "anomaly": True,
    },
    "❄️ Arctic Amplification": {
        "desc": "The Arctic is warming 2-4× faster than the global average. "
                "Compare high-latitude warming vs. tropical regions.",
        "variable": "t2m",
        "year": 2020,
        "month": 9,
        "lat": 75.0,
        "lon": 0.0,
        "anomaly": True,
    },
    "🌧️ ITCZ Precipitation Band": {
        "desc": "The Intertropical Convergence Zone (ITCZ) drives intense "
                "rainfall near the equator. Watch it shift seasonally.",
        "variable": "tp",
        "year": 2010,
        "month": 8,
        "lat": 5.0,
        "lon": -60.0,
        "anomaly": False,
    },
    "🌬️ Mid-Latitude Jet Streams": {
        "desc": "Strong westerly winds form the jet streams around 45° latitude. "
                "Wind speed peaks in winter months.",
        "variable": "u10",
        "year": 2005,
        "month": 1,
        "lat": 45.0,
        "lon": -30.0,
        "anomaly": False,
    },
}


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> xr.Dataset:
    """Load and lightly normalise a NetCDF file."""
    ds = xr.open_dataset(path, engine="netcdf4")

    
    rename = {}
    for old, new in [("lat", "latitude"), ("lon", "longitude")]:
        if old in ds.coords and new not in ds.coords:
            rename[old] = new
    if rename:
        ds = ds.rename(rename)

    
    if "latitude" in ds.coords and ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))

    return ds


@st.cache_data(show_spinner=False)
def get_time_index(_ds: xr.Dataset) -> pd.DatetimeIndex:
    """Return a DatetimeIndex regardless of the underlying time encoding."""
    try:
        times = xr.decode_cf(_ds).indexes["time"]
        return pd.DatetimeIndex(times)
    except Exception:
        
        raw = _ds["time"].values
        origin = pd.Timestamp("1990-01-01")
        return pd.DatetimeIndex([origin + pd.Timedelta(days=float(d)) for d in raw])


def get_variables(ds: xr.Dataset) -> list[str]:
    
    spatial = []
    for name, var in ds.data_vars.items():
        dims = set(var.dims)
        has_spatial = (
            ("latitude" in dims or "lat" in dims) and
            ("longitude" in dims or "lon" in dims)
        )
        if has_spatial:
            spatial.append(name)
    return spatial


def get_var_meta(var: str) -> dict:
    
    if var in VARIABLE_META:
        return VARIABLE_META[var]
    return {
        "label":      var,
        "unit":       "",
        "unit_raw":   "",
        "colorscale": "Viridis",
        "convert":    lambda x: x,
        "icon":       "📊",
    }


@st.cache_data(show_spinner=False)
def get_spatial_slice(
    _ds: xr.Dataset,
    var: str,
    time_idx: int,
    anomaly: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    arr = _ds[var].isel(time=time_idx).values.astype(np.float32)
    lats = _ds["latitude"].values
    lons = _ds["longitude"].values

    meta = get_var_meta(var)
    arr  = meta["convert"](arr)

    if anomaly:
        clim = get_climatology(_ds, var)
       
        times = get_time_index(_ds)
        month = times[time_idx].month
        arr   = arr - clim[month - 1]

    return lats, lons, arr


@st.cache_data(show_spinner=False)
def get_time_series(
    _ds: xr.Dataset,
    var: str,
    lat: float,
    lon: float,
    anomaly: bool = False,
) -> pd.Series:
    
    da   = _ds[var]
    meta = get_var_meta(var)

    
    data = da.sel(latitude=lat, longitude=lon, method="nearest").values.astype(np.float64)
    data = meta["convert"](data)

    times = get_time_index(_ds)
    ts    = pd.Series(data, index=times, name=meta["label"])

    if anomaly:
        clim = get_climatology(_ds, var)
        months = times.month
        baseline = np.array([clim[m - 1] for m in months])
        
        lat_idx = int(np.argmin(np.abs(_ds["latitude"].values - lat)))
        lon_idx = int(np.argmin(np.abs(_ds["longitude"].values - lon)))
        ts = ts - pd.Series(
            [clim[m - 1][lat_idx, lon_idx] for m in months],
            index=times
        )
        ts.name = f"{meta['label']} anomaly"

    return ts


@st.cache_data(show_spinner=False)
def get_climatology(_ds: xr.Dataset, var: str) -> list[np.ndarray]:
    """Return list of 12 monthly mean 2D arrays (climatological baseline)."""
    times = get_time_index(_ds)
    meta  = get_var_meta(var)
    clim  = []
    for m in range(1, 13):
        idx   = [i for i, t in enumerate(times) if t.month == m]
        data  = _ds[var].isel(time=idx).mean(dim="time").values.astype(np.float32)
        data  = meta["convert"](data)
        clim.append(data)
    return clim


@st.cache_data(show_spinner=False)
def get_global_mean_series(_ds: xr.Dataset, var: str) -> pd.Series:
    
    da    = _ds[var]
    lats  = np.deg2rad(_ds["latitude"].values)
    wts   = np.cos(lats)
    wts  /= wts.sum()
    meta  = get_var_meta(var)

    data  = da.values.astype(np.float64)          # (T, lat, lon)
    data  = meta["convert"](data)
    gm    = (data * wts[np.newaxis, :, np.newaxis]).mean(axis=(1, 2))
    times = get_time_index(_ds)
    return pd.Series(gm, index=times, name=f"Global mean {meta['label']}")
