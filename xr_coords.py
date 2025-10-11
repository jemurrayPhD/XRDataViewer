
from __future__ import annotations
import numpy as np

def guess_phys_coords(da) -> dict:
    """
    Inspect a 2D xarray.DataArray and try to extract physical coordinates:
      - 1D coords aligned with each axis -> keys 'x','y'
      - 2D coords on both dims -> keys 'X','Y'
    """
    out = {}
    if getattr(da, "ndim", 0) != 2:
        return out
    ydim, xdim = da.dims[-2], da.dims[-1]
    Ny, Nx = da.sizes[ydim], da.sizes[xdim]
    # 2D coords
    for key, coord in da.coords.items():
        try:
            if coord.ndim == 2 and coord.dims == (ydim, xdim):
                if key.lower().startswith('x'):
                    out["X"] = np.asarray(coord.values, float)
                elif key.lower().startswith('y'):
                    out["Y"] = np.asarray(coord.values, float)
        except Exception:
            pass
    if "X" in out and "Y" in out:
        return out
    # 1D coords
    if xdim in da.coords and da.coords[xdim].ndim == 1 and da.sizes[xdim] == Nx:
        out["x"] = np.asarray(da.coords[xdim].values, float)
    if ydim in da.coords and da.coords[ydim].ndim == 1 and da.sizes[ydim] == Ny:
        out["y"] = np.asarray(da.coords[ydim].values, float)
    # common alternates
    if "x" not in out:
        for cand in ("x","X","lon","longitude","cols","column"):
            if cand in da.coords and da.coords[cand].ndim == 1 and da.coords[cand].sizes.get(cand, Nx) == Nx:
                out["x"] = np.asarray(da.coords[cand].values, float); break
    if "y" not in out:
        for cand in ("y","Y","lat","latitude","rows","row"):
            if cand in da.coords and da.coords[cand].ndim == 1 and da.coords[cand].sizes.get(cand, Ny) == Ny:
                out["y"] = np.asarray(da.coords[cand].values, float); break
    return out
