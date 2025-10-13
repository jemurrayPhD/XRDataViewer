"""I/O utilities for XRDataViewer."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import xarray as xr

__all__ = ["open_dataset", "FORCE_XARRAY_ENGINE"]

FORCE_XARRAY_ENGINE: Optional[str] = None


def open_dataset(path: Path) -> xr.Dataset:
    """Open an :class:`xarray.Dataset` from *path*.

    The helper centralises how datasets are opened throughout the application so
    alternative engines (for example, ``netcdf4``) can be forced in one place.
    """

    if path.suffix.lower() == ".zarr" or path.name.lower().endswith(".zarr"):
        return xr.open_zarr(str(path))
    if FORCE_XARRAY_ENGINE:
        return xr.open_dataset(str(path), engine=FORCE_XARRAY_ENGINE)
    return xr.open_dataset(str(path))
