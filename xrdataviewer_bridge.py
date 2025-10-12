"""Utilities to register xarray results with XRDataViewer interactive sessions."""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Optional

import urllib.error
import urllib.request

import xarray as xr

__all__ = ["register_dataset", "register_dataarray", "register"]


def _bridge_url() -> str:
    url = os.environ.get("XRVIEWER_BRIDGE_URL")
    if not url:
        raise RuntimeError(
            "XRVIEWER_BRIDGE_URL is not set. Launch XRDataViewer and open the Interactive Processing tab first."
        )
    return url


def _send_payload(payload: dict, timeout: float) -> str:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _bridge_url(), data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.URLError as exc:  # pragma: no cover - network errors depend on runtime env
        raise RuntimeError(f"Failed to contact XRDataViewer bridge: {exc}") from exc
    try:
        reply = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("Received invalid response from XRDataViewer bridge") from exc
    if not reply.get("ok"):
        raise RuntimeError(str(reply.get("error") or "Bridge request failed"))
    return str(reply.get("label") or "")


def register_dataset(dataset: xr.Dataset, label: Optional[str] = None, timeout: float = 30.0) -> str:
    """Send an :class:`xarray.Dataset` to XRDataViewer without writing it to disk."""

    if not isinstance(dataset, xr.Dataset):
        raise TypeError("register_dataset expects an xarray.Dataset instance")
    chosen_label = label or str(
        dataset.attrs.get("title") or dataset.attrs.get("name") or "interactive_dataset"
    )
    buffer = io.BytesIO()
    dataset.to_netcdf(buffer)
    payload = {
        "kind": "dataset",
        "label": chosen_label,
        "payload": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }
    return _send_payload(payload, timeout)


def register_dataarray(dataarray: xr.DataArray, label: Optional[str] = None, timeout: float = 30.0) -> str:
    """Send an :class:`xarray.DataArray` to XRDataViewer as an in-memory dataset."""

    if not isinstance(dataarray, xr.DataArray):
        raise TypeError("register_dataarray expects an xarray.DataArray instance")
    var_name = dataarray.name or "variable"
    chosen_label = label or str(dataarray.name or "interactive_array")
    dataset = dataarray.to_dataset(name=var_name)
    buffer = io.BytesIO()
    dataset.to_netcdf(buffer)
    payload = {
        "kind": "dataarray",
        "label": chosen_label,
        "var_name": var_name,
        "payload": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }
    return _send_payload(payload, timeout)


def register(data, label: Optional[str] = None, timeout: float = 30.0) -> str:
    """Register either a Dataset or DataArray with XRDataViewer."""

    if isinstance(data, xr.Dataset):
        return register_dataset(data, label=label, timeout=timeout)
    if isinstance(data, xr.DataArray):
        return register_dataarray(data, label=label, timeout=timeout)
    raise TypeError("register expects an xarray Dataset or DataArray")
