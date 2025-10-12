"""Utilities to register xarray results with XRDataViewer interactive sessions."""

from __future__ import annotations

import base64
import json
import os
import threading
import warnings
from typing import Dict, Mapping, Optional, Tuple

import urllib.error
import urllib.request

import xarray as xr

try:  # pragma: no cover - optional dependency inside notebooks only
    from IPython import get_ipython  # type: ignore
except Exception:  # pragma: no cover - IPython not available outside notebooks
    def get_ipython():  # type: ignore
        return None

__all__ = [
    "register_dataset",
    "register_dataarray",
    "register",
    "sync_namespace",
    "enable_auto_sync",
    "disable_auto_sync",
]


_AUTO_TRACK: Dict[int, Tuple[str, str]] = {}
_AUTO_LOCK = threading.Lock()
_AUTO_HOOK: Optional[Tuple[object, object]] = None
_AUTO_TIMEOUT = 30.0
_AUTO_INCLUDE_PRIVATE = False


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
    data_bytes = dataset.to_netcdf()
    payload = {
        "kind": "dataset",
        "label": chosen_label,
        "payload": base64.b64encode(data_bytes).decode("ascii"),
    }
    return _send_payload(payload, timeout)


def register_dataarray(dataarray: xr.DataArray, label: Optional[str] = None, timeout: float = 30.0) -> str:
    """Send an :class:`xarray.DataArray` to XRDataViewer as an in-memory dataset."""

    if not isinstance(dataarray, xr.DataArray):
        raise TypeError("register_dataarray expects an xarray.DataArray instance")
    var_name = dataarray.name or "variable"
    chosen_label = label or str(dataarray.name or "interactive_array")
    dataset = dataarray.to_dataset(name=var_name)
    data_bytes = dataset.to_netcdf()
    payload = {
        "kind": "dataarray",
        "label": chosen_label,
        "var_name": var_name,
        "payload": base64.b64encode(data_bytes).decode("ascii"),
    }
    return _send_payload(payload, timeout)


def register(data, label: Optional[str] = None, timeout: float = 30.0) -> str:
    """Register either a Dataset or DataArray with XRDataViewer."""

    if isinstance(data, xr.Dataset):
        return register_dataset(data, label=label, timeout=timeout)
    if isinstance(data, xr.DataArray):
        return register_dataarray(data, label=label, timeout=timeout)
    raise TypeError("register expects an xarray Dataset or DataArray")


def _normalise_label(candidate: str, fallback: str) -> str:
    candidate = (candidate or "").strip()
    return candidate or fallback


def sync_namespace(
    namespace: Mapping[str, object],
    *,
    include_private: bool = False,
    timeout: float = 30.0,
) -> Dict[str, str]:
    """Register all xarray objects from *namespace* with XRDataViewer.

    Parameters
    ----------
    namespace:
        Mapping containing variables to scan (e.g. ``globals()`` or an IPython
        ``user_ns``).
    include_private:
        Whether to include variables whose names start with an underscore.
    timeout:
        Bridge request timeout in seconds.

    Returns
    -------
    dict
        Mapping from variable names to the labels assigned by XRDataViewer for
        the registered objects.
    """

    registered: Dict[str, str] = {}
    seen_ids: set[int] = set()

    with _AUTO_LOCK:
        for name, value in namespace.items():
            if not include_private and str(name).startswith("_"):
                continue

            if isinstance(value, xr.Dataset):
                label = _normalise_label(str(name), "interactive_dataset")
                obj_id = id(value)
                seen_ids.add(obj_id)
                previous = _AUTO_TRACK.get(obj_id)
                if previous and previous == (label, "dataset"):
                    continue
                assigned = register_dataset(value, label=label, timeout=timeout)
                _AUTO_TRACK[obj_id] = (label, "dataset")
                registered[str(name)] = assigned
            elif isinstance(value, xr.DataArray):
                var_name = value.name or str(name) or "variable"
                label = _normalise_label(str(name), str(var_name) or "interactive_array")
                obj_id = id(value)
                seen_ids.add(obj_id)
                previous = _AUTO_TRACK.get(obj_id)
                if previous and previous == (label, var_name):
                    continue
                assigned = register_dataarray(value, label=label, timeout=timeout)
                _AUTO_TRACK[obj_id] = (label, var_name)
                registered[str(name)] = assigned

        # Drop references to objects that are no longer present so Python can reuse ids safely
        stale = [obj_id for obj_id in _AUTO_TRACK.keys() if obj_id not in seen_ids]
        for obj_id in stale:
            _AUTO_TRACK.pop(obj_id, None)

    return registered


def enable_auto_sync(*, include_private: bool = False, timeout: float = 30.0) -> bool:
    """Mirror xarray objects from the active IPython namespace automatically."""

    shell = get_ipython()
    if shell is None:
        raise RuntimeError("enable_auto_sync requires an active IPython shell")

    global _AUTO_HOOK, _AUTO_TIMEOUT, _AUTO_INCLUDE_PRIVATE

    if _AUTO_HOOK is not None:
        return False

    events = getattr(shell, "events", None)
    if events is None:
        raise RuntimeError("IPython shell does not expose events API; cannot enable auto sync")

    _AUTO_TIMEOUT = timeout
    _AUTO_INCLUDE_PRIVATE = include_private

    def _autosync_hook(*_args, **_kwargs):
        try:
            sync_namespace(
                getattr(shell, "user_ns", {}),
                include_private=_AUTO_INCLUDE_PRIVATE,
                timeout=_AUTO_TIMEOUT,
            )
        except Exception as exc:  # pragma: no cover - defensive logging inside notebooks
            warnings.warn(f"XRDataViewer auto-sync failed: {exc}")

    events.register("post_run_cell", _autosync_hook)
    _AUTO_HOOK = (events, _autosync_hook)

    # Initial sync so existing variables appear immediately
    sync_namespace(getattr(shell, "user_ns", {}), include_private=include_private, timeout=timeout)
    return True


def disable_auto_sync() -> bool:
    """Disable automatic namespace mirroring for the current IPython shell."""

    global _AUTO_HOOK
    if _AUTO_HOOK is None:
        return False

    events, hook = _AUTO_HOOK
    try:
        events.unregister("post_run_cell", hook)
    except Exception:  # pragma: no cover - defensive cleanup
        pass
    _AUTO_HOOK = None
    return True
