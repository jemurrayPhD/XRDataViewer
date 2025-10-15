"""Scientific colormap registration helpers.

This module vendors a curated subset of Fabio Crameri's scientific colour maps
so they are available through :mod:`pyqtgraph` even when the optional
``cmcrameri`` dependency is not installed.  The colour definitions are
approximations chosen to preserve the perceptual intent of the originals while
remaining compact enough for inclusion directly in the source tree.

The :func:`register_scientific_colormaps` function can be called during
application start-up to ensure the maps are registered exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:  # pragma: no cover - pyqtgraph is part of the runtime, not tests
    import pyqtgraph as pg
except ModuleNotFoundError:  # pragma: no cover
    pg = None  # type: ignore[assignment]


def _patch_colormap_menu() -> None:
    """Ensure ``ColorMapMenu`` can build nested menus on all Qt bindings.

    Older PySide2 releases sometimes return a ``QWidget`` (or ``QAction``)
    from :func:`QMenu.addMenu` when invoked with a string title.  PyQtGraph's
    colour map helper expects a real :class:`QMenu`, so the mismatch bubbles up
    as an ``AttributeError`` during HistogramLUT initialisation.  To keep the
    bundled scientific maps compatible with those environments we temporarily
    wrap :func:`QMenu.addMenu` while the menu is being constructed and coerce
    the return value back to a menu when necessary.
    """

    if pg is None:  # pragma: no cover - nothing to patch without pyqtgraph
        return

    try:  # pragma: no cover - depends on optional runtime components
        from pyqtgraph.widgets import ColorMapMenu
        from pyqtgraph.Qt import QtWidgets
    except Exception:
        return

    patched_flag = "_xr_colormap_menu_patched"
    if getattr(ColorMapMenu.ColorMapMenu, patched_flag, False):
        return

    original_init = ColorMapMenu.ColorMapMenu.__init__

    def patched_init(self, *args, **kwargs):
        original_add_menu = QtWidgets.QMenu.addMenu

        def safe_add_menu(menu_self, *inner_args, **inner_kwargs):
            result = original_add_menu(menu_self, *inner_args, **inner_kwargs)
            if isinstance(result, QtWidgets.QMenu):
                return result

            # PySide2 may yield a QAction/QWidgetAction instead of the submenu.
            action = None
            if isinstance(result, QtWidgets.QAction):
                action = result
            else:
                widget_action_type = getattr(QtWidgets, "QWidgetAction", None)
                if widget_action_type is not None and isinstance(result, widget_action_type):
                    action = result
            if action is not None:
                try:
                    sub_menu = action.menu()
                except Exception:
                    sub_menu = None
                if isinstance(sub_menu, QtWidgets.QMenu):
                    return sub_menu

            # Some bindings even hand back a QWidget; fall back to a manual menu.
            if inner_args and isinstance(inner_args[0], str):
                try:
                    new_menu = QtWidgets.QMenu(inner_args[0], menu_self)
                except Exception:
                    return result
                original_add_menu(menu_self, new_menu)
                return new_menu

            return result

        QtWidgets.QMenu.addMenu = safe_add_menu
        try:
            original_init(self, *args, **kwargs)
        finally:
            QtWidgets.QMenu.addMenu = original_add_menu

    ColorMapMenu.ColorMapMenu.__init__ = patched_init  # type: ignore[assignment]
    setattr(ColorMapMenu.ColorMapMenu, patched_flag, True)


@dataclass(frozen=True)
class _ScientificMap:
    name: str
    stops: Sequence[str]


def _hex_to_rgb_stop(color: str) -> Tuple[int, int, int]:
    """Convert a ``#rrggbb`` colour string to a 0-255 RGB tuple."""

    colour = color.strip().lstrip("#")
    if len(colour) != 6:
        raise ValueError(f"Unsupported colour specification: {color!r}")
    return tuple(int(colour[i : i + 2], 16) for i in (0, 2, 4))


_SCIENTIFIC_COLOURMAPS: Tuple[_ScientificMap, ...] = (
    _ScientificMap(
        "acton",
        (
            "#0b0724",
            "#1f164e",
            "#35296f",
            "#4d3e85",
            "#655697",
            "#7f6fa6",
            "#9a8bb0",
            "#b7a8b3",
            "#d3c7b0",
            "#efe6a8",
        ),
    ),
    _ScientificMap(
        "bamako",
        (
            "#001521",
            "#0f2d3f",
            "#1f465b",
            "#2f6072",
            "#3f7c83",
            "#4f9891",
            "#67b395",
            "#8bcb95",
            "#bcd88d",
            "#f0e689",
        ),
    ),
    _ScientificMap(
        "batlow",
        (
            "#01112d",
            "#052a53",
            "#0f4475",
            "#205c8e",
            "#3475a1",
            "#4d8db0",
            "#6aa4b7",
            "#8fbab4",
            "#bcd0a5",
            "#f5eb70",
        ),
    ),
    _ScientificMap(
        "berlin",
        (
            "#1a0c27",
            "#2d1d4d",
            "#3d326a",
            "#464c7f",
            "#456793",
            "#3f829c",
            "#489e99",
            "#6fb58a",
            "#a3c979",
            "#e2d86b",
        ),
    ),
    _ScientificMap(
        "davos",
        (
            "#041021",
            "#10243f",
            "#1f3a58",
            "#30506e",
            "#446685",
            "#5a7c99",
            "#7392aa",
            "#8ea9b4",
            "#abbfb7",
            "#cad4aa",
        ),
    ),
    _ScientificMap(
        "devon",
        (
            "#160720",
            "#2b1743",
            "#422a61",
            "#5a3f7a",
            "#734e8e",
            "#8d5e9e",
            "#a96fa9",
            "#c581ae",
            "#df96ad",
            "#f6ae9f",
        ),
    ),
    _ScientificMap(
        "glasgow",
        (
            "#021619",
            "#093033",
            "#154b4c",
            "#27665f",
            "#3d8270",
            "#5a9b7b",
            "#7eb27f",
            "#a9c687",
            "#d5d68f",
            "#fbe49d",
        ),
    ),
    _ScientificMap(
        "lapaz",
        (
            "#061a1e",
            "#1b3635",
            "#334f4a",
            "#4b6860",
            "#658075",
            "#7f988a",
            "#9aa096",
            "#b6a394",
            "#d2a085",
            "#ee9a74",
        ),
    ),
    _ScientificMap(
        "lajolla",
        (
            "#2a0a2c",
            "#4b173f",
            "#6d244f",
            "#8f344d",
            "#b2473f",
            "#cd6237",
            "#e28040",
            "#f5a059",
            "#ffd480",
            "#fff8b2",
        ),
    ),
    _ScientificMap(
        "lisbon",
        (
            "#06131b",
            "#0c2733",
            "#133b4c",
            "#1f5164",
            "#2d6878",
            "#3f7f86",
            "#558f8d",
            "#709e8d",
            "#90a989",
            "#b4b07f",
        ),
    ),
    _ScientificMap(
        "nuuk",
        (
            "#150c2d",
            "#241d4c",
            "#312f6b",
            "#3c4186",
            "#44539d",
            "#4967af",
            "#4f7cb9",
            "#5b91bd",
            "#73a6ba",
            "#95b7ac",
        ),
    ),
    _ScientificMap(
        "oleron",
        (
            "#090f27",
            "#182447",
            "#283a65",
            "#3b517e",
            "#536795",
            "#6f7fab",
            "#8b97bd",
            "#a8b0c9",
            "#c5c8cd",
            "#e1e0c7",
        ),
    ),
    _ScientificMap(
        "oslo",
        (
            "#0b141e",
            "#112939",
            "#183f55",
            "#21566f",
            "#2c6d84",
            "#3a8493",
            "#4b9a9b",
            "#5fb0a0",
            "#76c5a2",
            "#91d8a2",
        ),
    ),
    _ScientificMap(
        "roma",
        (
            "#0b0d36",
            "#1a2a68",
            "#204789",
            "#2c649f",
            "#3d81a9",
            "#579cac",
            "#78b3a6",
            "#99c29d",
            "#bebf8f",
            "#dfad7e",
            "#f6956e",
            "#fc705e",
        ),
    ),
    _ScientificMap(
        "tokyo",
        (
            "#0c1d2a",
            "#14343f",
            "#1f4b55",
            "#2c6269",
            "#39797a",
            "#478f86",
            "#58a690",
            "#6fb98f",
            "#90c98a",
            "#b8d584",
            "#e2da7e",
            "#fdd97b",
        ),
    ),
    _ScientificMap(
        "vik",
        (
            "#2b0a30",
            "#4e1f4f",
            "#6d3669",
            "#8a4f7c",
            "#a46b8a",
            "#bb8790",
            "#d1a395",
            "#e6bf97",
            "#f6d99a",
            "#fcefb0",
        ),
    ),
)

_SCIENTIFIC_NAME_SET: Set[str] = {m.name for m in _SCIENTIFIC_COLOURMAPS}
_SCIENTIFIC_BY_NAME: Dict[str, _ScientificMap] = {m.name: m for m in _SCIENTIFIC_COLOURMAPS}
_DEFAULT_FALLBACKS: Tuple[str, ...] = (
    "gray",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "turbo",
)
_REGISTERED_MAP_OBJECTS: Dict[str, "pg.ColorMap"] = {}


def _normalize_stop_color(color: Sequence[float]) -> Tuple[int, int, int, int]:
    """Convert a colour stop tuple to 0-255 RGBA integers."""

    arr = np.array(color, dtype=float).ravel()
    if arr.size == 0:
        arr = np.array([0.0, 0.0, 0.0, 255.0], dtype=float)
    if arr.size < 3:
        padded = np.zeros(4, dtype=float)
        padded[: arr.size] = arr
        padded[3] = 1.0
        arr = padded
    if arr.size == 3:
        arr = np.concatenate([arr, [1.0]])
    if arr.size > 4:
        arr = arr[:4]
    try:
        max_val = np.nanmax(np.abs(arr))
    except ValueError:
        max_val = 0.0
    if max_val <= 1.0:
        arr = arr * 255.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    arr = np.clip(np.round(arr), 0.0, 255.0)
    if arr.size < 4:
        arr = np.pad(arr, (0, 4 - arr.size), constant_values=255.0)
    return tuple(int(v) for v in arr[:4])


def _normalize_lut(lut: object) -> Optional[np.ndarray]:
    """Normalise lookup tables to an ``Nx4`` ``uint8`` array."""

    if lut is None:
        return None
    arr = np.array(lut, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        return None
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 3, axis=1)
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, arr[:, :1]], axis=1)
    if arr.shape[1] == 3:
        alpha = np.full((arr.shape[0], 1), 255.0)
        arr = np.hstack([arr, alpha])
    elif arr.shape[1] > 4:
        arr = arr[:, :4]
    try:
        max_val = np.nanmax(np.abs(arr))
    except ValueError:
        max_val = 0.0
    if max_val <= 1.0:
        arr = arr * 255.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    arr = np.clip(np.round(arr), 0.0, 255.0).astype(np.uint8)
    if arr.shape[1] == 3:
        alpha = np.full((arr.shape[0], 1), 255, dtype=np.uint8)
        arr = np.hstack([arr, alpha])
    return arr


def colormap_gradient_state(cmap: "pg.ColorMap", *, name: Optional[str] = None) -> Optional[dict]:
    """Return a gradient state dictionary for *cmap* if possible."""

    stops: Optional[Sequence[Tuple[float, Sequence[float]]]] = None
    getter = getattr(cmap, "getStops", None)
    if callable(getter):
        try:
            stops = list(getter())
        except Exception:
            stops = None
    if not stops and name:
        spec = _SCIENTIFIC_BY_NAME.get(name)
        if spec is not None:
            positions = np.linspace(0.0, 1.0, num=len(spec.stops))
            stops = [
                (float(pos), _hex_to_rgb_stop(stop) + (255,))
                for pos, stop in zip(positions, spec.stops)
            ]
    if not stops:
        return None
    ticks: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for entry in stops:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        pos, colour = entry[0], entry[1]
        try:
            pos_f = float(pos)
        except Exception:
            continue
        rgba = _normalize_stop_color(colour)
        ticks.append((pos_f, rgba))
    if not ticks:
        return None
    ticks.sort(key=lambda item: item[0])
    first = ticks[0][0]
    last = ticks[-1][0]
    span = last - first
    if span <= 1e-9:
        normalized = [(0.0, ticks[0][1]), (1.0, ticks[-1][1])]
    else:
        normalized = [((pos - first) / span, rgba) for pos, rgba in ticks]
    return {"mode": "rgb", "ticks": normalized}


def colormap_lookup_table(
    cmap: "pg.ColorMap", *, name: Optional[str] = None, size: int = 256
) -> Optional[np.ndarray]:
    """Return a lookup table suitable for :class:`~pyqtgraph.ImageItem`."""

    if cmap is None:
        return None
    getter = getattr(cmap, "getLookupTable", None)
    if callable(getter):
        attempts: Tuple[Tuple[object, ...], ...] = (
            (0.0, 1.0, size, True),
            (0.0, 1.0, size),
            (size,),
        )
        for args in attempts:
            try:
                lut = getter(*args)
            except TypeError:
                continue
            except Exception:
                lut = None
            normalized = _normalize_lut(lut)
            if normalized is not None:
                return normalized
    state = colormap_gradient_state(cmap, name=name)
    if not state:
        return None
    ticks = state.get("ticks", [])
    if len(ticks) < 2:
        return None
    positions = np.array([pos for pos, _ in ticks], dtype=float)
    colours = np.array([rgba[:3] for _, rgba in ticks], dtype=float)
    alphas = np.array([rgba[3] for _, rgba in ticks], dtype=float)
    xs = np.linspace(0.0, 1.0, max(2, int(size)))
    lut = np.empty((xs.size, 4), dtype=float)
    for channel in range(3):
        lut[:, channel] = np.interp(xs, positions, colours[:, channel])
    lut[:, 3] = np.interp(xs, positions, alphas)
    return _normalize_lut(lut)


_REGISTERED = False


def register_scientific_colormaps() -> Sequence[str]:
    """Register the bundled scientific colour maps with :mod:`pyqtgraph`.

    Returns the tuple of scientific map names so callers can highlight them in
    UI elements.  If :mod:`pyqtgraph` is not available the call becomes a no-op
    and an empty tuple is returned.
    """

    global _REGISTERED

    if pg is None:
        return ()

    _patch_colormap_menu()

    if _REGISTERED:
        return tuple(m.name for m in _SCIENTIFIC_COLOURMAPS)

    names: List[str] = []
    has_map = getattr(pg.colormap, "hasMap", None)
    list_maps = getattr(pg.colormap, "listMaps", None)
    existing = set()
    if callable(list_maps):  # pragma: no branch - compatibility shim
        try:
            existing = set(list_maps())
        except Exception:
            existing = set()

    for cmap in _SCIENTIFIC_COLOURMAPS:
        names.append(cmap.name)
        already_registered = False
        if callable(has_map):  # pragma: no branch - runtime compatibility
            try:
                already_registered = bool(has_map(cmap.name))
            except Exception:
                already_registered = False
        elif existing:
            already_registered = cmap.name in existing
        if already_registered:
            try:
                current = pg.colormap.get(cmap.name)
            except Exception:
                current = None
            if current is not None:
                _REGISTERED_MAP_OBJECTS[cmap.name] = current
            continue
        positions = np.linspace(0.0, 1.0, num=len(cmap.stops))
        colors = np.array([_hex_to_rgb_stop(stop) for stop in cmap.stops], dtype=float)
        colors /= 255.0
        try:
            color_map = pg.colormap.ColorMap(positions, colors, mapping="rgb")
        except (TypeError, KeyError):  # pragma: no cover - compatibility fallback
            try:
                color_map = pg.colormap.ColorMap(positions, colors, mode="rgb")
            except (TypeError, KeyError):  # pragma: no cover - legacy fallback
                color_map = pg.colormap.ColorMap(positions, colors)
        _REGISTERED_MAP_OBJECTS[cmap.name] = color_map
        register = getattr(pg.colormap, "register", None)
        if callable(register):  # pragma: no branch - very small helper
            register(cmap.name, color_map)
        elif hasattr(color_map, "save"):
            color_map.save(cmap.name)  # type: ignore[attr-defined]
        try:
            from pyqtgraph.graphicsItems import GradientEditorItem

            if cmap.name not in GradientEditorItem.Gradients:
                ticks = [
                    (float(pos), (_hex_to_rgb_stop(stop) + (255,)))
                    for pos, stop in zip(positions, cmap.stops)
                ]
                GradientEditorItem.Gradients[cmap.name] = {  # type: ignore[attr-defined]
                    "mode": "rgb",
                    "ticks": ticks,
                }
        except Exception:
            pass

    _REGISTERED = True
    return tuple(names)


def scientific_colormap_names() -> Sequence[str]:
    """Expose the ordered list of bundled scientific colour map names."""

    return tuple(m.name for m in _SCIENTIFIC_COLOURMAPS)


def available_colormap_names() -> List[str]:
    """Return a union of scientific and runtime-provided colormap names."""

    if pg is None:
        return list(_DEFAULT_FALLBACKS)

    register_scientific_colormaps()

    names: List[str] = []
    seen: Set[str] = set()

    for name in scientific_colormap_names():
        if name not in seen:
            names.append(name)
            seen.add(name)

    try:
        runtime_maps = list(pg.colormap.listMaps())
    except Exception:
        runtime_maps = []

    for name in runtime_maps:
        if not name or name in seen:
            continue
        names.append(name)
        seen.add(name)

    for name in _DEFAULT_FALLBACKS:
        if name not in seen:
            names.append(name)
            seen.add(name)

    return names


def is_scientific_colormap(name: str) -> bool:
    """Return True if *name* refers to one of the bundled scientific maps."""

    return str(name) in _SCIENTIFIC_NAME_SET


def get_colormap(name: str) -> Optional["pg.ColorMap"]:
    """Resolve a colour map by name, including bundled scientific entries."""

    if not name or pg is None:
        return None

    register_scientific_colormaps()

    try:
        cmap = pg.colormap.get(name)
    except Exception:
        cmap = None

    if cmap is not None:
        _REGISTERED_MAP_OBJECTS[name] = cmap
        return cmap

    cached = _REGISTERED_MAP_OBJECTS.get(name)
    if cached is not None:
        return cached

    spec = _SCIENTIFIC_BY_NAME.get(name)
    if spec is None:
        return None

    positions = np.linspace(0.0, 1.0, num=len(spec.stops))
    colors = np.array([_hex_to_rgb_stop(stop) for stop in spec.stops], dtype=float)
    colors /= 255.0
    try:
        cmap = pg.colormap.ColorMap(positions, colors, mapping="rgb")
    except (TypeError, KeyError):  # pragma: no cover - compatibility fallback
        try:
            cmap = pg.colormap.ColorMap(positions, colors, mode="rgb")
        except (TypeError, KeyError):  # pragma: no cover - legacy fallback
            cmap = pg.colormap.ColorMap(positions, colors)

    _REGISTERED_MAP_OBJECTS[name] = cmap
    return cmap

