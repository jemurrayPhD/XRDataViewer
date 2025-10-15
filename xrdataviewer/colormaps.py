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
from typing import List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - pyqtgraph is part of the runtime, not tests
    import pyqtgraph as pg
except ModuleNotFoundError:  # pragma: no cover
    pg = None  # type: ignore[assignment]


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
        register = getattr(pg.colormap, "register", None)
        if callable(register):  # pragma: no branch - very small helper
            register(cmap.name, color_map)
        elif hasattr(color_map, "save"):
            color_map.save(cmap.name)  # type: ignore[attr-defined]

    _REGISTERED = True
    return tuple(names)


def scientific_colormap_names() -> Sequence[str]:
    """Expose the ordered list of bundled scientific colour map names."""

    return tuple(m.name for m in _SCIENTIFIC_COLOURMAPS)

