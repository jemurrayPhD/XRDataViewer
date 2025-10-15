from __future__ import annotations

from typing import Any

try:  # pragma: no cover - depends on runtime Qt binding
    from PySide2 import QtWidgets
except Exception:  # pragma: no cover - PySide2 missing in tests
    QtWidgets = None  # type: ignore[assignment]


_PATCHED_HEADER_RESIZE = False


def ensure_header_resize_compat() -> None:
    """Provide ``setSectionResizeMode`` fallbacks on older Qt bindings."""

    global _PATCHED_HEADER_RESIZE

    if _PATCHED_HEADER_RESIZE or QtWidgets is None:  # pragma: no cover - runtime guard
        return

    header_cls: Any = getattr(QtWidgets, "QHeaderView", None)
    if header_cls is not None and not hasattr(header_cls, "setSectionResizeMode"):
        def _set_section_resize_mode(self: Any, *args: Any, **kwargs: Any) -> None:
            fallback = getattr(self, "setResizeMode", None)
            if callable(fallback):
                try:
                    fallback(*args, **kwargs)
                except Exception:
                    return
        header_cls.setSectionResizeMode = _set_section_resize_mode  # type: ignore[attr-defined]

    if not hasattr(QtWidgets.QWidget, "setSectionResizeMode"):
        def _widget_set_section_resize_mode(self: Any, *args: Any, **kwargs: Any) -> None:
            fallback = getattr(self, "setResizeMode", None)
            if callable(fallback):
                try:
                    fallback(*args, **kwargs)
                except Exception:
                    return
        QtWidgets.QWidget.setSectionResizeMode = _widget_set_section_resize_mode  # type: ignore[attr-defined]

    _PATCHED_HEADER_RESIZE = True


__all__ = ["ensure_header_resize_compat"]
