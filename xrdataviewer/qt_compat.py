from __future__ import annotations

from typing import Any, Optional

try:  # pragma: no cover - depends on runtime Qt binding
    from PySide2 import QtCore, QtWidgets
except Exception:  # pragma: no cover - PySide2 missing in tests
    QtCore = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]


_PATCHED_HEADER_RESIZE = False
_MESSAGE_BOX_FILTER: Optional["_MessageBoxSizingFilter"] = None


if QtCore is not None and QtWidgets is not None:

    class _MessageBoxSizingFilter(QtCore.QObject):  # type: ignore[misc]
        """Event filter that enforces roomy, wrapped message-box text."""

        def __init__(self, min_width: int = 520, min_height: int = 200) -> None:
            super().__init__()
            self._min_width = int(min_width)
            self._min_height = int(min_height)

        def eventFilter(self, obj: "QtCore.QObject", event: "QtCore.QEvent") -> bool:  # type: ignore[override]
            if isinstance(obj, QtWidgets.QMessageBox):
                event_type = event.type()
                if event_type in (
                    QtCore.QEvent.Show,
                    QtCore.QEvent.ShowToParent,
                    QtCore.QEvent.Resize,
                ):
                    self._apply_sizing(obj)
            return super().eventFilter(obj, event)

        def _apply_sizing(self, box: "QtWidgets.QMessageBox") -> None:
            for name in ("qt_msgbox_label", "qt_msgbox_informativelabel"):
                label = box.findChild(QtWidgets.QLabel, name)
                if label is not None:
                    label.setWordWrap(True)
                    label.setTextInteractionFlags(
                        QtCore.Qt.TextSelectableByMouse | QtCore.Qt.LinksAccessibleByMouse
                    )
            hint = box.sizeHint()
            target_width = max(hint.width(), self._min_width)
            target_height = max(hint.height(), self._min_height)

            max_size = box.maximumSize()
            if max_size.width() > 0:
                target_width = min(target_width, max_size.width())
            if max_size.height() > 0:
                target_height = min(target_height, max_size.height())

            resize_width = max(box.width(), target_width)
            resize_height = max(box.height(), target_height)
            if resize_width != box.width() or resize_height != box.height():
                box.resize(resize_width, resize_height)
            box.setSizeGripEnabled(True)

else:  # pragma: no cover - occurs when Qt bindings unavailable during tests

    class _MessageBoxSizingFilter:  # type: ignore[too-many-ancestors]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def eventFilter(self, *args: Any, **kwargs: Any) -> bool:
            return False


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


def ensure_messagebox_sizing(min_width: int = 520, min_height: int = 200) -> None:
    """Install an event filter so message boxes have room for longer text."""

    global _MESSAGE_BOX_FILTER

    if QtWidgets is None or QtCore is None:  # pragma: no cover - runtime guard
        return

    if _MESSAGE_BOX_FILTER is not None:
        return

    app = QtWidgets.QApplication.instance()
    if app is None:  # pragma: no cover - requires running Qt app
        return

    _MESSAGE_BOX_FILTER = _MessageBoxSizingFilter(min_width=min_width, min_height=min_height)
    app.installEventFilter(_MESSAGE_BOX_FILTER)


__all__ = ["ensure_header_resize_compat", "ensure_messagebox_sizing"]
