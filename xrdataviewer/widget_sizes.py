"""Helpers for showing each widget's pixel dimensions within the UI.

This module installs a global event filter that attaches a lightweight label to
every QWidget in the application. The label sits at the bottom-right corner of
the widget and displays its current width and height in pixels. Labels update
in response to resize and layout events so they remain accurate even as the UI
changes.
"""

from __future__ import annotations

import functools
import weakref

from PySide2 import QtCore, QtWidgets

_OVERLAY_OBJECT_NAME = "_xr_widget_size_overlay"
_HELPER_ATTRIBUTE = "_xr_widget_size_helper"


def enable_widget_size_overlays() -> None:
    """Ensure widget size overlays are active for the running application."""

    app = QtWidgets.QApplication.instance()
    if app is None:
        raise RuntimeError("A QApplication must exist before enabling overlays")
    WidgetSizeOverlayManager.ensure(app)


class WidgetSizeOverlayManager(QtCore.QObject):
    """Global controller that installs overlays on widgets as they appear."""

    _instance: "weakref.ReferenceType[WidgetSizeOverlayManager] | None" = None

    def __init__(self, app: QtWidgets.QApplication):
        super().__init__(app)
        self._app = app
        self._helpers: "weakref.WeakKeyDictionary[QtWidgets.QWidget, _WidgetSizeHelper]" = (
            weakref.WeakKeyDictionary()
        )
        app.installEventFilter(self)
        self._initialize_existing_widgets()

    @classmethod
    def ensure(cls, app: QtWidgets.QApplication) -> "WidgetSizeOverlayManager":
        instance = cls._instance() if cls._instance is not None else None
        if instance is None:
            instance = cls(app)
            cls._instance = weakref.ref(instance)
        return instance

    # ------------------------------------------------------------------
    # Widget discovery helpers
    # ------------------------------------------------------------------
    def _initialize_existing_widgets(self) -> None:
        for widget in self._app.allWidgets():
            self._maybe_attach(widget)

    def _maybe_attach(self, widget: QtWidgets.QWidget | None) -> None:
        if widget is None:
            return
        if not isinstance(widget, QtWidgets.QWidget):
            return
        if widget.objectName() == _OVERLAY_OBJECT_NAME:
            return
        if getattr(widget, _HELPER_ATTRIBUTE, None) is not None:
            return
        flags = widget.windowFlags()
        if widget.inherits("QMenu") or flags & QtCore.Qt.ToolTip:
            return

        helper = _WidgetSizeHelper(widget)
        setattr(widget, _HELPER_ATTRIBUTE, helper)
        self._helpers[widget] = helper

    # ------------------------------------------------------------------
    # Qt event filter
    # ------------------------------------------------------------------
    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # type: ignore[override]
        if event.type() == QtCore.QEvent.ChildAdded:
            child = event.child()
            if isinstance(child, QtWidgets.QWidget):
                QtCore.QTimer.singleShot(0, functools.partial(self._maybe_attach, child))
        elif event.type() == QtCore.QEvent.ChildRemoved:
            child = event.child()
            if isinstance(child, QtWidgets.QWidget):
                helper = getattr(child, _HELPER_ATTRIBUTE, None)
                if helper is not None:
                    helper.cleanup()
        return super().eventFilter(watched, event)


class _WidgetSizeHelper(QtCore.QObject):
    """Maintain a size label for a specific widget."""

    def __init__(self, widget: QtWidgets.QWidget):
        super().__init__(widget)
        self._widget_ref = weakref.ref(widget)
        self._label = QtWidgets.QLabel(widget)
        self._label.setObjectName(_OVERLAY_OBJECT_NAME)
        self._label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 130);"
            "color: white;"
            "font-size: 8px;"
            "padding: 1px 3px;"
            "border-radius: 3px;"
        )
        self._label.hide()
        widget.installEventFilter(self)
        self._update_label()

    def cleanup(self) -> None:
        widget = self._widget_ref()
        if widget is not None:
            try:
                widget.removeEventFilter(self)
            except RuntimeError:
                pass
        if self._label is not None:
            self._label.deleteLater()
        self.deleteLater()

    # ------------------------------------------------------------------
    # Event handling for the tracked widget
    # ------------------------------------------------------------------
    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # type: ignore[override]
        if event.type() in (
            QtCore.QEvent.Show,
            QtCore.QEvent.Resize,
            QtCore.QEvent.LayoutRequest,
            QtCore.QEvent.PolishRequest,
        ):
            self._update_label()
        elif event.type() == QtCore.QEvent.Hide:
            self._label.hide()
        elif event.type() == QtCore.QEvent.Destroyed:
            self.cleanup()
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_label(self) -> None:
        widget = self._widget_ref()
        if widget is None:
            return
        if not widget.isVisible():
            self._label.hide()
            return

        width = widget.width()
        height = widget.height()
        if width <= 0 or height <= 0:
            self._label.hide()
            return

        self._label.setText(f"{width}×{height}")
        self._label.adjustSize()

        rect = widget.contentsRect()
        label_size = self._label.size()
        x = rect.right() - label_size.width() - 4
        y = rect.bottom() - label_size.height() - 2
        x = max(x, rect.left())
        y = max(y, rect.top())

        self._label.move(x, y)
        self._label.show()
        self._label.raise_()

