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

try:
    from shiboken2 import shiboken2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    shiboken2 = None  # type: ignore

try:
    import sip  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sip = None  # type: ignore

from PySide2 import QtCore, QtWidgets

_OVERLAY_OBJECT_NAME = "_xr_widget_size_overlay"
_HELPER_ATTRIBUTE = "_xr_widget_size_helper"

_DESTROY_EVENT = getattr(QtCore.QEvent, "Destroyed", None)
if _DESTROY_EVENT is None:  # pragma: no cover - PySide2 naming
    _DESTROY_EVENT = getattr(QtCore.QEvent, "Destroy", None)


def _is_valid_qobject(obj: QtCore.QObject | None) -> bool:
    """Return ``True`` when ``obj`` still wraps a live C++ instance."""

    if obj is None:
        return False
    if shiboken2 is not None:  # pragma: no branch - preferred path under PySide
        try:
            return bool(shiboken2.isValid(obj))
        except Exception:  # pragma: no cover - safety net
            return False
    if sip is not None:
        try:
            return not sip.isdeleted(obj)
        except Exception:  # pragma: no cover - safety net
            return False
    try:
        # Accessing a simple attribute forces PyQt/PySide to validate the wrapper.
        obj.objectName()
    except RuntimeError:
        return False
    except AttributeError:
        return False
    return True


def _is_valid_widget(widget: QtWidgets.QWidget | None) -> bool:
    return isinstance(widget, QtWidgets.QWidget) and _is_valid_qobject(widget)


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
            if _is_valid_widget(widget):
                self._maybe_attach(widget)

    def _maybe_attach(self, widget: QtWidgets.QWidget | None) -> None:
        if not _is_valid_widget(widget):
            return

        try:
            name = widget.objectName()
        except RuntimeError:
            return
        if name == _OVERLAY_OBJECT_NAME:
            return
        if getattr(widget, _HELPER_ATTRIBUTE, None) is not None:
            return

        if isinstance(widget, (QtWidgets.QMenu, QtWidgets.QMenuBar)):
            return

        try:
            flags = widget.windowFlags()
        except RuntimeError:
            return
        try:
            is_menu = widget.inherits("QMenu")
        except RuntimeError:
            is_menu = False
        try:
            is_menu_bar = widget.inherits("QMenuBar")
        except RuntimeError:
            is_menu_bar = False
        if is_menu or is_menu_bar or flags & QtCore.Qt.ToolTip:
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
            if _is_valid_widget(child):
                QtCore.QTimer.singleShot(0, functools.partial(self._maybe_attach, child))
        elif event.type() == QtCore.QEvent.ChildRemoved:
            child = event.child()
            if _is_valid_widget(child):
                try:
                    helper = getattr(child, _HELPER_ATTRIBUTE, None)
                except RuntimeError:
                    helper = None
                if helper is not None:
                    helper.cleanup()
        return super().eventFilter(watched, event)


class _WidgetSizeHelper(QtCore.QObject):
    """Maintain a size label for a specific widget."""

    def __init__(self, widget: QtWidgets.QWidget):
        super().__init__(widget)
        self._widget_ref = weakref.ref(widget)
        self._label: QtWidgets.QLabel | None = QtWidgets.QLabel(widget)
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
        try:
            widget.destroyed.connect(self._on_widget_destroyed)
        except Exception:  # pragma: no cover - defensive
            pass
        self._update_label()

    def cleanup(self) -> None:
        widget = self._widget_ref()
        if widget is not None:
            try:
                widget.removeEventFilter(self)
            except RuntimeError:
                pass
            try:
                delattr(widget, _HELPER_ATTRIBUTE)
            except Exception:
                pass
        if self._label is not None:
            if _is_valid_widget(self._label):
                self._label.hide()
                self._label.deleteLater()
            self._label = None
        self.deleteLater()

    # ------------------------------------------------------------------
    # Event handling for the tracked widget
    # ------------------------------------------------------------------
    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # type: ignore[override]
        label = self._label
        if not _is_valid_widget(label):
            return False

        if event.type() in (
            QtCore.QEvent.Show,
            QtCore.QEvent.Resize,
            QtCore.QEvent.LayoutRequest,
            QtCore.QEvent.PolishRequest,
        ):
            self._update_label()
        elif event.type() == QtCore.QEvent.Hide:
            label.hide()
        elif _DESTROY_EVENT is not None and event.type() == _DESTROY_EVENT:
            self.cleanup()
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_label(self) -> None:
        label = self._label
        if not _is_valid_widget(label):
            return

        widget = self._widget_ref()
        if not _is_valid_widget(widget):
            return
        try:
            visible = widget.isVisible()
        except RuntimeError:
            return
        if not visible:
            label.hide()
            return

        try:
            width = widget.width()
            height = widget.height()
        except RuntimeError:
            label.hide()
            return
        if width <= 0 or height <= 0:
            label.hide()
            return

        label.setText(f"{width}×{height}")
        label.adjustSize()

        try:
            rect = widget.contentsRect()
        except RuntimeError:
            label.hide()
            return
        label_size = label.size()
        x = rect.right() - label_size.width() - 4
        y = rect.bottom() - label_size.height() - 2
        x = max(x, rect.left())
        y = max(y, rect.top())

        label.move(x, y)
        label.show()
        label.raise_()

    def _on_widget_destroyed(self, *_args) -> None:  # pragma: no cover - Qt callback
        self.cleanup()

