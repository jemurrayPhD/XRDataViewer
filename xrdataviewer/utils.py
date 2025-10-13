"""Utility helpers shared across XRDataViewer widgets."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets
import warnings

__all__ = [
    "nan_aware_reducer",
    "compose_snapshot",
    "image_with_label",
    "save_snapshot",
    "qimage_to_array",
    "ask_layout_label",
    "process_events",
    "sanitize_filename",
    "ensure_extension",
]


def nan_aware_reducer(func: Callable[[np.ndarray, Optional[int]], np.ndarray]):
    """Wrap *func* so all-NaN reductions remain NaN without warnings."""

    def wrapped(arr, axis=None):
        data = np.asarray(arr, float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = func(data, axis=axis)
        if axis is None:
            if isinstance(result, np.ndarray) and result.shape == ():
                return result.item()
            return result
        try:
            axes = axis
            if axes is None:
                return result
            if isinstance(axes, (list, tuple, set)):
                axes = tuple(int(a) for a in axes)
            else:
                axes = (int(axes),)
            ndim = data.ndim
            normalized = []
            for ax in axes:
                normalized.append((ax + ndim) % ndim if ax < 0 else ax)
            mask = np.isnan(data)
            for ax in sorted(normalized, reverse=True):
                mask = np.all(mask, axis=ax)
            if isinstance(result, np.ndarray) and np.any(mask):
                result = np.array(result, dtype=float, copy=True)
                result[mask] = np.nan
        except Exception:
            pass
        return result

    return wrapped


# Backwards compatibility for legacy imports.
_nan_aware_reducer = nan_aware_reducer


def compose_snapshot(widget: QtWidgets.QWidget, label: str = "") -> QtGui.QImage:
    """Grab *widget* as an image and optionally prepend *label*."""

    pixmap = widget.grab()
    image = pixmap.toImage()
    label = (label or "").strip()
    if not label:
        return image
    return image_with_label(image, label)


_def_font = QtGui.QFont()
_def_font.setPointSize(11)


def image_with_label(image: QtGui.QImage, label: str) -> QtGui.QImage:
    label = (label or "").strip()
    if not label:
        return image
    font = _def_font
    metrics = QtGui.QFontMetrics(font)
    margin = 8
    label_height = metrics.height() + 2 * margin
    width = image.width()
    result = QtGui.QImage(width, image.height() + label_height, QtGui.QImage.Format_ARGB32)
    result.fill(QtGui.QColor("white"))
    painter = QtGui.QPainter(result)
    painter.fillRect(QtCore.QRect(0, 0, width, label_height), QtGui.QColor("white"))
    painter.setPen(QtGui.QColor("black"))
    painter.setFont(font)
    painter.drawText(QtCore.QRect(0, 0, width, label_height), QtCore.Qt.AlignCenter, label)
    painter.drawImage(0, label_height, image)
    painter.end()
    return result


_image_with_label = image_with_label


def save_snapshot(widget: QtWidgets.QWidget, path: Path, label: str = "") -> bool:
    image = compose_snapshot(widget, label)
    try:
        return image.save(str(path))
    except Exception:
        return False


_save_snapshot = save_snapshot


def qimage_to_array(image: QtGui.QImage) -> np.ndarray:
    converted = image.convertToFormat(QtGui.QImage.Format_RGBA8888)
    width = converted.width()
    height = converted.height()
    bytes_per_line = converted.bytesPerLine()
    ptr = converted.bits()
    ptr.setsize(converted.byteCount())
    buffer = np.frombuffer(ptr, np.uint8).reshape((height, bytes_per_line // 4, 4))
    return np.array(buffer[:, :width, :], copy=True)


_qimage_to_array = qimage_to_array


def ask_layout_label(parent: QtWidgets.QWidget, title: str, default_text: str = "") -> Tuple[bool, str]:
    text, ok = QtWidgets.QInputDialog.getText(
        parent,
        title,
        "Enter a label to display above the saved layout (optional):",
        text=str(default_text or ""),
    )
    if not ok:
        return False, ""
    return True, str(text).strip()


_ask_layout_label = ask_layout_label


def process_events():
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)


_process_events = process_events


def sanitize_filename(text: str) -> str:
    safe = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text)]
    cleaned = "".join(safe).strip("_")
    return cleaned or "image"


_sanitize_filename = sanitize_filename


def ensure_extension(path: str, default_suffix: str) -> Path:
    p = Path(path)
    if not p.suffix:
        p = p.with_suffix(default_suffix)
    return p


_ensure_extension = ensure_extension
