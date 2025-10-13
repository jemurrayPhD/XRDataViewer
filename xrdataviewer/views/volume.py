from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets

try:  # Optional dependency
    import pyqtgraph.opengl as gl  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    gl = None  # type: ignore

from app_logging import log_action

from ..preferences import PreferencesManager
from ..utils import _ensure_extension, _save_snapshot


class VolumeAlphaHandle(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, owner: "VolumeAlphaCurveWidget", x_norm: float, y_norm: float):
        radius = 5.0
        super().__init__(-radius, -radius, radius * 2.0, radius * 2.0)
        self._owner = owner
        self.x_norm = float(x_norm)
        self.y_norm = float(y_norm)
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        pen = QtGui.QPen(QtGui.QColor(20, 20, 20))
        pen.setWidthF(1.2)
        self.setPen(pen)
        self.setZValue(20)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)

    def itemChange(self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            if isinstance(value, QtCore.QPointF):
                point = value
            else:
                point = QtCore.QPointF(value)
            return self._owner.clamp_handle_position(self, point)
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self._owner.handle_moved(self)
        return super().itemChange(change, value)

class VolumeAlphaCurveWidget(QtWidgets.QWidget):
    curveChanged = QtCore.Signal(list)

    def __init__(self, parent=None, default_value: float = 0.5):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self._view = QtWidgets.QGraphicsView(self._scene)
        self._view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._view.setRenderHints(
            QtGui.QPainter.Antialiasing
            | QtGui.QPainter.SmoothPixmapTransform
            | QtGui.QPainter.TextAntialiasing
        )
        self._view.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self._view.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._view.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        )
        self._view.viewport().installEventFilter(self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._gradient_item = QtWidgets.QGraphicsPixmapItem()
        self._gradient_item.setZValue(0)
        self._scene.addItem(self._gradient_item)

        border_pen = QtGui.QPen(QtGui.QColor(200, 200, 200))
        border_pen.setWidthF(1.0)
        self._border_item = self._scene.addRect(QtCore.QRectF(0, 0, 1, 1), border_pen)
        self._border_item.setZValue(5)

        curve_pen = QtGui.QPen(QtGui.QColor(245, 245, 245))
        curve_pen.setWidthF(2.0)
        self._curve_item = self._scene.addPath(QtGui.QPainterPath(), curve_pen)
        self._curve_item.setZValue(15)

        self._handles: List[VolumeAlphaHandle] = []
        self._default_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        self._default_value = max(0.0, min(1.0, float(default_value)))
        self._margin_left = 28.0
        self._margin_right = 16.0
        self._margin_top = 12.0
        self._margin_bottom = 26.0
        self._colormap_name = "viridis"
        self._updating = False

        self.setMinimumHeight(120)
        self.setMaximumHeight(220)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.setSizePolicy(size_policy)
        self._update_scene_geometry()
        self.reset_curve()

    # ----- geometry helpers -----
    def showEvent(self, event: QtGui.QShowEvent):
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._update_scene_geometry)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self._update_scene_geometry()

    def _effective_rect(self) -> QtCore.QRectF:
        rect = self._scene.sceneRect()
        width = max(1.0, rect.width() - self._margin_left - self._margin_right)
        height = max(1.0, rect.height() - self._margin_top - self._margin_bottom)
        return QtCore.QRectF(
            rect.left() + self._margin_left,
            rect.top() + self._margin_top,
            width,
            height,
        )

    def _update_scene_geometry(self):
        viewport = self._view.viewport()
        width = max(1, viewport.width())
        height = max(1, viewport.height())
        self._scene.setSceneRect(0, 0, float(width), float(height))
        eff = self._effective_rect()
        self._update_gradient_pixmap(int(max(2.0, eff.width())), int(max(2.0, eff.height())))
        self._gradient_item.setPos(eff.left(), eff.top())
        self._border_item.setRect(eff)
        self._position_handles()
        self._update_curve_path()

    # ----- colormap -----
    def set_colormap(self, name: str):
        if not name:
            name = "viridis"
        if self._colormap_name == name:
            return
        self._colormap_name = name
        self._update_gradient_pixmap()
        self._update_curve_path()

    def _update_gradient_pixmap(self, width: Optional[int] = None, height: Optional[int] = None):
        eff = self._effective_rect()
        w = int(width or max(2.0, eff.width()))
        h = int(height or max(2.0, eff.height()))
        try:
            cmap = pg.colormap.get(self._colormap_name)
        except Exception:
            cmap = pg.colormap.get("viridis")
        lut = cmap.map(np.linspace(0.0, 1.0, max(2, w)), mode="byte")
        gradient = np.repeat(lut[np.newaxis, :, :3], max(2, h), axis=0)
        alpha = np.full((gradient.shape[0], gradient.shape[1], 1), 255, dtype=np.uint8)
        rgba = np.concatenate((gradient, alpha), axis=2)
        image = QtGui.QImage(
            rgba.data, rgba.shape[1], rgba.shape[0], int(rgba.strides[0]), QtGui.QImage.Format_RGBA8888
        )
        image = image.copy()
        self._gradient_item.setPixmap(QtGui.QPixmap.fromImage(image))

    # ----- handle interactions -----
    def clamp_handle_position(
        self, handle: VolumeAlphaHandle, value: QtCore.QPointF
    ) -> QtCore.QPointF:
        if self._updating:
            return value
        eff = self._effective_rect()
        width = eff.width()
        height = eff.height()
        x_norm = 0.0
        if width > 0:
            raw_x_norm = (float(value.x()) - eff.left()) / width
            idx = self._handles.index(handle)
            min_norm = 0.0 if idx == 0 else self._handles[idx - 1].x_norm + 1e-4
            max_norm = 1.0 if idx == len(self._handles) - 1 else self._handles[idx + 1].x_norm - 1e-4
            x_norm = max(min_norm, min(max_norm, raw_x_norm))
            x_norm = max(0.0, min(1.0, x_norm))
        x = eff.left() + x_norm * max(width, 1.0)
        y = float(value.y())
        if height > 0:
            y = max(eff.top(), min(eff.bottom(), y))
        return QtCore.QPointF(x, y)

    def handle_moved(self, handle: VolumeAlphaHandle):
        if self._updating:
            return
        eff = self._effective_rect()
        width = eff.width()
        height = eff.height()
        if width <= 0 or height <= 0:
            return
        pos = handle.pos()
        x_norm = (pos.x() - eff.left()) / width
        y_norm = (eff.bottom() - pos.y()) / height
        x_norm = max(0.0, min(1.0, float(x_norm)))
        y_norm = max(0.0, min(1.0, float(y_norm)))
        idx = self._handles.index(handle)
        if idx > 0:
            x_norm = max(x_norm, self._handles[idx - 1].x_norm + 1e-4)
        if idx < len(self._handles) - 1:
            x_norm = min(x_norm, self._handles[idx + 1].x_norm - 1e-4)
        handle.y_norm = y_norm
        handle.x_norm = x_norm
        self._handles.sort(key=lambda item: item.x_norm)
        self._position_handles()
        self._update_curve_path()
        self.curveChanged.emit(self.curve_points())

    def _position_handles(self):
        eff = self._effective_rect()
        if eff.height() <= 0 or eff.width() <= 0:
            return
        self._updating = True
        try:
            for handle in self._handles:
                x = eff.left() + handle.x_norm * eff.width()
                y = eff.bottom() - handle.y_norm * eff.height()
                handle.setPos(QtCore.QPointF(x, y))
        finally:
            self._updating = False

    def _update_curve_path(self):
        if not self._handles:
            self._curve_item.setPath(QtGui.QPainterPath())
            return
        sorted_handles = sorted(self._handles, key=lambda item: item.x_norm)
        path = QtGui.QPainterPath()
        first = sorted_handles[0]
        path.moveTo(first.pos())
        for handle in sorted_handles[1:]:
            path.lineTo(handle.pos())
        self._curve_item.setPath(path)

    # ----- curve helpers -----
    def curve_points(self) -> List[Tuple[float, float]]:
        return [(handle.x_norm, handle.y_norm) for handle in sorted(self._handles, key=lambda h: h.x_norm)]

    def set_curve(self, points: List[Tuple[float, float]]):
        if not points:
            return
        for handle in list(self._handles):
            self._scene.removeItem(handle)
        self._handles.clear()
        for x, y in points:
            self._add_handle(float(x), float(y), emit=False)
        self._update_curve_path()
        self.curveChanged.emit(self.curve_points())

    def reset_curve(self):
        for handle in list(self._handles):
            self._scene.removeItem(handle)
        self._handles.clear()
        for x_norm in self._default_positions:
            self._add_handle(float(x_norm), float(self._default_value), emit=False)
        self._update_curve_path()
        self.curveChanged.emit(self.curve_points())

    def _add_handle(self, x_norm: float, y_norm: float, *, emit: bool = True):
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        for existing in self._handles:
            if abs(existing.x_norm - x_norm) < 1e-4:
                existing.x_norm = x_norm
                existing.y_norm = y_norm
                self._handles.sort(key=lambda item: item.x_norm)
                self._position_handles()
                self._update_curve_path()
                if emit:
                    self.curveChanged.emit(self.curve_points())
                return
        handle = VolumeAlphaHandle(self, x_norm, y_norm)
        self._scene.addItem(handle)
        self._handles.append(handle)
        self._handles.sort(key=lambda item: item.x_norm)
        self._position_handles()
        self._update_curve_path()
        if emit:
            self.curveChanged.emit(self.curve_points())

    def eventFilter(self, obj, event):
        if (
            obj is self._view.viewport()
            and event.type() == QtCore.QEvent.MouseButtonDblClick
            and self.isEnabled()
        ):
            if isinstance(event, QtGui.QMouseEvent) and event.button() == QtCore.Qt.LeftButton:
                scene_pos = self._view.mapToScene(event.pos())
                eff = self._effective_rect()
                if eff.contains(scene_pos):
                    width = eff.width() if eff.width() > 0 else 1.0
                    height = eff.height() if eff.height() > 0 else 1.0
                    x_norm = (scene_pos.x() - eff.left()) / width
                    y_norm = (eff.bottom() - scene_pos.y()) / height
                    self._add_handle(x_norm, y_norm)
                    return True
        return super().eventFilter(obj, event)

class SequentialVolumeWindow(QtWidgets.QWidget):
    closed = QtCore.Signal()

    def __init__(self, parent=None, preferences: Optional[PreferencesManager] = None):
        if gl is None:
            raise RuntimeError("pyqtgraph.opengl is not available")
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Sequential Volume Viewer")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.resize(680, 520)

        self._data: Optional[np.ndarray] = None
        self._data_min: float = 0.0
        self._data_max: float = 1.0
        self._colormap_name: str = "viridis"
        self.preferences: Optional[PreferencesManager] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        cmap_controls = QtWidgets.QHBoxLayout()
        cmap_controls.setSpacing(6)
        cmap_controls.addWidget(QtWidgets.QLabel("Colormap:"))

        self.cmb_colormap = QtWidgets.QComboBox()
        self.cmb_colormap.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cmb_colormap.currentIndexChanged.connect(self._on_colormap_combo_changed)
        cmap_controls.addWidget(self.cmb_colormap, 0)

        cmap_controls.addStretch(1)
        layout.addLayout(cmap_controls)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(QtWidgets.QLabel("Opacity curves:"))

        self.btn_reset_curve = QtWidgets.QPushButton("Reset curves")
        self.btn_reset_curve.clicked.connect(self._on_reset_curve)
        controls.addWidget(self.btn_reset_curve)

        self.btn_reset_view = QtWidgets.QPushButton("Reset view")
        self.btn_reset_view.setEnabled(False)
        self.btn_reset_view.clicked.connect(self._on_reset_view)
        controls.addWidget(self.btn_reset_view)

        self.btn_export = QtWidgets.QToolButton()
        self.btn_export.setText("Export")
        self.btn_export.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        export_menu = QtWidgets.QMenu(self.btn_export)
        act_snapshot = export_menu.addAction("Save volume snapshot…")
        act_snapshot.triggered.connect(self._export_volume_snapshot)
        act_layout = export_menu.addAction("Save volume layout…")
        act_layout.triggered.connect(self._export_volume_layout)
        self.btn_export.setMenu(export_menu)
        controls.addWidget(self.btn_export)

        controls.addStretch(1)
        layout.addLayout(controls)

        curves_row = QtWidgets.QHBoxLayout()
        curves_row.setSpacing(8)
        layout.addLayout(curves_row)

        self._curve_keys: Tuple[str, ...] = ("value", "slice", "row", "column")
        self._axis_labels: Dict[str, str] = {
            "value": "Value",
            "slice": "Slice axis",
            "row": "Row axis",
            "column": "Column axis",
        }
        self._curve_widgets: Dict[str, VolumeAlphaCurveWidget] = {}
        self._curve_labels: Dict[str, QtWidgets.QLabel] = {}

        for key in self._curve_keys:
            column_layout = QtWidgets.QVBoxLayout()
            column_layout.setSpacing(4)
            label = QtWidgets.QLabel(self._axis_labels[key])
            label.setAlignment(QtCore.Qt.AlignHCenter)
            label.setWordWrap(True)
            column_layout.addWidget(label)

            widget = VolumeAlphaCurveWidget(default_value=0.25)
            widget.setMinimumWidth(150)
            widget.setSizePolicy(
                QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            )
            widget.setToolTip(
                "Drag control points to sculpt opacity, double-click to add a new point."
            )
            widget.curveChanged.connect(lambda points, k=key: self._on_alpha_curve_changed(k, points))
            column_layout.addWidget(widget)

            curves_row.addLayout(column_layout, 1)
            self._curve_widgets[key] = widget
            self._curve_labels[key] = label

        self._volume_item: Optional[gl.GLVolumeItem] = None
        self._volume_scalar: Optional[np.ndarray] = None
        self._volume_shape: Tuple[int, int, int] = (1, 1, 1)
        self._curve_points: Dict[str, List[Tuple[float, float]]] = {}
        self._curve_lut_x: Dict[str, np.ndarray] = {}
        self._curve_lut_y: Dict[str, np.ndarray] = {}
        for key, widget in self._curve_widgets.items():
            points = widget.curve_points()
            xs = np.array([max(0.0, min(1.0, float(x))) for x, _ in points], dtype=float)
            ys = np.array([max(0.0, min(1.0, float(y))) for _, y in points], dtype=float)
            self._curve_points[key] = [(float(x), float(y)) for x, y in zip(xs, ys)]
            self._curve_lut_x[key] = xs
            self._curve_lut_y[key] = ys
        self._alpha_scale_base: float = 101.0

        self._populate_colormap_choices()
        self._update_alpha_controls()

        self.view = gl.GLViewWidget()
        self.view.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        )
        self.view.setMinimumHeight(260)
        self.view.opts["distance"] = 400
        self.view.setBackgroundColor(QtGui.QColor(20, 20, 20))
        layout.addWidget(self.view, 1)

    def set_preferences(self, preferences: Optional[PreferencesManager]):
        if self.preferences is preferences:
            return
        if self.preferences:
            try:
                self.preferences.changed.disconnect(self._on_preferences_changed)
            except Exception:
                pass
        self.preferences = preferences
        if preferences is not None:
            try:
                preferences.changed.connect(self._on_preferences_changed)
            except Exception:
                pass
        self._apply_preference_defaults()

    def _on_preferences_changed(self, _data):
        self._apply_preference_defaults()

    def _apply_preference_defaults(self):
        if not self.preferences:
            return
        preferred = self.preferences.preferred_colormap(None)
        if preferred:
            self.set_colormap(preferred)

    def _default_layout_label(self) -> str:
        if self.preferences:
            return self.preferences.default_layout_label()
        return ""

    def _default_export_dir(self) -> str:
        if self.preferences:
            return self.preferences.default_export_directory()
        return ""

    def _store_export_dir(self, directory: str):
        if self.preferences and directory:
            data = self.preferences.data()
            misc = data.setdefault("misc", {})
            misc["default_export_dir"] = directory
            self.preferences.update(data)

    def _initial_path(self, filename: str) -> str:
        base = self._default_export_dir()
        if base:
            return str(Path(base) / filename)
        return filename

        self.set_preferences(preferences)

    # ----- public API -----
    def set_volume(self, data: Optional[np.ndarray]):
        if data is None or data.size == 0:
            self._data = None
            self._remove_volume()
            self._update_alpha_controls()
            return
        arr = np.asarray(data, float)
        finite = np.isfinite(arr)
        if not finite.any():
            arr = np.zeros_like(arr, dtype=float)
            self._data_min = 0.0
            self._data_max = 1.0
        else:
            min_val = float(np.nanmin(arr))
            max_val = float(np.nanmax(arr))
            if min_val == max_val:
                max_val = min_val + 1.0
            arr = np.nan_to_num(arr, nan=min_val)
            self._data_min = min_val
            self._data_max = max_val
        self._data = arr.astype(np.float32, copy=False)
        self._volume_scalar = self._prepare_volume_array(self._data)
        self._volume_shape = self._volume_scalar.shape if self._volume_scalar is not None else (1, 1, 1)
        self._update_alpha_controls()
        self._ensure_volume_item()
        self._center_volume_item()
        self._update_volume_visual()
        self._reset_camera()

    def set_colormap(self, name: Optional[str]):
        if name:
            self._colormap_name = str(name)
        else:
            self._colormap_name = "viridis"
        for widget in self._curve_widgets.values():
            try:
                widget.set_colormap(self._colormap_name)
            except Exception:
                continue
        self._sync_colormap_combo()
        self._update_volume_visual()

    def clear_volume(self):
        self._data = None
        self._volume_scalar = None
        self._volume_shape = (1, 1, 1)
        self._remove_volume()
        self._update_alpha_controls()
        self.btn_reset_view.setEnabled(False)

    # ----- helpers -----
    def _ensure_volume_item(self):
        if self._volume_item is not None:
            return
        data = self._data
        if data is None:
            return
        scalar = self._prepare_volume_array(data)
        self._volume_scalar = scalar
        rgba = self._compute_rgba_volume(scalar)
        self._volume_item = gl.GLVolumeItem(rgba, smooth=False)
        # Use translucent blending so colors remain readable instead of
        # saturating to white as layers accumulate with additive blending.
        self._volume_item.setGLOptions("translucent")
        self._volume_item.resetTransform()
        self._center_volume_item()
        self.view.addItem(self._volume_item)
        if hasattr(self._volume_item, "update"):
            try:
                self._volume_item.update()
            except Exception:
                pass
        self.btn_reset_view.setEnabled(True)

    def _prepare_volume_array(self, data: np.ndarray) -> np.ndarray:
        if data.ndim != 3:
            return np.zeros((1, 1, 1), dtype=np.float32)
        transposed = np.transpose(data, (2, 1, 0))
        return np.ascontiguousarray(transposed, dtype=np.float32)

    def _compute_rgba_volume(self, scalar: np.ndarray) -> np.ndarray:
        if scalar.size == 0:
            return np.zeros(scalar.shape + (4,), dtype=np.ubyte)
        try:
            cmap = pg.colormap.get(self._colormap_name)
        except Exception:
            cmap = pg.colormap.get("viridis")
        data_min = float(self._data_min)
        data_max = float(self._data_max)
        if not np.isfinite(data_min) or not np.isfinite(data_max) or data_min == data_max:
            data_min = float(np.nanmin(scalar))
            data_max = float(np.nanmax(scalar))
            if not np.isfinite(data_min):
                data_min = 0.0
            if not np.isfinite(data_max) or data_max == data_min:
                data_max = data_min + 1.0
        scale = data_max - data_min
        if scale == 0.0:
            scale = 1.0
        norm = (scalar - data_min) / scale
        norm = np.clip(norm, 0.0, 1.0)
        rgba = cmap.map(norm.reshape(-1), mode="byte").reshape(scalar.shape + (4,))

        alpha_value = self._apply_alpha_scale(self._sample_alpha_curve("value", norm))
        alpha_total = alpha_value
        if scalar.ndim >= 3:
            col_len, row_len, slice_len = scalar.shape
            if slice_len > 1:
                slice_positions = np.linspace(0.0, 1.0, slice_len, dtype=float)
                slice_alpha = self._apply_alpha_scale(
                    self._sample_alpha_curve("slice", slice_positions)
                ).reshape(1, 1, slice_len)
                alpha_total = alpha_total * slice_alpha
            if row_len > 1:
                row_positions = np.linspace(0.0, 1.0, row_len, dtype=float)
                row_alpha = self._apply_alpha_scale(
                    self._sample_alpha_curve("row", row_positions)
                ).reshape(1, row_len, 1)
                alpha_total = alpha_total * row_alpha
            if col_len > 1:
                col_positions = np.linspace(0.0, 1.0, col_len, dtype=float)
                col_alpha = self._apply_alpha_scale(
                    self._sample_alpha_curve("column", col_positions)
                ).reshape(col_len, 1, 1)
                alpha_total = alpha_total * col_alpha
        alpha = np.clip(alpha_total * 255.0, 0.0, 255.0)
        rgba = rgba.copy()
        rgba[..., 3] = alpha.astype(np.uint8)
        return np.ascontiguousarray(rgba, dtype=np.ubyte)

    def _update_volume_visual(self):
        if self._volume_item is None or self._volume_scalar is None:
            return
        rgba = self._compute_rgba_volume(self._volume_scalar)
        try:
            self._volume_item.setData(rgba)
        except TypeError:
            # Older pyqtgraph releases expect the array as the first argument.
            self._volume_item.setData(data=rgba)
        self._center_volume_item()
        self.view.update()

    def _center_volume_item(self):
        if self._volume_item is None or self._volume_scalar is None:
            return
        try:
            self._volume_item.resetTransform()
        except Exception:
            pass
        shape = self._volume_scalar.shape
        if len(shape) != 3:
            return
        offset = [-dim / 2.0 for dim in shape]
        try:
            self._volume_item.translate(*offset)
        except Exception:
            pass

    def _reset_camera(self):
        if self._volume_scalar is None or self._volume_scalar.size == 0:
            return
        shape = self._volume_scalar.shape
        max_dim = float(max(shape)) if shape else 1.0
        distance = max(200.0, max_dim * 2.2)
        try:
            self.view.opts["center"] = pg.Vector(0.0, 0.0, 0.0)
        except Exception:
            try:
                self.view.opts["center"] = QtGui.QVector3D(0.0, 0.0, 0.0)
            except Exception:
                pass
        try:
            self.view.setCameraPosition(distance=distance, elevation=26, azimuth=32)
        except Exception:
            self.view.opts["distance"] = distance
        self.view.update()

    def _populate_colormap_choices(self):
        candidates = [
            "gray",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "thermal",
            "blues",
            "reds",
        ]
        self.cmb_colormap.blockSignals(True)
        self.cmb_colormap.clear()
        for name in candidates:
            try:
                pg.colormap.get(name)
            except Exception:
                continue
            self.cmb_colormap.addItem(name.title(), name)
        if self.cmb_colormap.count() == 0:
            self.cmb_colormap.addItem("Viridis", "viridis")
        self.cmb_colormap.blockSignals(False)
        self._sync_colormap_combo()

    def _sync_colormap_combo(self):
        if not hasattr(self, "cmb_colormap"):
            return
        name = self._colormap_name or "viridis"
        block = self.cmb_colormap.blockSignals(True)
        idx = self.cmb_colormap.findData(name)
        if idx < 0 and self.cmb_colormap.count():
            idx = 0
            name = self.cmb_colormap.itemData(0)
            self._colormap_name = name
        if idx >= 0:
            self.cmb_colormap.setCurrentIndex(idx)
        self.cmb_colormap.blockSignals(block)

    def _on_colormap_combo_changed(self):
        name = self.cmb_colormap.currentData()
        if not name:
            name = "viridis"
        if name == self._colormap_name:
            return
        self.set_colormap(name)
        if hasattr(self._volume_item, "update"):
            try:
                self._volume_item.update()
            except Exception:
                pass

    def _remove_volume(self):
        if self._volume_item is None:
            return
        try:
            self.view.removeItem(self._volume_item)
        except Exception:
            pass
        self._volume_item = None

    def _update_alpha_controls(self):
        has_data = self._data is not None
        for widget in self._curve_widgets.values():
            widget.setEnabled(has_data)
        self.btn_reset_curve.setEnabled(has_data)
        self.btn_reset_view.setEnabled(has_data and self._volume_item is not None)
        if hasattr(self, "btn_export"):
            self.btn_export.setEnabled(has_data)

    def _export_volume_snapshot(self):
        if self._volume_item is None:
            QtWidgets.QMessageBox.information(
                self,
                "No volume",
                "Load data into the volume view before exporting.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save volume snapshot",
            self._initial_path("volume-snapshot.png"),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        ok, label = _ask_layout_label(self, "Snapshot label", self._default_layout_label())
        if not ok:
            return
        suffix = ".jpg" if path.lower().endswith((".jpg", ".jpeg")) else ".png"
        target = _ensure_extension(path, suffix)
        try:
            image = self.view.grabFramebuffer()
        except Exception:
            image = QtGui.QImage()
        if image.isNull():
            QtWidgets.QMessageBox.warning(self, "Save failed", "Unable to capture the 3D view.")
            return
        if label:
            image = _image_with_label(image, label)
        if not image.save(str(target)):
            QtWidgets.QMessageBox.warning(self, "Save failed", "Unable to save the snapshot.")
            return
        self._store_export_dir(str(Path(target).parent))
        log_action(f"Saved volume snapshot to {target}")

    def _export_volume_layout(self):
        if self._volume_item is None:
            QtWidgets.QMessageBox.information(
                self,
                "No volume",
                "Load data into the volume view before exporting.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save volume layout",
            self._initial_path("volume-layout.png"),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        ok, label = _ask_layout_label(self, "Layout label", self._default_layout_label())
        if not ok:
            return
        suffix = ".jpg" if path.lower().endswith((".jpg", ".jpeg")) else ".png"
        target = _ensure_extension(path, suffix)
        if not _save_snapshot(self, target, label):
            QtWidgets.QMessageBox.warning(self, "Save failed", "Unable to save the layout image.")
            return
        self._store_export_dir(str(Path(target).parent))
        log_action(f"Saved volume layout to {target}")

    def set_axis_labels(self, slice_label: str, row_label: str, column_label: str):
        self._axis_labels.update(
            {
                "slice": slice_label or "Slice axis",
                "row": row_label or "Row axis",
                "column": column_label or "Column axis",
                "value": "Value",
            }
        )
        for key, label_widget in self._curve_labels.items():
            label = self._axis_labels.get(key, key.title())
            if key == "value":
                text = label
            else:
                pretty = str(label)
                lower = pretty.lower()
                if lower.endswith("axis"):
                    text = pretty
                else:
                    text = f"{pretty} axis"
            label_widget.setText(text)

    def _store_curve(self, key: str, points: List[Tuple[float, float]]):
        xs = np.array([max(0.0, min(1.0, float(x))) for x, _ in points], dtype=float)
        ys = np.array([max(0.0, min(1.0, float(y))) for _, y in points], dtype=float)
        if xs.size == 0 or ys.size == 0:
            xs = np.array([0.0, 1.0], dtype=float)
            ys = np.array([0.0, 1.0], dtype=float)
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        normalized_points = [(float(x), float(y)) for x, y in zip(xs, ys)]
        previous = self._curve_points.get(key)
        self._curve_points[key] = normalized_points
        self._curve_lut_x[key] = xs
        self._curve_lut_y[key] = ys
        changed = previous is None or len(previous) != len(normalized_points) or any(
            abs(px - nx) > 1e-6 or abs(py - ny) > 1e-6
            for (px, py), (nx, ny) in zip(previous or [], normalized_points)
        )
        if changed:
            self._update_volume_visual()

    def _on_alpha_curve_changed(self, key: str, points: List[Tuple[float, float]]):
        if key not in self._curve_keys:
            key = "value"
        if not points:
            default_value = 0.25
            widget = self._curve_widgets.get(key)
            if widget is not None and hasattr(widget, "_default_value"):
                default_value = float(getattr(widget, "_default_value"))
            default_value = max(0.0, min(1.0, default_value))
            points = [(0.0, default_value), (1.0, default_value)]
        self._store_curve(key, points)

    def _on_reset_curve(self):
        for widget in self._curve_widgets.values():
            widget.reset_curve()
        for key, widget in self._curve_widgets.items():
            self._store_curve(key, widget.curve_points())

    def _on_reset_view(self):
        self._center_volume_item()
        self._reset_camera()

    def _sample_alpha_curve(self, key: str, values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, 0.0, 1.0)
        flat = clipped.reshape(-1)
        lut_x = self._curve_lut_x.get(key)
        lut_y = self._curve_lut_y.get(key)
        if lut_x is None or lut_y is None or lut_x.size < 2:
            mapped = flat
        else:
            mapped = np.interp(flat, lut_x, lut_y)
        return mapped.reshape(clipped.shape)

    def _apply_alpha_scale(self, alpha_norm: np.ndarray) -> np.ndarray:
        base = max(2.0, float(self._alpha_scale_base))
        clamped = np.clip(alpha_norm, 0.0, 1.0)
        scaled = (np.power(base, clamped) - 1.0) / (base - 1.0)
        return np.clip(scaled, 0.0, 1.0)

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.closed.emit()
        except Exception:
            pass
        super().closeEvent(event)
