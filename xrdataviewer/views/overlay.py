from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide2 import QtCore, QtGui, QtWidgets

from app_logging import log_action
from xr_plot_widget import (
    LineStyleConfig,
    PlotAnnotationConfig,
    apply_plotitem_annotation,
    clone_line_style,
    plotitem_annotation_state,
)

from ..annotations import LineStyleDialog, PlotAnnotationDialog
from ..datasets import (
    MemoryDatasetRegistry,
    MemorySliceRef,
    MemoryVarRef,
    VarRef,
    decode_mime_payloads,
)
from ..processing import (
    ParameterForm,
    ProcessingManager,
    apply_processing_step,
    list_processing_functions,
)
from ..preferences import PreferencesManager
from ..utils import ask_layout_label, ensure_extension, process_events, sanitize_filename, save_snapshot


def _compute_line_rect(xs: np.ndarray, ys: np.ndarray) -> QtCore.QRectF:
    xs = np.asarray(xs, float).reshape(-1)
    ys = np.asarray(ys, float).reshape(-1)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if not mask.any():
        return QtCore.QRectF(0.0, 0.0, 1.0, 1.0)
    x_vals = xs[mask]
    y_vals = ys[mask]
    x0 = float(np.nanmin(x_vals))
    x1 = float(np.nanmax(x_vals))
    y0 = float(np.nanmin(y_vals))
    y1 = float(np.nanmax(y_vals))
    if not np.isfinite(x0) or not np.isfinite(x1):
        x0, x1 = 0.0, float(xs.size - 1 if xs.size > 1 else 1.0)
    if not np.isfinite(y0) or not np.isfinite(y1):
        y0, y1 = 0.0, 1.0
    if x0 == x1:
        pad = 0.5 if x_vals.size <= 1 else max(1e-6, abs(x0) * 0.01)
        x0 -= pad
        x1 += pad
    if y0 == y1:
        pad = 0.5 if y_vals.size <= 1 else max(1e-6, abs(y0) * 0.05)
        y0 -= pad
        y1 += pad
    return QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)


class OverlayLayer(QtCore.QObject):
    def __init__(
        self,
        view: "OverlayView",
        title: str,
        data: np.ndarray,
        rect: QtCore.QRectF | None,
        *,
        kind: str = "image",
        x_values: Optional[np.ndarray] = None,
        line_style: Optional[LineStyleConfig] = None,
    ):
        super().__init__(view)
        self.view = view
        self.title = title
        self.kind = kind if kind in {"image", "line"} else "image"
        self.rect = rect
        self.colormap_name = "viridis"
        self.visible = True
        self.opacity = 1.0
        self.current_processing = "none"
        self.processing_params: dict = {}
        self.widget: Optional["OverlayLayerWidget"] = None
        self.legend_proxy: pg.PlotDataItem
        self.image_item: Optional[pg.ImageItem]
        self.graphics_item: pg.GraphicsObject
        self._levels: Tuple[float, float]
        self._line_style = clone_line_style(line_style)
        self._line_x = np.array([], dtype=float)

        if self.is_line_layer():
            self.base_data = np.asarray(data, float).reshape(-1)
            self.processed_data = np.array(self.base_data, copy=True)
            self._line_x = self._normalise_line_x(x_values, self.processed_data.size)
            self.opacity = self._line_style.normalized_opacity()
            self.graphics_item = pg.PlotDataItem()
            try:
                self.graphics_item.setName(self.title)
            except Exception:
                pass
            self.graphics_item.setVisible(True)
            self.image_item = None
            self.legend_proxy = self.graphics_item
            self._levels = (0.0, 1.0)
            self._update_line_item(autorange=False)
        else:
            self.base_data = np.asarray(data, float)
            self.processed_data = np.array(self.base_data, copy=True)
            self.graphics_item = pg.ImageItem()
            self.image_item = self.graphics_item
            self.graphics_item.setImage(self.processed_data, autoLevels=False)
            if rect is not None:
                try:
                    self.graphics_item.setRect(rect)
                except Exception:
                    pass
            self.graphics_item.setOpacity(1.0)
            self.graphics_item.setVisible(True)
            self._levels = self._compute_levels(self.processed_data)
            try:
                self.graphics_item.setLevels(self._levels)
            except Exception:
                pass
            self.legend_proxy = pg.PlotDataItem([0], [0])
            try:
                self.legend_proxy.setPen(pg.mkPen((220, 220, 220)))
            except Exception:
                pass
            self.set_colormap(self.colormap_name)

    # ---------- helpers ----------
    def is_line_layer(self) -> bool:
        return self.kind == "line"

    def supports_colormap(self) -> bool:
        return not self.is_line_layer()

    def supports_levels(self) -> bool:
        return not self.is_line_layer()

    def line_style_config(self) -> LineStyleConfig:
        return clone_line_style(self._line_style)

    def set_line_style(self, style: LineStyleConfig):
        if not self.is_line_layer():
            return
        self._line_style = clone_line_style(style)
        self.opacity = self._line_style.normalized_opacity()
        self._update_line_item(autorange=False)
        if self.widget:
            self.widget.update_opacity_label(self.opacity)
            try:
                block = self.widget.sld_opacity.blockSignals(True)
                self.widget.sld_opacity.setValue(int(round(self.opacity * 100)))
                self.widget.sld_opacity.blockSignals(block)
            except Exception:
                pass

    def _compute_levels(self, data: np.ndarray) -> Tuple[float, float]:
        data = np.asarray(data, float)
        finite = np.isfinite(data)
        if not finite.any():
            return (0.0, 1.0)
        vals = data[finite]
        try:
            lo = float(np.nanmin(vals))
            hi = float(np.nanmax(vals))
        except Exception:
            lo, hi = 0.0, 1.0
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = 1.0
        if hi == lo:
            hi = lo + 1.0
        return (lo, hi)

    def set_widget(self, widget: "OverlayLayerWidget"):
        self.widget = widget
        widget.update_from_layer()

    # ---------- layer controls ----------
    def set_visible(self, on: bool):
        self.visible = bool(on)
        try:
            self.graphics_item.setVisible(self.visible)
        except Exception:
            pass

    def set_opacity(self, alpha: float):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self.opacity = alpha
        if self.is_line_layer():
            self._line_style.opacity = alpha
            self._update_line_item(autorange=False)
        else:
            try:
                self.graphics_item.setOpacity(alpha)
            except Exception:
                pass
        if self.widget:
            self.widget.update_opacity_label(alpha)

    def set_colormap(self, name: str):
        self.colormap_name = name or "viridis"
        if not self.supports_colormap():
            return
        try:
            cmap = pg.colormap.get(self.colormap_name)
            if hasattr(cmap, "getLookupTable"):
                lut = cmap.getLookupTable(0.0, 1.0, 256)
                self.graphics_item.setLookupTable(lut)
        except Exception:
            pass

    def set_levels(self, lo: float, hi: float, *, update_widget: bool = True):
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        if hi <= lo:
            hi = lo + max(abs(lo) * 1e-6, 1e-6)
        self._levels = (float(lo), float(hi))
        if not self.supports_levels():
            return
        try:
            self.graphics_item.setLevels(self._levels)
        except Exception:
            pass
        if update_widget and self.widget:
            self.widget.update_level_spins(self._levels[0], self._levels[1])

    def auto_levels(self):
        if not self.supports_levels():
            return
        lo, hi = self._compute_levels(self.processed_data)
        self.set_levels(lo, hi, update_widget=True)

    def legend_item(self):
        return self.legend_proxy

    def apply_processing(self, mode: str, params: dict):
        data = np.asarray(self.base_data, float)
        mode = mode or "none"
        params = dict(params or {})

        if mode.startswith("pipeline:"):
            name = mode.split(":", 1)[1]
            manager = getattr(self.view, "processing_manager", None)
            pipeline = manager.get_pipeline(name) if manager else None
            if pipeline is None:
                QtWidgets.QMessageBox.warning(self.view, "Processing failed", f"Pipeline '{name}' is not available.")
                return
            try:
                data = pipeline.apply(data)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self.view, "Processing failed", str(e))
                return
            params = {}
        elif mode != "none":
            try:
                data = apply_processing_step(mode, data, params)
            except KeyError:
                QtWidgets.QMessageBox.warning(self.view, "Processing failed", f"Unknown processing mode: {mode}")
                return
            except Exception as e:
                QtWidgets.QMessageBox.warning(self.view, "Processing failed", str(e))
                return

        self.current_processing = mode
        self.processing_params = dict(params)
        if self.is_line_layer():
            self.processed_data = np.asarray(data, float).reshape(-1)
            self._update_line_item(autorange=False)
        else:
            self.processed_data = np.asarray(data, float)
            try:
                self.graphics_item.setImage(self.processed_data, autoLevels=False)
            except Exception:
                pass
            self.auto_levels()

    def get_display_rect(self):
        if self.is_line_layer():
            return self.rect
        rect = self.rect
        if rect is None and isinstance(self.graphics_item, pg.ImageItem):
            try:
                rect = self.graphics_item.mapRectToParent(self.graphics_item.boundingRect())
            except Exception:
                rect = None
        return rect

    # ---------- line helpers ----------
    def _normalise_line_x(self, values: Optional[np.ndarray], length: int) -> np.ndarray:
        if length <= 0:
            return np.array([], dtype=float)
        if values is None:
            return np.linspace(0.0, float(max(length - 1, 1)), length, dtype=float) if length > 1 else np.zeros(1)
        arr = np.asarray(values, float).reshape(-1)
        if arr.size == length:
            return arr.astype(float, copy=False)
        if arr.size == 0:
            return np.linspace(0.0, float(max(length - 1, 1)), length, dtype=float) if length > 1 else np.zeros(1)
        start = float(arr[0])
        end = float(arr[-1]) if arr.size > 1 else start + 1.0
        if not np.isfinite(start) or not np.isfinite(end):
            start, end = 0.0, float(length - 1 if length > 1 else 1.0)
        if start == end:
            end = start + 1.0
        return np.linspace(start, end, length, dtype=float)

    def _resample_line_axis(self, xs: np.ndarray, length: int) -> np.ndarray:
        xs = np.asarray(xs, float).reshape(-1)
        if length <= 0:
            return np.array([], dtype=float)
        if xs.size == length:
            return xs
        if xs.size == 0:
            return np.linspace(0.0, float(max(length - 1, 1)), length, dtype=float) if length > 1 else np.zeros(1)
        start = float(xs[0])
        end = float(xs[-1]) if xs.size > 1 else start + 1.0
        if not np.isfinite(start) or not np.isfinite(end):
            start, end = 0.0, float(length - 1 if length > 1 else 1.0)
        if start == end:
            end = start + 1.0
        return np.linspace(start, end, length, dtype=float)

    def _update_line_item(self, autorange: bool):
        if not self.is_line_layer():
            return
        ys = np.asarray(self.processed_data, float).reshape(-1)
        if ys.size != self._line_x.size:
            self._line_x = self._resample_line_axis(self._line_x, ys.size)
        xs = np.asarray(self._line_x, float)
        style = self._line_style or LineStyleConfig()
        y_plot = ys
        x_plot = xs
        step_mode = style.curve_mode == "step"
        if style.curve_mode == "smooth" and ys.size >= 3:
            try:
                window = style.smooth_window(ys.size)
                y_plot = pg.functions.smooth(ys, window=window)
            except Exception:
                y_plot = ys
        color = style.effective_color()
        color.setAlphaF(style.normalized_opacity())
        width = max(0.1, float(style.width))
        pen = pg.mkPen(color, width=width)
        try:
            pen_style = {
                "solid": QtCore.Qt.SolidLine,
                "dashed": QtCore.Qt.DashLine,
                "dotted": QtCore.Qt.DotLine,
                "dashdot": QtCore.Qt.DashDotLine,
            }.get(style.pen_style, QtCore.Qt.SolidLine)
            pen.setStyle(pen_style)
        except Exception:
            pass

        kwargs: Dict[str, object] = {}
        if step_mode:
            kwargs["stepMode"] = True
            x_plot = self._step_edges(xs)

        marker_brush: Optional[QtGui.QBrush] = None
        marker_pen: Optional[QtGui.QPen] = None
        marker_size: Optional[int] = None
        symbol = None
        if style.markers and not step_mode:
            key = str(style.marker_style).strip().lower()
            symbol_map = {
                "o": "o",
                "circle": "o",
                "●": "o",
                "•": "o",
                "s": "s",
                "square": "s",
                "□": "s",
                "t": "t",
                "triangle": "t",
                "triangleup": "t",
                "^": "t",
                "d": "d",
                "diamond": "d",
                "+": "+",
                "plus": "+",
                "x": "x",
                "cross": "x",
            }
            symbol = symbol_map.get(key, "o")
            marker_color = QtGui.QColor(color)
            marker_pen = QtGui.QPen(marker_color)
            marker_pen.setWidthF(max(1.0, width * 0.75))
            marker_brush = QtGui.QBrush(marker_color)
            marker_size = max(1, int(style.marker_size))
        try:
            if symbol:
                self.graphics_item.setData(
                    x_plot,
                    y_plot,
                    pen=pen,
                    symbol=symbol,
                    symbolBrush=marker_brush,
                    symbolPen=marker_pen,
                    symbolSize=marker_size,
                    **kwargs,
                )
            else:
                self.graphics_item.setData(x_plot, y_plot, pen=pen, symbol=None, **kwargs)
        except Exception:
            pass
        try:
            self.graphics_item.setVisible(self.visible)
        except Exception:
            pass
        self.rect = _compute_line_rect(x_plot, y_plot)

    def _step_edges(self, xs: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, float)
        n = xs.size
        if n == 0:
            return xs
        if n == 1:
            x0 = float(xs[0])
            return np.array([x0 - 0.5, x0 + 0.5], dtype=float)
        diffs = np.diff(xs)
        if np.all(diffs > 0):
            edges = np.empty(n + 1, dtype=float)
            edges[1:-1] = xs[:-1] + diffs / 2.0
            edges[0] = xs[0] - diffs[0] / 2.0
            edges[-1] = xs[-1] + diffs[-1] / 2.0
            return edges
        edges = np.empty(n + 1, dtype=float)
        edges[:-1] = xs
        edges[-1] = xs[-1]
        return edges

class OverlayLayerWidget(QtWidgets.QGroupBox):
    def __init__(self, view: "OverlayView", layer: OverlayLayer):
        super().__init__(layer.title)
        self.view = view
        self.layer = layer
        self._ready = False

        self.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        )

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(6)

        # Visibility / remove
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        self.chk_visible = QtWidgets.QCheckBox("Visible")
        self.chk_visible.setChecked(True)
        self.chk_visible.toggled.connect(self._on_visibility)
        header.addWidget(self.chk_visible)
        self.btn_style = QtWidgets.QToolButton()
        self.btn_style.setText("Style…")
        self.btn_style.clicked.connect(self._on_style)
        header.addWidget(self.btn_style)
        header.addStretch(1)
        self.btn_remove = QtWidgets.QToolButton()
        self.btn_remove.setText("✕")
        self.btn_remove.setToolTip("Remove layer")
        self.btn_remove.clicked.connect(self._on_remove)
        header.addWidget(self.btn_remove)
        lay.addLayout(header)

        # Colormap selection
        self._colormap_row = QtWidgets.QWidget()
        cmap_row = QtWidgets.QHBoxLayout(self._colormap_row)
        cmap_row.setContentsMargins(0, 0, 0, 0)
        cmap_row.setSpacing(6)
        cmap_row.addWidget(QtWidgets.QLabel("Colormap:"))
        self.cmb_colormap = QtWidgets.QComboBox()
        self.cmb_colormap.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_colormap.setMinimumContentsLength(12)
        self.cmb_colormap.setMinimumWidth(200)
        try:
            cmaps = sorted(pg.colormap.listMaps())
        except Exception:
            cmaps = ["viridis", "plasma", "magma", "cividis", "gray"]
        for name in cmaps:
            self.cmb_colormap.addItem(name)
        self.cmb_colormap.currentTextChanged.connect(self._on_colormap)
        cmap_row.addWidget(self.cmb_colormap, 1)
        lay.addWidget(self._colormap_row)

        # Levels controls
        self._levels_row = QtWidgets.QWidget()
        lvl_row = QtWidgets.QHBoxLayout(self._levels_row)
        lvl_row.setContentsMargins(0, 0, 0, 0)
        lvl_row.setSpacing(6)
        lvl_row.addWidget(QtWidgets.QLabel("Levels:"))
        self.spin_min = QtWidgets.QDoubleSpinBox()
        self.spin_min.setDecimals(6)
        self.spin_min.setRange(-1e12, 1e12)
        self.spin_min.valueChanged.connect(self._on_levels_changed)
        lvl_row.addWidget(self.spin_min)
        lvl_row.addWidget(QtWidgets.QLabel("→"))
        self.spin_max = QtWidgets.QDoubleSpinBox()
        self.spin_max.setDecimals(6)
        self.spin_max.setRange(-1e12, 1e12)
        self.spin_max.valueChanged.connect(self._on_levels_changed)
        lvl_row.addWidget(self.spin_max)
        self.btn_autoscale = QtWidgets.QPushButton("Auto")
        self.btn_autoscale.clicked.connect(self._on_autoscale)
        lvl_row.addWidget(self.btn_autoscale)
        lay.addWidget(self._levels_row)

        # Opacity slider
        opacity_row = QtWidgets.QHBoxLayout()
        opacity_row.addWidget(QtWidgets.QLabel("Opacity:"))
        self.sld_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_opacity.setRange(0, 100)
        self.sld_opacity.setValue(100)
        self.sld_opacity.valueChanged.connect(self._on_opacity)
        opacity_row.addWidget(self.sld_opacity, 1)
        self.lbl_opacity = QtWidgets.QLabel("100%")
        opacity_row.addWidget(self.lbl_opacity)
        lay.addLayout(opacity_row)

        # Processing controls
        manager_ref = getattr(view, "processing_manager", None)
        self.manager: Optional[ProcessingManager] = None
        proc_box = QtWidgets.QGroupBox("Processing")
        proc_box.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        )
        proc_layout = QtWidgets.QVBoxLayout(proc_box)
        proc_layout.setContentsMargins(6, 6, 6, 6)
        proc_layout.setSpacing(4)

        proc_row = QtWidgets.QHBoxLayout()
        proc_row.addWidget(QtWidgets.QLabel("Operation:"))
        self.cmb_processing = QtWidgets.QComboBox()
        self.cmb_processing.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_processing.setMinimumContentsLength(14)
        self.cmb_processing.setMinimumWidth(220)
        self.cmb_processing.currentIndexChanged.connect(self._on_processing_mode_changed)
        proc_row.addWidget(self.cmb_processing, 1)
        proc_layout.addLayout(proc_row)

        self.param_stack = QtWidgets.QStackedWidget()
        self.param_stack.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        )
        self._function_forms: Dict[str, ParameterForm] = {}
        self._function_indices: Dict[str, int] = {}
        self._none_index: int = 0
        self._pipeline_summary_index: int = 0
        self.lbl_pipeline_summary = QtWidgets.QLabel()
        self._rebuild_forms_for_dims()

        proc_layout.addWidget(self.param_stack)
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.clicked.connect(self._apply_processing)
        proc_layout.addWidget(self.btn_apply, alignment=QtCore.Qt.AlignRight)
        lay.addWidget(proc_box)
        self.set_processing_manager(manager_ref)

        self._ready = True
        self._on_processing_mode_changed()

    # ---------- UI helpers ----------
    def update_from_layer(self):
        self._ready = False
        self.setTitle(self.layer.title)
        self.chk_visible.setChecked(self.layer.visible)
        is_line = self.layer.is_line_layer()
        self.btn_style.setVisible(is_line)
        supports_cmap = self.layer.supports_colormap()
        self._colormap_row.setVisible(supports_cmap)
        self.cmb_colormap.setEnabled(supports_cmap)
        supports_levels = self.layer.supports_levels()
        self._levels_row.setVisible(supports_levels)
        self.spin_min.setEnabled(supports_levels)
        self.spin_max.setEnabled(supports_levels)
        self.btn_autoscale.setEnabled(supports_levels)
        if supports_cmap:
            self._set_colormap_selection(self.layer.colormap_name)
        if supports_levels:
            lo, hi = getattr(self.layer, "_levels", (0.0, 1.0))
            self.update_level_spins(lo, hi)
        self.update_opacity_label(self.layer.opacity)
        block = self.sld_opacity.blockSignals(True)
        self.sld_opacity.setValue(int(round(self.layer.opacity * 100)))
        self.sld_opacity.blockSignals(block)
        current_mode = self.layer.current_processing or "none"
        self._refresh_processing_options()
        self._select_processing_mode(current_mode)
        if current_mode.startswith("pipeline:"):
            name = current_mode.split(":", 1)[1]
            self._update_pipeline_summary(name)
        else:
            form = self._function_forms.get(current_mode)
            if form:
                form.set_values(self.layer.processing_params)
        self._ready = True
        self._apply_processing()

    def _set_colormap_selection(self, name: str):
        if not self.layer.supports_colormap():
            return
        idx = self.cmb_colormap.findText(name, QtCore.Qt.MatchFixedString)
        if idx < 0:
            idx = self.cmb_colormap.findText("viridis", QtCore.Qt.MatchFixedString)
        if idx >= 0:
            block = self.cmb_colormap.blockSignals(True)
            self.cmb_colormap.setCurrentIndex(idx)
            self.cmb_colormap.blockSignals(block)

    def update_level_spins(self, lo: float, hi: float):
        if not self.layer.supports_levels():
            return
        block = self.spin_min.blockSignals(True)
        self.spin_min.setValue(float(lo))
        self.spin_min.blockSignals(block)
        block = self.spin_max.blockSignals(True)
        self.spin_max.setValue(float(hi))
        self.spin_max.blockSignals(block)

    def update_opacity_label(self, alpha: float):
        pct = int(round(float(alpha) * 100))
        self.lbl_opacity.setText(f"{pct}%")

    def processing_parameters(self) -> dict:
        data = self._current_processing_data()
        if data.get("type") == "function":
            form = self._function_forms.get(data.get("key", ""))
            if form:
                return form.values()
        return {}

    def current_processing(self) -> str:
        data = self._current_processing_data()
        if data.get("type") == "function":
            return data.get("key", "none")
        if data.get("type") == "pipeline":
            return f"pipeline:{data.get('name', '')}"
        return "none"

    def _current_processing_data(self) -> Dict[str, object]:
        data = self.cmb_processing.currentData()
        if isinstance(data, dict):
            return data
        return {"type": "none"}

    def _find_data_index(self, target: Dict[str, object]) -> int:
        for idx in range(self.cmb_processing.count()):
            data = self.cmb_processing.itemData(idx)
            if isinstance(data, dict) and data.get("type") == target.get("type"):
                if data.get("type") == "function" and data.get("key") == target.get("key"):
                    return idx
                if data.get("type") == "pipeline" and data.get("name") == target.get("name"):
                    return idx
                if data.get("type") == "none":
                    return idx
        return -1

    def _current_dims(self) -> Tuple[str, ...]:
        if self.layer.is_line_layer():
            return ("line",)
        data = getattr(self.layer, "base_data", None)
        if data is None:
            return ()
        return tuple(f"axis{i}" for i in range(np.ndim(data)))

    def _rebuild_forms_for_dims(self):
        while self.param_stack.count():
            widget = self.param_stack.widget(0)
            self.param_stack.removeWidget(widget)
            widget.deleteLater()
        self._function_forms.clear()
        self._function_indices.clear()
        none_widget = QtWidgets.QWidget()
        self._none_index = self.param_stack.addWidget(none_widget)
        dims = self._current_dims()
        ndim = len(dims) if dims else None
        for spec in list_processing_functions(ndim):
            form = ParameterForm(spec.parameters_for_data(dims))
            form.parametersChanged.connect(self._on_processing_params_changed)
            idx = self.param_stack.addWidget(form)
            self._function_forms[spec.key] = form
            self._function_indices[spec.key] = idx
        summary_container = QtWidgets.QWidget()
        summary_layout = QtWidgets.QVBoxLayout(summary_container)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(0)
        self.lbl_pipeline_summary = QtWidgets.QLabel("Select a pipeline to view steps.")
        self.lbl_pipeline_summary.setWordWrap(True)
        self.lbl_pipeline_summary.setStyleSheet("color: #666;")
        summary_layout.addWidget(self.lbl_pipeline_summary)
        self._pipeline_summary_index = self.param_stack.addWidget(summary_container)

    def _refresh_processing_options(self):
        current = self._current_processing_data()
        self._rebuild_forms_for_dims()
        dims = self._current_dims()
        ndim = len(dims) if dims else None
        block = self.cmb_processing.blockSignals(True)
        self.cmb_processing.clear()
        self.cmb_processing.addItem("None", {"type": "none"})
        for spec in list_processing_functions(ndim):
            self.cmb_processing.addItem(spec.label, {"type": "function", "key": spec.key})
        if self.manager:
            for name in self.manager.pipeline_names():
                self.cmb_processing.addItem(f"Pipeline: {name}", {"type": "pipeline", "name": name})
        idx = self._find_data_index(current)
        if idx < 0:
            idx = self._find_data_index({"type": "none"})
        if idx < 0:
            idx = 0
        self.cmb_processing.setCurrentIndex(idx)
        self.cmb_processing.blockSignals(block)
        self._on_processing_mode_changed()

    def _update_pipeline_summary(self, name: str):
        if not self.manager:
            self.lbl_pipeline_summary.setText("No processing manager available.")
            return
        pipeline = self.manager.get_pipeline(name)
        if pipeline is None:
            self.lbl_pipeline_summary.setText(f"Pipeline '{name}' not found.")
            return
        lines = []
        for i, step in enumerate(pipeline.steps, start=1):
            spec = get_processing_function(step.key)
            label = spec.label if spec else step.key
            params = summarize_parameters(step.key, step.params)
            if params:
                lines.append(f"{i}. {label} ({params})")
            else:
                lines.append(f"{i}. {label}")
        self.lbl_pipeline_summary.setText("\n".join(lines))

    def set_processing_manager(self, manager: Optional[ProcessingManager]):
        if getattr(self, "manager", None) is manager:
            return
        if getattr(self, "manager", None):
            try:
                self.manager.pipelines_changed.disconnect(self._on_pipelines_changed)
            except Exception:
                pass
        self.manager = manager
        if manager:
            manager.pipelines_changed.connect(self._on_pipelines_changed)
        self._refresh_processing_options()

    def _on_pipelines_changed(self):
        self._refresh_processing_options()

    # ---------- Slots ----------
    def _on_visibility(self, on: bool):
        self.layer.set_visible(on)
        if on:
            self.view.auto_view_range()

    def _on_remove(self):
        self.view.remove_layer(self.layer)

    def _on_colormap(self, name: str):
        if not self._ready or not self.layer.supports_colormap():
            return
        self.layer.set_colormap(name)

    def _on_levels_changed(self):
        if not self._ready or not self.layer.supports_levels():
            return
        lo = self.spin_min.value()
        hi = self.spin_max.value()
        if hi <= lo:
            hi = lo + max(abs(lo) * 1e-6, 1e-6)
            block = self.spin_max.blockSignals(True)
            self.spin_max.setValue(hi)
            self.spin_max.blockSignals(block)
        self.layer.set_levels(lo, hi, update_widget=False)

    def _on_autoscale(self):
        if not self.layer.supports_levels():
            return
        self.layer.auto_levels()

    def _on_opacity(self, value: int):
        alpha = float(value) / 100.0
        self.layer.set_opacity(alpha)

    def _on_style(self):
        if not self.layer.is_line_layer():
            return
        dialog = LineStyleDialog(self, initial=self.layer.line_style_config())
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        style = dialog.line_style()
        if style is None:
            return
        self.layer.set_line_style(style)
        block = self.sld_opacity.blockSignals(True)
        self.sld_opacity.setValue(int(round(self.layer.opacity * 100)))
        self.sld_opacity.blockSignals(block)
        self.update_opacity_label(self.layer.opacity)
        self.view.auto_view_range()

    def _on_processing_mode_changed(self):
        data = self._current_processing_data()
        if data.get("type") == "function":
            idx = self._function_indices.get(data.get("key", ""), self._none_index)
            try:
                self.param_stack.setCurrentIndex(idx)
            except Exception:
                pass
        elif data.get("type") == "pipeline":
            self._update_pipeline_summary(str(data.get("name", "")))
            try:
                self.param_stack.setCurrentIndex(self._pipeline_summary_index)
            except Exception:
                pass
        else:
            try:
                self.param_stack.setCurrentIndex(self._none_index)
            except Exception:
                pass
        self._apply_processing()

    def _on_processing_params_changed(self, *_):
        self._apply_processing()

    def _apply_processing(self):
        if not self._ready:
            return
        mode = self.current_processing()
        params = self.processing_parameters()
        self.layer.apply_processing(mode, params)

    def _select_processing_mode(self, mode: str):
        if mode.startswith("pipeline:"):
            name = mode.split(":", 1)[1]
            target = {"type": "pipeline", "name": name}
        elif mode and mode != "none":
            target = {"type": "function", "key": mode}
        else:
            target = {"type": "none"}
        idx = self._find_data_index(target)
        if idx < 0:
            idx = 0
        block = self.cmb_processing.blockSignals(True)
        self.cmb_processing.setCurrentIndex(idx)
        self.cmb_processing.blockSignals(block)
        self._on_processing_mode_changed()

class OverlayView(QtWidgets.QWidget):
    def __init__(
        self,
        processing_manager: Optional[ProcessingManager] = None,
        preferences: Optional[PreferencesManager] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.layers: List[OverlayLayer] = []
        self.processing_manager: Optional[ProcessingManager] = processing_manager
        self.preferences: Optional[PreferencesManager] = None
        self._annotation_config: Optional[PlotAnnotationConfig] = None
        self._legend_item: Optional[pg.LegendItem] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(10)

        def _make_group(title: str, widgets: Iterable[QtWidgets.QWidget]) -> QtWidgets.QGroupBox:
            box = QtWidgets.QGroupBox(title)
            layout = QtWidgets.QHBoxLayout(box)
            layout.setContentsMargins(8, 6, 8, 6)
            layout.setSpacing(6)
            for widget in widgets:
                layout.addWidget(widget)
            return box

        self.btn_auto_view = QtWidgets.QPushButton("Auto view")
        self.btn_auto_view.clicked.connect(self.auto_view_range)
        self.btn_clear = QtWidgets.QPushButton("Clear layers")
        self.btn_clear.clicked.connect(self.clear_layers)
        self.btn_export = QtWidgets.QToolButton()
        self.btn_export.setText("Export")
        self.btn_export.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        export_menu = QtWidgets.QMenu(self.btn_export)
        act_composite = export_menu.addAction("Save composite image…")
        act_composite.triggered.connect(self._export_composite)
        act_layers = export_menu.addAction("Save layers to folder…")
        act_layers.triggered.connect(self._export_individual_layers)
        export_menu.addSeparator()
        act_layout = export_menu.addAction("Save overlay layout…")
        act_layout.triggered.connect(self._export_full_layout)
        self.btn_export.setMenu(export_menu)
        self.btn_annotations = QtWidgets.QPushButton("Set annotations…")
        self.btn_annotations.clicked.connect(self._open_annotation_dialog)

        toolbar.addWidget(_make_group("Layers", (self.btn_clear,)))
        toolbar.addWidget(_make_group("Scaling", (self.btn_auto_view,)))
        toolbar.addWidget(_make_group("Style", (self.btn_annotations,)))
        toolbar.addWidget(_make_group("Export", (self.btn_export,)))
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # Layer controls panel
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(360)
        panel.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        )
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)

        self.layer_scroll = QtWidgets.QScrollArea()
        self.layer_scroll.setWidgetResizable(True)
        self.layer_list_widget = QtWidgets.QWidget()
        self.layer_list_layout = QtWidgets.QVBoxLayout(self.layer_list_widget)
        self.layer_list_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_list_layout.setSpacing(6)
        self.layer_list_layout.addStretch(1)
        self.layer_scroll.setWidget(self.layer_list_widget)
        panel_layout.addWidget(self.layer_scroll, 1)

        self.lbl_hint = QtWidgets.QLabel("Drag datasets here to overlay them.")
        self.lbl_hint.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_hint.setWordWrap(True)
        panel_layout.addWidget(self.lbl_hint)

        splitter.addWidget(panel)

        # Plot area
        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.invertY(False)
        self.plot.showGrid(x=False, y=False)
        self.plot.setLabel("left", "Y")
        self.plot.setLabel("bottom", "X")
        splitter.addWidget(self.glw)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        QtCore.QTimer.singleShot(0, lambda: splitter.setSizes([420, 780]))

        self.set_preferences(preferences)

    # ---------- drag & drop ----------
    def dragEnterEvent(self, ev):
        ev.acceptProposedAction() if ev.mimeData().hasText() else ev.ignore()

    def dropEvent(self, ev):
        payload = ev.mimeData().text()
        payloads = decode_mime_payloads(payload)
        if not payloads:
            payloads = [payload]

        added = False
        for entry in payloads:
            if self._add_payload(entry):
                added = True

        if added:
            self._apply_preferences_to_layers()
            self._update_hint()
            self.auto_view_range()
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def _add_payload(self, payload: str) -> bool:
        vr = VarRef.from_mime(payload)
        mem_var = None if vr else MemoryVarRef.from_mime(payload)
        slice_ref = None if (vr or mem_var) else MemorySliceRef.from_mime(payload)
        if not vr and not mem_var and not slice_ref:
            return False
        try:
            if vr:
                da, coords = vr.load()
                label = f"{vr.path.name}:{vr.var}"
            elif mem_var:
                da, coords = mem_var.load()
                label = f"{MemoryDatasetRegistry.get_label(mem_var.dataset_key)}:{mem_var.var}"
            else:
                da, coords, alias = slice_ref.load()
                label = f"{slice_ref.display_label()}:{alias}"
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(e))
            return False

        prepared = self._prepare_layer(da, coords)
        if prepared is None:
            return False
        if prepared.get("type") == "line":
            style = LineStyleConfig()
            try:
                style.color = QtGui.QColor(pg.intColor(len(self.layers)))
            except Exception:
                pass
            layer = OverlayLayer(
                self,
                label,
                prepared.get("y", np.array([], dtype=float)),
                prepared.get("rect"),
                kind="line",
                x_values=prepared.get("x"),
                line_style=style,
            )
        else:
            layer = OverlayLayer(
                self,
                label,
                prepared.get("data"),
                prepared.get("rect"),
                kind="image",
            )
        self.plot.addItem(layer.graphics_item)
        widget = OverlayLayerWidget(self, layer)
        widget.set_processing_manager(self.processing_manager)
        layer.set_widget(widget)
        self.layers.append(layer)
        self._insert_layer_widget(widget)
        self._apply_legend_config(self._annotation_config)
        return True

    # ---------- layer management ----------
    def _insert_layer_widget(self, widget: OverlayLayerWidget):
        stretch = self.layer_list_layout.itemAt(self.layer_list_layout.count() - 1)
        if stretch and stretch.spacerItem():
            self.layer_list_layout.insertWidget(self.layer_list_layout.count() - 1, widget)
        else:
            self.layer_list_layout.addWidget(widget)

    def remove_layer(self, layer: OverlayLayer):
        if layer in self.layers:
            self.layers.remove(layer)
        try:
            self.plot.removeItem(layer.graphics_item)
        except Exception:
            pass
        if layer.widget:
            w = layer.widget
            layer.widget = None
            w.setParent(None)
            w.deleteLater()
        self._update_hint()
        self.auto_view_range()
        self._apply_legend_config(self._annotation_config)

    def clear_layers(self):
        for layer in list(self.layers):
            self.remove_layer(layer)

    def _open_annotation_dialog(self):
        initial = plotitem_annotation_state(self.plot)
        if self._annotation_config is not None:
            initial = replace(self._annotation_config, apply_to_all=False)
        dialog = PlotAnnotationDialog(self, initial=initial, allow_apply_all=False)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        config = dialog.annotation_config()
        if config is None:
            return
        config = replace(config, apply_to_all=False)
        self._annotation_config = config
        apply_plotitem_annotation(
            self.plot,
            config,
            background_widget=self.glw,
        )
        self._apply_legend_config(config)

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
        self._apply_preferences_to_layers()

    def _on_preferences_changed(self, _data):
        self._apply_preferences_to_layers()

    def _apply_preferences_to_layers(self):
        if not self.preferences:
            return
        preferred = self.preferences.preferred_colormap(None)
        if not preferred:
            return
        fallback_names = {"", "default", "viridis"}
        for layer in self.layers:
            if layer.colormap_name not in fallback_names:
                continue
            if preferred != layer.colormap_name:
                layer.set_colormap(preferred)
                if layer.widget is not None:
                    layer.widget._set_colormap_selection(preferred)

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

    # ---------- export helpers ----------
    def _export_composite(self):
        if not self.layers:
            QtWidgets.QMessageBox.information(
                self,
                "No layers",
                "Add a layer before exporting the composite image.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save composite image",
            self._initial_path("overlay-composite.png"),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        suffix = ".jpg" if path.lower().endswith((".jpg", ".jpeg")) else ".png"
        target = ensure_extension(path, suffix)
        if not save_snapshot(self.glw, target):
            QtWidgets.QMessageBox.warning(self, "Save failed", "Unable to save the composite image.")
            return
        self._store_export_dir(str(Path(target).parent))
        log_action(f"Saved overlay composite to {target}")

    def _export_individual_layers(self):
        if not self.layers:
            QtWidgets.QMessageBox.information(
                self,
                "No layers",
                "Add a layer before exporting individual images.",
            )
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select export folder",
            self._default_export_dir(),
        )
        if not directory:
            return
        base = Path(directory)
        self._store_export_dir(directory)
        original_states = [(layer, bool(layer.visible)) for layer in self.layers]
        count = 0
        try:
            for idx, layer in enumerate(self.layers, start=1):
                for other in self.layers:
                    other.set_visible(other is layer)
                process_events()
                name = sanitize_filename(layer.title) or f"layer_{idx}"
                target = base / f"{name}_{idx:02d}.png"
                if save_snapshot(self.glw, target):
                    count += 1
        finally:
            for layer, state in original_states:
                layer.set_visible(state)
            process_events()
            self.auto_view_range()
        if count == 0:
            QtWidgets.QMessageBox.warning(self, "Export failed", "No layers were exported.")
            return
        QtWidgets.QMessageBox.information(
            self,
            "Export complete",
            f"Saved {count} layer image(s) to {base}",
        )
        log_action(f"Exported {count} overlay layers to {base}")

    def _export_full_layout(self):
        if not self.layers:
            QtWidgets.QMessageBox.information(
                self,
                "No layers",
                "Add a layer before exporting the layout.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save overlay layout",
            self._initial_path("overlay-layout.png"),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        ok, label = ask_layout_label(self, "Layout label", self._default_layout_label())
        if not ok:
            return
        suffix = ".jpg" if path.lower().endswith((".jpg", ".jpeg")) else ".png"
        target = ensure_extension(path, suffix)
        if not save_snapshot(self, target, label):
            QtWidgets.QMessageBox.warning(self, "Save failed", "Unable to save the layout image.")
            return
        self._store_export_dir(str(Path(target).parent))
        log_action(f"Saved overlay layout to {target}")

    def set_processing_manager(self, manager: Optional[ProcessingManager]):
        self.processing_manager = manager
        for layer in self.layers:
            if layer.widget:
                layer.widget.set_processing_manager(manager)

    def auto_view_range(self):
        rects = []
        for layer in self.layers:
            if not layer.visible:
                continue
            rect = layer.get_display_rect()
            if rect is None or rect.isNull():
                continue
            rects.append(rect)
        if not rects:
            return
        union = rects[0]
        for r in rects[1:]:
            try:
                union = union.united(r)
            except Exception:
                pass
        try:
            self.plot.vb.setRange(rect=union, padding=0.0)
        except Exception:
            pass

    def _update_hint(self):
        self.lbl_hint.setVisible(not self.layers)

    def _ensure_legend_item(self) -> pg.LegendItem:
        if self._legend_item is None:
            legend = pg.LegendItem(offset=(10, 10))
            try:
                legend.setParentItem(self.plot.vb)
            except Exception:
                legend.setParentItem(self.plot.graphicsItem())
            self._legend_item = legend
        return self._legend_item

    def _release_legend_item(self):
        if self._legend_item is None:
            return
        legend = self._legend_item
        self._legend_item = None
        try:
            legend.hide()
        except Exception:
            pass
        try:
            scene = legend.scene()
        except Exception:
            scene = None
        if scene is not None:
            try:
                scene.removeItem(legend)
            except Exception:
                pass
        try:
            legend.setParentItem(None)
        except Exception:
            pass

    def _apply_legend_config(self, config: Optional[PlotAnnotationConfig]):
        if config is None or not config.legend_visible or not self.layers:
            self._release_legend_item()
            return
        legend = self._ensure_legend_item()
        try:
            legend.clear()
        except Exception:
            pass
        entries = list(config.legend_entries or [])
        if not entries:
            entries = [layer.title for layer in self.layers]
        for idx, layer in enumerate(self.layers):
            label = entries[idx] if idx < len(entries) else layer.title
            item = layer.legend_item()
            try:
                legend.addItem(item, label)
            except Exception:
                proxy = pg.PlotDataItem([0], [0])
                proxy.setPen(pg.mkPen((200, 200, 200)))
                legend.addItem(proxy, label)
        anchor_map = {
            "top-left": ((0, 0), (0, 0)),
            "top-right": ((1, 0), (1, 0)),
            "bottom-left": ((0, 1), (0, 1)),
            "bottom-right": ((1, 1), (1, 1)),
        }
        pos = anchor_map.get(config.legend_position, anchor_map["top-right"])
        try:
            legend.anchor(pos[0], pos[1])
        except Exception:
            pass
        try:
            legend.show()
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            self._release_legend_item()
        except Exception:
            pass
        super().closeEvent(event)

    # ---------- data prep ----------
    def _prepare_layer(self, da, coords):
        coords = coords or {}
        values = getattr(da, "values", da)
        Z = np.asarray(values, float)
        if Z.ndim == 0:
            data = np.asarray(Z, float).reshape(1, 1)
            rect = self._rect_to_qrectf(0.0, 1.0, 0.0, 1.0)
            return {"type": "image", "data": data, "rect": rect}
        if Z.ndim == 1:
            y_vals = np.asarray(Z, float).reshape(-1)
            x_vals = self._line_axis_from_coords(coords, y_vals.size)
            rect = _compute_line_rect(x_vals, y_vals)
            return {"type": "line", "x": x_vals, "y": y_vals, "rect": rect}
        if "X" in coords and "Y" in coords:
            data, rect = self._resample_warped(coords["X"], coords["Y"], Z)
            return {"type": "image", "data": data, "rect": rect}
        if "x" in coords and "y" in coords:
            data, rect = self._resample_rectilinear(coords["x"], coords["y"], Z)
            return {"type": "image", "data": data, "rect": rect}
        data = np.asarray(Z, float)
        Ny, Nx = data.shape
        rect = self._rect_to_qrectf(0.0, float(Nx), 0.0, float(Ny))
        return {"type": "image", "data": data, "rect": rect}

    def _line_axis_from_coords(self, coords: Dict[str, object], length: int) -> np.ndarray:
        if length <= 0:
            return np.array([], dtype=float)
        if not isinstance(coords, dict):
            coords = {}
        candidates = (
            "line_values",
            "line",
            "values",
            "x",
            "X",
            "index",
            "indices",
            "sample",
            "samples",
            "y",
            "Y",
            "row_values",
            "col_values",
        )
        for key in candidates:
            if key not in coords:
                continue
            raw = coords.get(key)
            try:
                arr = np.asarray(getattr(raw, "values", raw), float)
            except Exception:
                continue
            if arr.ndim == 0:
                continue
            arr = arr.reshape(-1)
            if arr.size == length:
                return arr.astype(float, copy=False)
        if length == 1:
            return np.zeros(1)
        return np.linspace(0.0, float(length - 1), length, dtype=float)

    def _rect_to_qrectf(self, x0, x1, y0, y1):
        from PySide2.QtCore import QRectF
        return QRectF(float(x0), float(y0), float(x1 - x0), float(y1 - y0))

    def _resample_rectilinear(self, x1, y1, Z):
        x1 = np.asarray(x1, float)
        y1 = np.asarray(y1, float)
        Z = np.asarray(Z, float)
        Ny, Nx = Z.shape
        xs = np.argsort(x1)
        ys = np.argsort(y1)
        x_sorted = x1[xs]
        y_sorted = y1[ys]
        Zs = Z[np.ix_(ys, xs)]
        x_uni = np.linspace(x_sorted[0], x_sorted[-1], Nx)
        y_uni = np.linspace(y_sorted[0], y_sorted[-1], Ny)
        Zx = np.empty((Ny, Nx), float)
        for i in range(Ny):
            Zx[i, :] = np.interp(x_uni, x_sorted, Zs[i, :], left=np.nan, right=np.nan)
        Zu = np.empty((Ny, Nx), float)
        for j in range(Nx):
            col = Zx[:, j]
            m = np.isfinite(col)
            if m.sum() >= 2:
                Zu[:, j] = np.interp(y_uni, y_sorted[m], col[m], left=np.nan, right=np.nan)
            else:
                Zu[:, j] = np.nan
        rect = self._rect_to_qrectf(x_uni[0], x_uni[-1], y_uni[0], y_uni[-1])
        return Zu, rect

    def _resample_warped(self, X, Y, Z):
        try:
            from scipy.interpolate import griddata
        except Exception:
            rect = self._rect_to_qrectf(0.0, float(Z.shape[1]), 0.0, float(Z.shape[0]))
            return np.asarray(Z, float), rect
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        Z = np.asarray(Z, float)
        Ny, Nx = Z.shape
        xmin, xmax = np.nanmin(X), np.nanmax(X)
        ymin, ymax = np.nanmin(Y), np.nanmax(Y)
        x_t = np.linspace(xmin, xmax, Nx)
        y_t = np.linspace(ymin, ymax, Ny)
        XX, YY = np.meshgrid(x_t, y_t)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        vals = Z.ravel()
        Zu = griddata(pts, vals, (XX, YY), method="linear")
        if np.isnan(Zu).any():
            Zun = griddata(pts, vals, (XX, YY), method="nearest")
            mask = np.isnan(Zu)
            Zu[mask] = Zun[mask]
        rect = self._rect_to_qrectf(x_t[0], x_t[-1], y_t[0], y_t[-1])
        return np.asarray(Zu, float), rect
