from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import xarray as xr
from PySide2 import QtCore, QtGui, QtWidgets

from app_logging import log_action
from xr_coords import guess_phys_coords
from xr_plot_widget import (
    CentralPlotWidget,
    LineStyleConfig,
    PlotAnnotationConfig,
    apply_plotitem_annotation,
    clone_line_style,
    plotitem_annotation_state,
)

from ..annotations import LineStyleDialog, PlotAnnotationDialog
from ..colormaps import available_colormap_names, get_colormap, is_scientific_colormap
from ..datasets import (
    DataSetRef,
    HighDimVarRef,
    MemoryDatasetRef,
    MemoryDatasetRegistry,
    MemorySliceRef,
    MemoryVarRef,
    VarRef,
    decode_mime_payloads,
)
from ..processing import apply_processing_step, ProcessingManager, ProcessingSelectionDialog
from ..preferences import PreferencesManager
from ..utils import ask_layout_label, ensure_extension, sanitize_filename, save_snapshot, open_dataset


class ViewerFrame(QtWidgets.QFrame):
    request_close = QtCore.Signal(object)
    display_mode_changed = QtCore.Signal(str)

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setObjectName("viewerFrame")
        self.setProperty("selected", False)
        self.setStyleSheet(
            "QFrame#viewerFrame { border: 1px solid #888; border-radius: 2px; }"
            "QFrame#viewerFrame[selected=\"true\"] { border: 2px solid #1d72b8; "
            "background-color: rgba(29, 114, 184, 40); }"
        )

        self._base_title = title
        self._raw_data: Optional[np.ndarray] = None
        self._processed_data: Optional[np.ndarray] = None
        self._coords: Dict[str, np.ndarray] = {}
        self._line_x: Optional[np.ndarray] = None
        self._line_label: str = ""
        self._display_mode: str = "image"
        self._current_processing: str = "none"
        self._processing_params: Dict[str, object] = {}
        self._selected: bool = False
        self._dataset_path: Optional[Path] = None
        self._dataset: Optional[xr.Dataset] = None
        self._available_variables: List[str] = []
        self._variable_hints: Dict[str, str] = {}
        self._current_variable: Optional[str] = None
        self.preferences: Optional[PreferencesManager] = None
        self._line_style = LineStyleConfig()

        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(2,2,2,2); lay.setSpacing(2)
        # Header
        hdr = QtWidgets.QFrame(); hl = QtWidgets.QHBoxLayout(hdr); hl.setContentsMargins(6,3,6,3)
        self.lbl = QtWidgets.QLabel(title); hl.addWidget(self.lbl, 1)
        self._line_style_btn = QtWidgets.QToolButton()
        self._line_style_btn.setText("Style…")
        self._line_style_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self._line_style_btn.clicked.connect(self._open_line_style_dialog)
        self._line_style_btn.setVisible(False)
        hl.addWidget(self._line_style_btn, 0)
        self._crosshair_btn = QtWidgets.QToolButton()
        self._crosshair_btn.setText("Crosshair")
        self._crosshair_btn.setCheckable(True)
        self._crosshair_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self._crosshair_btn.toggled.connect(self._on_crosshair_toggled)
        hl.addWidget(self._crosshair_btn, 0)
        btn_close = QtWidgets.QToolButton(); btn_close.setText("×")
        btn_close.clicked.connect(lambda: self.request_close.emit(self))
        hl.addWidget(btn_close, 0)
        lay.addWidget(hdr, 0)

        # Variable selector
        selector = QtWidgets.QFrame()
        selector_layout = QtWidgets.QHBoxLayout(selector)
        selector_layout.setContentsMargins(6, 0, 6, 0)
        selector_layout.setSpacing(6)
        self._var_label = QtWidgets.QLabel("Variable:")
        self._var_label.setEnabled(False)
        selector_layout.addWidget(self._var_label, 0)
        self.cmb_variable = QtWidgets.QComboBox()
        self.cmb_variable.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cmb_variable.addItem("Select variable…", None)
        self.cmb_variable.setEnabled(False)
        self.cmb_variable.currentIndexChanged.connect(self._on_variable_combo_changed)
        selector_layout.addWidget(self.cmb_variable, 1)
        lay.addWidget(selector, 0)
        self._variable_bar = selector

        # Center: image on left, histogram on right (toggle visibility)
        self.center_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.center_split.setChildrenCollapsible(False)
        self.center_split.setHandleWidth(6)
        lay.addWidget(self.center_split, 1)

        self.viewer = CentralPlotWidget(self)
        try:
            self.viewer.sigLocalCrosshairToggled.connect(self._on_viewer_crosshair_toggled)
            self._on_viewer_crosshair_toggled(self.viewer.local_crosshair_enabled())
        except Exception:
            pass
        try:
            self.viewer.set_line_style(self._line_style)
        except Exception:
            pass
        self.center_split.addWidget(self.viewer)

        self._hist_master_enabled = True
        self._hist_local_enabled = True
        self._hist_last_split_sizes: Optional[List[int]] = None
        self._hist_widget = self.viewer.histogram_widget()
        if self._hist_widget is not None:
            if self.center_split.indexOf(self._hist_widget) == -1:
                self.center_split.addWidget(self._hist_widget)
            try:
                self.center_split.setStretchFactor(0, 1)
                self.center_split.setStretchFactor(1, 0)
            except Exception:
                pass
        try:
            self.center_split.splitterMoved.connect(self._record_histogram_sizes)
        except Exception:
            pass
        try:
            self.viewer.configure_histogram_toggle(
                getter=self.is_histogram_local_enabled,
                setter=self.set_histogram_local_enabled,
            )
        except Exception:
            pass
        self._update_histogram_visibility()
        self._update_line_style_button()

    def _dataset_display_name(self) -> str:
        if self._dataset_path is not None:
            return self._dataset_path.name
        return self._base_title or "Dataset"

    def _set_header_text(self, variable: Optional[str], *, missing: bool = False, custom: Optional[str] = None):
        if custom is not None:
            self.lbl.setText(custom)
            return
        base = self._dataset_display_name()
        if variable:
            suffix = f"{variable} (missing)" if missing else variable
            self.lbl.setText(f"{base} — {suffix}")
        else:
            self.lbl.setText(base)

    def _update_variable_combo(self):
        block = self.cmb_variable.blockSignals(True)
        current = self._current_variable if self._current_variable in self._available_variables else None
        self.cmb_variable.clear()
        self.cmb_variable.addItem("Select variable…", None)
        for var in self._available_variables:
            label = f"{var}{self._variable_hints.get(var, '')}"
            self.cmb_variable.addItem(label, var)
        self.cmb_variable.blockSignals(block)
        has_vars = bool(self._available_variables)
        self.cmb_variable.setEnabled(has_vars)
        self._var_label.setEnabled(has_vars)
        self._select_combo_value(current)

    def _select_combo_value(self, value: Optional[str]):
        block = self.cmb_variable.blockSignals(True)
        if value:
            idx = self.cmb_variable.findData(value)
            if idx >= 0:
                self.cmb_variable.setCurrentIndex(idx)
            else:
                self.cmb_variable.setCurrentIndex(0)
        else:
            self.cmb_variable.setCurrentIndex(0)
        self.cmb_variable.blockSignals(block)

    def _dispose_dataset(self):
        if self._dataset is not None:
            try:
                close = getattr(self._dataset, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
        self._dataset = None

    def dispose(self):
        self._dispose_dataset()

    def line_style_config(self) -> LineStyleConfig:
        return clone_line_style(self._line_style)

    def set_line_style_config(self, config: LineStyleConfig, *, refresh: bool = True):
        self._line_style = clone_line_style(config)
        try:
            self.viewer.set_line_style(self._line_style, refresh=refresh)
        except Exception:
            pass

    def _open_line_style_dialog(self):
        dialog = LineStyleDialog(self, initial=self.line_style_config())
        dialog.applied.connect(self._apply_line_style_dialog_result)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        style = dialog.line_style()
        if style is None:
            return
        self._apply_line_style_dialog_result(style)

    def _apply_line_style_dialog_result(self, style: LineStyleConfig) -> None:
        if style is None:
            return
        self.set_line_style_config(style, refresh=True)
        try:
            name = self._dataset_display_name()
        except Exception:
            name = "viewer"
        log_action(f"Updated line style for {name}")

    def is_line_display(self) -> bool:
        return self._display_mode == "line"

    def apply_colormap(self, name: str) -> bool:
        if self._display_mode == "line":
            return False
        cmap = get_colormap(name)
        if cmap is None:
            return False
        try:
            self.viewer.lut.gradient.setColorMap(cmap)
            self.viewer.lut.rehide_stops()
        except Exception:
            return False
        return True

    def set_dataset(self, dataset: xr.Dataset, path: Path, *, select: Optional[str] = None):
        self._dispose_dataset()
        self._dataset = dataset
        self._dataset_path = Path(path) if path is not None else None
        self._available_variables = []
        self._variable_hints = {}
        if self._dataset_path is not None:
            self.lbl.setToolTip(str(self._dataset_path))
        else:
            self.lbl.setToolTip("")

        try:
            data_vars = getattr(dataset, "data_vars", {})
        except Exception:
            data_vars = {}
        for var in data_vars:
            try:
                da = dataset[var]
            except Exception:
                continue
            ndim = getattr(da, "ndim", 0)
            if ndim not in (1, 2):
                continue
            self._available_variables.append(var)
            dims: List[str] = []
            try:
                if ndim == 1:
                    dim = da.dims[0] if getattr(da, "dims", None) else "axis"
                    sizes = getattr(da, "sizes", {})
                    size = sizes.get(dim) if isinstance(sizes, dict) else None
                    if size is None:
                        shape = getattr(da, "shape", ())
                        if shape:
                            size = shape[0]
                    if size is None:
                        size = "?"
                    dims = [f"{dim}[{size}]"]
                else:
                    dims = [
                        f"{dim}[{getattr(da, 'sizes', {}).get(dim, '?')}]"
                        for dim in da.dims[:2]
                    ]
            except Exception:
                dims = []
            if dims:
                joiner = " × " if len(dims) > 1 else ", "
                self._variable_hints[var] = " (" + joiner.join(dims) + ")"

        self._available_variables.sort()
        self._current_variable = None
        self._update_variable_combo()
        self._clear_display()
        if not self._available_variables:
            self._set_header_text(None)
            self._set_header_text(
                None,
                custom=f"{self._dataset_display_name()} — No 1D/2D variables",
            )
            return

        self._set_header_text(None)
        if select and select in self._available_variables:
            self.plot_variable(select)
        else:
            self._select_combo_value(None)

    def available_variables(self) -> List[str]:
        return list(self._available_variables)

    def current_variable(self) -> Optional[str]:
        return self._current_variable

    def _on_variable_combo_changed(self, index: int):
        data = self.cmb_variable.itemData(index)
        if data is None:
            if self._current_variable is not None:
                self._current_variable = None
                self._clear_display()
            else:
                self._clear_display()
            return
        self.plot_variable(str(data))

    def plot_variable(self, var_name: str) -> bool:
        name = str(var_name or "").strip()
        if not name:
            self._current_variable = None
            self._clear_display()
            self._select_combo_value(None)
            return False
        success = self._load_variable(name)
        if success:
            self._select_combo_value(name)
            return True
        self._show_missing_variable(name)
        self._select_combo_value(None)
        return False

    def _show_missing_variable(self, var_name: str):
        self._current_variable = None
        self._clear_display(preserve_header=True)
        self._set_header_text(var_name, missing=True)

    def _load_variable(self, var_name: str) -> bool:
        if self._dataset is None:
            return False
        try:
            da = self._dataset[var_name]
        except Exception:
            return False
        ndim = getattr(da, "ndim", 0)
        if ndim not in (1, 2):
            return False
        coords = guess_phys_coords(da)
        self.set_data(da, coords)
        self._current_variable = var_name
        self._set_header_text(var_name)
        return True

    def set_data(self, da, coords):
        values = np.asarray(getattr(da, "values", da), float)
        self._raw_data = np.asarray(values, float)
        self._processed_data = np.asarray(values, float)
        self._coords = {}
        self._line_x = None
        self._line_label = ""
        coords = dict(coords or {})
        if values.ndim == 1:
            self._display_mode = "line"
            axis = da.dims[0] if getattr(da, "ndim", 0) else "index"
            self._line_label = axis or "index"
            x_data = None
            for key in ("x", "X", axis):
                arr = coords.get(key)
                if arr is not None:
                    try:
                        x_data = np.asarray(arr, float)
                        break
                    except Exception:
                        x_data = None
            if x_data is None:
                try:
                    coord = da.coords.get(axis)
                    if coord is not None:
                        x_data = np.asarray(coord.values, float)
                except Exception:
                    x_data = None
            if x_data is None:
                x_data = np.arange(values.size, dtype=float)
            self._coords["line_x"] = np.asarray(x_data, float)
            self._line_x = self._coords["line_x"]
        elif "X" in coords and "Y" in coords:
            self._display_mode = "warped"
            self._coords["X"] = np.asarray(coords["X"], float)
            self._coords["Y"] = np.asarray(coords["Y"], float)
        elif "x" in coords and "y" in coords:
            self._display_mode = "rectilinear"
            self._coords["x"] = np.asarray(coords["x"], float)
            self._coords["y"] = np.asarray(coords["y"], float)
        else:
            self._display_mode = "image"
        self._current_processing = "none"
        self._processing_params = {}
        self._display_data(self._processed_data, autorange=True)
        if self._display_mode != "line":
            try:
                self.viewer.img_item.setVisible(True)
            except Exception:
                pass
        self._apply_preferences()

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
            self._apply_preferences()

    def _on_preferences_changed(self, _data):
        self._apply_preferences()

    def _apply_preferences(self):
        prefs = self.preferences
        if prefs is None:
            return
        try:
            self.viewer.set_value_precision(prefs.value_precision())
        except Exception:
            pass
        if self._display_mode == "line":
            return
        cmap_name = prefs.preferred_colormap(self._current_variable)
        if cmap_name:
            self.apply_colormap(cmap_name)
        if prefs.autoscale_on_load():
            try:
                self.viewer.autoscale_levels()
            except Exception:
                pass

    def set_selected(self, selected: bool):
        selected = bool(selected)
        if self._selected == selected:
            return
        self._selected = selected
        self.setProperty("selected", selected)
        try:
            self.style().unpolish(self)
            self.style().polish(self)
        except Exception:
            pass
        self.update()

    def is_selected(self) -> bool:
        return bool(self._selected)

    def apply_processing(self, mode: str, params: Dict[str, object], manager: Optional["ProcessingManager"]):
        if self._raw_data is None:
            return
        data = np.asarray(self._raw_data, float)
        mode = mode or "none"
        params = dict(params or {})

        if mode.startswith("pipeline:"):
            if not manager:
                raise RuntimeError("No processing manager is available for pipelines.")
            name = mode.split(":", 1)[1]
            pipeline = manager.get_pipeline(name)
            if pipeline is None:
                raise RuntimeError(f"Pipeline '{name}' is not available.")
            processed = pipeline.apply(data)
            params = {}
        elif mode != "none":
            processed = apply_processing_step(mode, data, params)
        else:
            processed = data

        self._processed_data = np.asarray(processed, float)
        self._current_processing = mode
        self._processing_params = dict(params)
        self._display_data(self._processed_data, autorange=True)

    def reset_processing(self):
        if self._raw_data is None:
            return
        self._processed_data = np.asarray(self._raw_data, float)
        self._current_processing = "none"
        self._processing_params = {}
        self._display_data(self._processed_data, autorange=True)

    def _display_data(self, data: np.ndarray, *, autorange: bool = False):
        if self._display_mode == "line":
            x = self._line_x if self._line_x is not None else np.arange(data.size, dtype=float)
            try:
                self.viewer.set_line_style(self._line_style, refresh=False)
            except Exception:
                pass
            self.viewer.set_line(x, data, autorange=autorange)
            y_label = self._current_variable or "Value"
            x_label = self._line_label or "Index"
            try:
                self.viewer.set_labels(x_label, y_label)
            except Exception:
                pass
            self.viewer.set_legend_sources([(self.viewer.line_item(), y_label)])
            try:
                self.display_mode_changed.emit("line")
            except Exception:
                pass
            self._update_line_style_button()
            return

        if self._display_mode == "warped" and "X" in self._coords and "Y" in self._coords:
            self.viewer.set_warped(self._coords["X"], self._coords["Y"], data, autorange=autorange)
        elif self._display_mode == "rectilinear" and "x" in self._coords and "y" in self._coords:
            self.viewer.set_rectilinear(self._coords["x"], self._coords["y"], data, autorange=autorange)
        else:
            self.viewer.set_image(data, autorange=autorange)
        label = self._current_variable or "Image"
        self.viewer.set_legend_sources([(self.viewer.image_item(), label)])
        try:
            self.display_mode_changed.emit(self._display_mode)
        except Exception:
            pass
        self._update_line_style_button()

    def set_histogram_visible(self, on: bool):
        self._hist_master_enabled = bool(on)
        self._update_histogram_visibility()

    def set_histogram_local_enabled(self, on: bool):
        enabled = bool(on)
        if self._hist_local_enabled == enabled:
            return
        self._hist_local_enabled = enabled
        self._update_histogram_visibility()

    def is_histogram_local_enabled(self) -> bool:
        return bool(self._hist_local_enabled)

    def _record_histogram_sizes(self, *_):
        if not self._hist_widget or not self._hist_widget.isVisible():
            return
        try:
            sizes = self.center_split.sizes()
        except Exception:
            return
        if len(sizes) < 2 or sizes[1] <= 0:
            return
        self._hist_last_split_sizes = list(sizes)

    def _update_histogram_visibility(self):
        hist_widget = self._hist_widget or self.viewer.histogram_widget()
        if hist_widget is None:
            return
        if self.center_split.indexOf(hist_widget) == -1:
            self.center_split.addWidget(hist_widget)
        want_visible = self._hist_master_enabled and self._hist_local_enabled
        if want_visible:
            hist_widget.setMinimumWidth(80)
            hist_widget.setMaximumWidth(16777215)
            hist_widget.show()
            sizes = self._hist_last_split_sizes
            if sizes and len(sizes) >= 2 and sizes[1] > 0:
                try:
                    self.center_split.setSizes(list(sizes))
                except Exception:
                    pass
            else:
                try:
                    current = self.center_split.sizes()
                except Exception:
                    current = []
                if len(current) < 2 or current[1] == 0:
                    total = sum(current) if len(current) >= 2 else 0
                    if total <= 0:
                        total = 400
                    hist_width = max(120, total // 4)
                    main_width = max(120, total - hist_width)
                    try:
                        self.center_split.setSizes([int(main_width), int(hist_width)])
                    except Exception:
                        pass
        else:
            if hist_widget.isVisible():
                try:
                    sizes = self.center_split.sizes()
                except Exception:
                    sizes = []
                if len(sizes) >= 2 and sizes[1] > 0:
                    self._hist_last_split_sizes = list(sizes)
            hist_widget.hide()
            hist_widget.setMinimumWidth(0)
            hist_widget.setMaximumWidth(0)

    def _clear_display(self, *, preserve_header: bool = False):
        self._raw_data = None
        self._processed_data = None
        self._coords = {}
        self._display_mode = "image"
        try:
            blank = np.zeros((1, 1), dtype=float)
            self.viewer.img_item.setImage(blank, autoLevels=False)
            try:
                self.viewer.img_item.setLevels((0.0, 1.0))
            except Exception:
                pass
        except Exception:
            pass
        try:
            self.viewer.img_item.setVisible(False)
        except Exception:
            pass
        try:
            self.viewer.hide_crosshair()
            self.viewer.clear_mirrored_crosshair()
        except Exception:
            pass
        self._update_line_style_button()
        if not preserve_header:
            self._set_header_text(None)

    def _update_line_style_button(self):
        try:
            visible = self._display_mode == "line"
        except Exception:
            visible = False
        btn = getattr(self, "_line_style_btn", None)
        if not isinstance(btn, QtWidgets.QToolButton):
            return
        btn.setVisible(visible)
        btn.setEnabled(visible)

    def processing_dims(self) -> Tuple[str, ...]:
        if self._display_mode == "line":
            label = self._line_label or self._current_variable or "line"
            return (str(label),)
        if self._dataset is not None and self._current_variable in self._dataset:
            try:
                dims = getattr(self._dataset[self._current_variable], "dims", None)
                if dims:
                    return tuple(str(d) for d in dims[: np.ndim(self._raw_data)])
            except Exception:
                pass
        data = self._processed_data if self._processed_data is not None else self._raw_data
        if data is None:
            return ()
        return tuple(f"axis{i}" for i in range(np.ndim(data)))

    def annotation_defaults(self) -> PlotAnnotationConfig:
        return self.viewer.annotation_defaults()

    def apply_annotation(self, config: PlotAnnotationConfig):
        self.viewer.apply_annotation(config)

    def _on_crosshair_toggled(self, enabled: bool):
        try:
            self.viewer.set_local_crosshair_enabled(enabled)
        except Exception:
            pass

    def _on_viewer_crosshair_toggled(self, enabled: bool):
        block = self._crosshair_btn.blockSignals(True)
        try:
            self._crosshair_btn.setChecked(bool(enabled))
        finally:
            self._crosshair_btn.blockSignals(block)

class MultiViewGrid(QtWidgets.QWidget):
    """
    A splitter-based grid of ViewerFrame tiles.
    - Drag a dataset or variable reference from the DatasetsPane onto this widget to add a tile.
    - 'Columns' spinbox controls how many tiles per row.
    - 'Show histograms' toggles the classic HistogramLUTItem to the right of each tile.
    """
    def __init__(
        self,
        processing_manager: Optional[ProcessingManager] = None,
        preferences: Optional[PreferencesManager] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.frames: List[ViewerFrame] = []
        self.processing_manager = processing_manager
        self.preferences: Optional[PreferencesManager] = None
        self._selected_frames: Set[ViewerFrame] = set()
        self._mouse_down = False
        self._drag_select_active = False
        self._drag_select_add = True

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        def _tag_compact_buttons(*widgets: QtWidgets.QWidget) -> None:
            for widget in widgets:
                if isinstance(widget, (QtWidgets.QPushButton, QtWidgets.QToolButton)):
                    widget.setProperty("sizeVariant", "compact")
                    widget.setMinimumWidth(0)
                    widget.setMinimumHeight(0)
                    widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        # Toolbar
        bar = QtWidgets.QGridLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setHorizontalSpacing(8)
        bar.setVerticalSpacing(6)

        def _make_group(title: str, widgets: Iterable[QtWidgets.QWidget]) -> QtWidgets.QWidget:
            items = tuple(widgets)
            if not items:
                spacer = QtWidgets.QWidget()
                spacer.setVisible(False)
                return spacer
            if len(items) == 1:
                widget = items[0]
                if isinstance(widget, QtWidgets.QAbstractButton):
                    widget.setToolTip(title)
                    _tag_compact_buttons(widget)
                return widget

            frame = QtWidgets.QFrame()
            frame.setProperty("toolbarGroup", True)
            layout = QtWidgets.QVBoxLayout(frame)
            layout.setContentsMargins(8, 6, 8, 6)
            layout.setSpacing(4)

            label = QtWidgets.QLabel(title)
            label.setProperty("toolbarGroupLabel", True)
            layout.addWidget(label)

            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            for widget in items:
                _tag_compact_buttons(widget)
                row.addWidget(widget)
            layout.addLayout(row)
            return frame

        lbl_columns = QtWidgets.QLabel("Columns:")
        self.col_spin = QtWidgets.QSpinBox()
        self.col_spin.setRange(1, 12)
        self.col_spin.setValue(3)
        self.col_spin.setMaximumWidth(72)
        self.col_spin.valueChanged.connect(self._reflow)

        self.chk_show_hist = QtWidgets.QCheckBox("Show histograms")
        self.chk_show_hist.setChecked(True)  # set False if you prefer off-by-default
        self.chk_show_hist.toggled.connect(self._apply_histogram_visibility)

        self.chk_link_levels = QtWidgets.QCheckBox("Lock colorscales")
        self.chk_link_levels.toggled.connect(self._on_link_levels_toggled)

        self.chk_link_panzoom = QtWidgets.QCheckBox("Lock pan/zoom")
        self.chk_link_panzoom.toggled.connect(self._on_link_panzoom_toggled)

        self.chk_cursor_mirror = QtWidgets.QCheckBox("Mirror cursor")
        self.chk_cursor_mirror.setChecked(False)
        self.chk_cursor_mirror.toggled.connect(self._on_link_cursor_toggled)

        self.btn_autoscale = QtWidgets.QPushButton("Autoscale colors")
        self.btn_autoscale.clicked.connect(self._autoscale_colors)

        self.btn_autopan = QtWidgets.QPushButton("Auto pan/zoom")
        self.btn_autopan.clicked.connect(self._auto_panzoom)

        self.btn_equalize_rows = QtWidgets.QPushButton("Equalize rows")
        self.btn_equalize_rows.clicked.connect(self.equalize_rows)

        self.btn_equalize_cols = QtWidgets.QPushButton("Equalize columns")
        self.btn_equalize_cols.clicked.connect(self.equalize_columns)

        self.btn_select_all = QtWidgets.QPushButton("Select All Plots")
        self.btn_select_all.setEnabled(False)
        self.btn_select_all.clicked.connect(self._select_all_frames)

        self.btn_apply_processing = QtWidgets.QPushButton("Apply processing…")
        self.btn_apply_processing.setEnabled(False)
        self.btn_apply_processing.clicked.connect(self._on_apply_processing_clicked)

        self.btn_export = QtWidgets.QToolButton()
        self.btn_export.setText("Export")
        self.btn_export.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        export_menu = QtWidgets.QMenu(self.btn_export)
        self.act_export_active = export_menu.addAction("Save active plot…")
        self.act_export_active.triggered.connect(self._export_active_plot)
        self.act_export_selected = export_menu.addAction("Save selected plots to folder…")
        self.act_export_selected.triggered.connect(self._export_selected_plots)
        export_menu.addSeparator()
        self.act_export_layout = export_menu.addAction("Save entire layout…")
        self.act_export_layout.triggered.connect(self._export_layout_image)
        self.btn_export.setMenu(export_menu)

        self.btn_line_style = QtWidgets.QPushButton("Line style…")
        self.btn_line_style.setEnabled(False)
        self.btn_line_style.clicked.connect(self._open_line_style_dialog)

        self.btn_annotations = QtWidgets.QPushButton("Set annotations…")
        self.btn_annotations.clicked.connect(self._open_annotation_dialog)

        self.btn_set_colormap = QtWidgets.QPushButton("Set colormap…")
        self.btn_set_colormap.clicked.connect(self._open_colormap_dialog)

        toolbar_groups = [
            _make_group(
                "Layout",
                (lbl_columns, self.col_spin, self.btn_equalize_rows, self.btn_equalize_cols, self.btn_select_all),
            ),
            _make_group("Display", (self.chk_show_hist, self.chk_cursor_mirror)),
            _make_group(
                "Scaling",
                (self.chk_link_levels, self.chk_link_panzoom, self.btn_autoscale, self.btn_autopan),
            ),
            _make_group("Processing", (self.btn_apply_processing,)),
            _make_group("Style", (self.btn_line_style, self.btn_annotations)),
            _make_group("Color", (self.btn_set_colormap,)),
            _make_group("Export", (self.btn_export,)),
        ]

        columns = 3
        for index, widget in enumerate(toolbar_groups):
            row = index // columns
            column = index % columns
            bar.addWidget(widget, row, column)

        rows = max(1, (len(toolbar_groups) + columns - 1) // columns)
        spacer = QtWidgets.QSpacerItem(
            0,
            0,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Minimum,
        )
        bar.addItem(spacer, 0, columns, rows, 1)
        v.addLayout(bar)

        # A vertical splitter holds "rows" (each row is a horizontal splitter of tiles)
        self.vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.vsplit.setChildrenCollapsible(False)
        v.addWidget(self.vsplit, 1)

        self._level_handlers = {}
        self._view_handlers = {}
        self._cursor_handlers = {}
        self._syncing_levels = False
        self._syncing_views = False

        self.set_preferences(preferences)

    def set_preferences(self, preferences: Optional[PreferencesManager]):
        if self.preferences is preferences:
            return
        if self.preferences:
            try:
                self.preferences.changed.disconnect(self._on_preferences_changed)
            except Exception:
                pass
        self.preferences = preferences
        for frame in self.frames:
            frame.set_preferences(self.preferences)
        if self.preferences:
            try:
                self.preferences.changed.connect(self._on_preferences_changed)
            except Exception:
                pass
        self._on_preferences_changed(None)

    # ---------- Drag & Drop ----------
    def dragEnterEvent(self, ev: QtGui.QDragEnterEvent):
        ev.acceptProposedAction() if ev.mimeData().hasText() else ev.ignore()

    def dropEvent(self, ev: QtGui.QDropEvent):
        text = ev.mimeData().text()
        payloads = decode_mime_payloads(text)
        if not payloads:
            payloads = [text]

        frames: List[ViewerFrame] = []
        high_dim_dropped = False
        for payload in payloads:
            if HighDimVarRef.from_mime(payload):
                high_dim_dropped = True
                continue
            frame = self._build_frame_from_payload(payload)
            if frame is not None:
                frames.append(frame)

        if not frames:
            if high_dim_dropped:
                QtWidgets.QMessageBox.information(
                    self,
                    "Unsupported variable",
                    "High-dimensional variables should be sent to the Slice Data tab.",
                )
            ev.ignore()
            return

        if high_dim_dropped:
            QtWidgets.QMessageBox.information(
                self,
                "Unsupported variable",
                "Some dropped items were high-dimensional and were ignored.",
            )

        for fr in frames:
            self.frames.append(fr)
            self._connect_frame_signals(fr)
            self._sync_new_frame_to_links(fr)
            fr.set_selected(False)

        self._reflow()
        self._update_apply_button_state()
        ev.acceptProposedAction()

    def _build_frame_from_payload(self, payload: str) -> Optional[ViewerFrame]:
        ds_ref = DataSetRef.from_mime(payload)
        mem_ds = MemoryDatasetRef.from_mime(payload) if not ds_ref else None
        vr = None if (ds_ref or mem_ds) else VarRef.from_mime(payload)
        mem_var = None if (ds_ref or mem_ds or vr) else MemoryVarRef.from_mime(payload)
        slice_ref = None if (ds_ref or mem_ds or vr or mem_var) else MemorySliceRef.from_mime(payload)

        if not any([ds_ref, mem_ds, vr, mem_var, slice_ref]):
            return None

        dataset = None
        frame_title = ""
        alias: Optional[str] = None
        try:
            if ds_ref:
                dataset = ds_ref.load()
                frame_title = ds_ref.path.name
            elif mem_ds:
                dataset = mem_ds.load()
                frame_title = mem_ds.display_name()
            elif vr:
                dataset = open_dataset(vr.path)
                frame_title = vr.path.name
            elif mem_var:
                dataset = MemoryDatasetRegistry.get_dataset(mem_var.dataset_key)
                if dataset is None:
                    raise RuntimeError("Dataset is no longer available in memory")
                frame_title = MemoryDatasetRegistry.get_label(mem_var.dataset_key)
            elif slice_ref:
                arr, coords, alias = slice_ref.load()
                dataset = xr.Dataset({alias: arr})
                for key, value in coords.items():
                    try:
                        dataset[alias] = dataset[alias].assign_coords({key: value})
                    except Exception:
                        pass
                frame_title = slice_ref.display_label()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(e))
            return None

        fr = ViewerFrame(title=frame_title, parent=self)
        fr.set_preferences(self.preferences)
        fr.request_close.connect(self._remove_frame)
        fr.display_mode_changed.connect(self._on_frame_mode_changed)

        try:
            if ds_ref:
                fr.set_dataset(dataset, ds_ref.path)
            elif mem_ds:
                fr.set_dataset(dataset, None)
            elif vr:
                fr.set_dataset(dataset, vr.path, select=vr.var)
            elif mem_var:
                if mem_var.var not in dataset.data_vars:
                    raise RuntimeError("Selected variable is no longer available")
                fr.set_dataset(dataset, None, select=mem_var.var)
            elif slice_ref and alias:
                fr.set_dataset(dataset, None, select=alias)
        except Exception as exc:
            try:
                close = getattr(dataset, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            fr.deleteLater()
            return None

        return fr

    # ---------- Tile management ----------
    def _remove_frame(self, fr: ViewerFrame):
        self._disconnect_frame_signals(fr)
        if fr in self.frames:
            self.frames.remove(fr)
        if fr in self._selected_frames:
            self._selected_frames.remove(fr)
            fr.set_selected(False)
        try:
            fr.setParent(None)
        except Exception:
            pass
        try:
            fr.dispose()
        except Exception:
            pass
        self._reflow()
        self._update_apply_button_state()

    def selected_frames(self) -> List[ViewerFrame]:
        return [fr for fr in self.frames if fr in self._selected_frames]

    def _update_apply_button_state(self):
        has_frames = bool(self.frames)
        if hasattr(self, "btn_select_all"):
            self.btn_select_all.setEnabled(has_frames)
        if hasattr(self, "btn_apply_processing"):
            self.btn_apply_processing.setEnabled(bool(self._selected_frames))
        if hasattr(self, "btn_line_style"):
            targets = self.selected_frames() or list(self.frames)
            has_line = any(fr.is_line_display() for fr in targets)
            self.btn_line_style.setEnabled(has_line)

    def _select_all_frames(self):
        if not self.frames:
            self._clear_selection()
            self._update_apply_button_state()
            return
        self._clear_selection()
        for fr in self.frames:
            self._set_frame_selected(fr, True, clear=False)
        self._update_apply_button_state()

    def _set_frame_selected(self, frame: Optional[ViewerFrame], selected: bool, *, clear: bool = False):
        if frame is None or frame not in self.frames:
            if clear:
                self._clear_selection()
            return
        if clear:
            self._clear_selection(exclude=frame if selected else None)
        if selected:
            self._selected_frames.add(frame)
        else:
            self._selected_frames.discard(frame)
        frame.set_selected(selected)
        self._update_apply_button_state()

    def _clear_selection(self, exclude: Optional[ViewerFrame] = None):
        changed = False
        for fr in list(self._selected_frames):
            if exclude is not None and fr is exclude:
                continue
            fr.set_selected(False)
            self._selected_frames.discard(fr)
            changed = True
        if changed:
            self._update_apply_button_state()

    def _frame_at_global_pos(self, global_pos: QtCore.QPoint) -> Optional[ViewerFrame]:
        widget = QtWidgets.QApplication.widgetAt(global_pos)
        while widget is not None:
            if isinstance(widget, ViewerFrame):
                return widget if widget in self.frames else None
            widget = widget.parentWidget()
        return None

    def eventFilter(self, obj, event):
        if isinstance(event, QtGui.QMouseEvent):
            etype = event.type()
            if etype == QtCore.QEvent.MouseButtonPress:
                if event.button() == QtCore.Qt.LeftButton:
                    frame = self._frame_at_global_pos(event.globalPos())
                    if frame is not None:
                        ctrl = bool(event.modifiers() & QtCore.Qt.ControlModifier)
                        if ctrl:
                            target_state = frame not in self._selected_frames
                            self._set_frame_selected(frame, target_state, clear=False)
                            self._drag_select_active = True
                            self._drag_select_add = target_state
                        else:
                            self._set_frame_selected(frame, True, clear=True)
                            self._drag_select_active = False
                        self._mouse_down = True
                elif event.button() == QtCore.Qt.RightButton:
                    frame = self._frame_at_global_pos(event.globalPos())
                    if frame is not None:
                        if frame not in self._selected_frames:
                            self._set_frame_selected(frame, True, clear=True)
                        self._show_plot_context_menu(event.globalPos())
                        return True
            elif etype == QtCore.QEvent.MouseMove:
                if self._mouse_down and self._drag_select_active and event.buttons() & QtCore.Qt.LeftButton:
                    frame = self._frame_at_global_pos(event.globalPos())
                    if frame is not None:
                        self._set_frame_selected(frame, self._drag_select_add, clear=False)
            elif etype == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                if self._mouse_down:
                    self._mouse_down = False
                    self._drag_select_active = False
        elif isinstance(event, QtGui.QContextMenuEvent):
            frame = self._frame_at_global_pos(event.globalPos())
            if frame is not None:
                if frame not in self._selected_frames:
                    self._set_frame_selected(frame, True, clear=True)
                self._show_plot_context_menu(event.globalPos())
                return True
        return super().eventFilter(obj, event)

    def _show_plot_context_menu(self, global_pos: QtCore.QPoint):
        menu = QtWidgets.QMenu(self)
        act_plot = menu.addAction("Plot Data…")
        if not self.selected_frames():
            act_plot.setEnabled(False)
        act_plot.triggered.connect(self._on_plot_data_requested)
        menu.exec_(global_pos)

    def _on_plot_data_requested(self):
        frames = self.selected_frames()
        if not frames:
            return
        available: Set[str] = set()
        for fr in frames:
            available.update(fr.available_variables())
        available_list = sorted(available)
        default = ""
        for fr in frames:
            cur = fr.current_variable()
            if cur:
                default = cur
                break
        items = list(available_list)
        if default:
            if default not in items:
                items.insert(0, default)
                default_index = 0
            else:
                default_index = items.index(default)
        else:
            default_index = 0
        if not items:
            items = [default] if default else [""]
            default_index = 0
        item, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Plot Data",
            "Data variable:",
            items,
            default_index,
            editable=True,
        )
        if not ok:
            return
        name = str(item).strip()
        for fr in frames:
            fr.plot_variable(name)

    def _open_annotation_dialog(self):
        frames = self.selected_frames()
        if not frames:
            frames = list(self.frames)
        if not frames:
            QtWidgets.QMessageBox.information(
                self,
                "No plots",
                "Add a plot before setting annotations.",
            )
            return
        initial = frames[0].annotation_defaults()
        initial.apply_to_all = False
        dialog = PlotAnnotationDialog(self, initial=initial, allow_apply_all=True)
        dialog.applied.connect(lambda config: self._apply_annotation_dialog_result(config, frames))
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        config = dialog.annotation_config()
        if config is None:
            return
        self._apply_annotation_dialog_result(config, frames)

    def _apply_annotation_dialog_result(
        self, config: PlotAnnotationConfig, frames: List["ViewerFrame"]
    ) -> None:
        if config is None:
            return
        targets = self.frames if config.apply_to_all else frames
        base = replace(config, apply_to_all=False)
        for frame in targets:
            frame.apply_annotation(base)

    def _apply_line_style(self, style: LineStyleConfig, targets: List["ViewerFrame"]) -> None:
        if style is None:
            return
        for fr in targets:
            fr.set_line_style_config(style, refresh=True)
        log_action(f"Updated line style for {len(targets)} plot(s)")
        self._update_apply_button_state()

    def _apply_processing_selection(
        self, mode: str, params: Dict[str, object], frames: List["ViewerFrame"]
    ) -> None:
        for frame in frames:
            try:
                frame.apply_processing(mode, params, self.processing_manager)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Processing failed", str(exc))
                break

    def _open_line_style_dialog(self):
        frames = self.selected_frames()
        if frames:
            targets = [fr for fr in frames if fr.is_line_display()]
        else:
            targets = [fr for fr in self.frames if fr.is_line_display()]
        if not targets:
            QtWidgets.QMessageBox.information(
                self,
                "No 1D plots",
                "Select a 1D plot before adjusting the line style.",
            )
            return
        dialog = LineStyleDialog(self, initial=targets[0].line_style_config())
        dialog.applied.connect(lambda style: self._apply_line_style(style, targets))
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        style = dialog.line_style()
        if style is None:
            return
        self._apply_line_style(style, targets)

    def _open_colormap_dialog(self):
        frames = [fr for fr in self.selected_frames() if not fr.is_line_display()]
        if not frames:
            frames = [fr for fr in self.frames if not fr.is_line_display()]
        if not frames:
            QtWidgets.QMessageBox.information(
                self,
                "No images",
                "Select at least one 2D plot before setting a colormap.",
            )
            return
        names = available_colormap_names()
        if not names:
            QtWidgets.QMessageBox.warning(
                self,
                "No colormaps",
                "No colormaps are available to apply.",
            )
            return
        options = []
        for name in names:
            label = name.replace("_", " ").title()
            if is_scientific_colormap(name):
                label = f"{label} (Scientific)"
            options.append((label, name))
        labels = [label for label, _ in options]
        selection, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Select colormap",
            "Colormap:",
            labels,
            0,
            False,
        )
        if not ok:
            return
        try:
            index = labels.index(selection)
        except ValueError:
            return
        chosen = options[index][1]
        applied = 0
        for frame in frames:
            if frame.apply_colormap(chosen):
                applied += 1
        if applied:
            log_action(f"Applied colormap '{chosen}' to {applied} plot(s)")
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Colormap",
                "Unable to apply the selected colormap to the chosen plots.",
            )

    # ---------- export helpers ----------
    def _on_preferences_changed(self, _data):
        for frame in self.frames:
            frame.set_preferences(self.preferences)

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

    def _export_active_plot(self):
        frames = self.selected_frames()
        if len(frames) != 1:
            QtWidgets.QMessageBox.information(
                self,
                "Select a plot",
                "Please select a single plot to export.",
            )
            return
        frame = frames[0]
        suggestion = sanitize_filename(frame.lbl.text()) + ".png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save plot",
            suggestion,
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        suffix = ".jpg" if path.lower().endswith((".jpg", ".jpeg")) else ".png"
        target = ensure_extension(path, suffix)
        if not save_snapshot(frame, target):
            QtWidgets.QMessageBox.warning(self, "Save failed", "Unable to write the selected file.")
            return
        log_action(f"Saved MultiView plot to {target}")

    def _export_selected_plots(self):
        frames = self.selected_frames()
        if not frames:
            QtWidgets.QMessageBox.information(
                self,
                "No plots selected",
                "Select one or more plots to export individually.",
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
        count = 0
        for idx, frame in enumerate(frames, start=1):
            label = frame.lbl.text() or f"plot_{idx}"
            name = sanitize_filename(label) or f"plot_{idx}"
            target = base / f"{name}_{idx:02d}.png"
            if save_snapshot(frame, target):
                count += 1
        if count == 0:
            QtWidgets.QMessageBox.warning(self, "Export failed", "No plots were exported.")
            return
        QtWidgets.QMessageBox.information(
            self,
            "Export complete",
            f"Saved {count} plot(s) to {base}",
        )
        log_action(f"Exported {count} MultiView plots to {base}")

    def _export_layout_image(self):
        if not self.frames:
            QtWidgets.QMessageBox.information(
                self,
                "No plots",
                "Add at least one plot before exporting the layout.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save layout",
            "multiview-layout.png",
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
            QtWidgets.QMessageBox.warning(
                self,
                "Save failed",
                "Unable to save the layout image.",
            )
            return
        log_action(f"Saved MultiView layout to {target}")

    def _on_apply_processing_clicked(self):
        frames = self.selected_frames()
        if not frames:
            return
        dims = frames[0].processing_dims() if frames else ()
        dialog = ProcessingSelectionDialog(
            self.processing_manager,
            self,
            dims=dims if dims else None,
        )
        dialog.applied.connect(lambda mode, params: self._apply_processing_selection(mode, params, frames))
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        mode, params = dialog.selected_processing()
        self._apply_processing_selection(mode, params, frames)


    def _reflow(self):
        """Rebuild the splitter grid according to the current column count."""
        # Detach any existing children from row splitters, then clear rows
        for i in range(self.vsplit.count()):
            w = self.vsplit.widget(i)
            if isinstance(w, QtWidgets.QSplitter):
                while w.count():
                    cw = w.widget(0)
                    if cw:
                        cw.setParent(None)
        while self.vsplit.count():
            w = self.vsplit.widget(0)
            w.setParent(None)
            w.deleteLater()

        cols = max(1, self.col_spin.value())
        for r_start in range(0, len(self.frames), cols):
            row_frames = self.frames[r_start:r_start + cols]
            h = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            h.setChildrenCollapsible(False)
            for fr in row_frames:
                h.addWidget(fr)
            self.vsplit.addWidget(h)

        # Re-apply current histogram visibility to all tiles
        self._apply_histogram_visibility(self.chk_show_hist.isChecked())
        self.equalize_columns()

    def _apply_histogram_visibility(self, on: bool):
        """Show/hide the classic HistogramLUTItem on every tile."""
        on = bool(on)
        for fr in self.frames:
            try:
                fr.set_histogram_visible(on)
            except Exception:
                pass

    def _connect_frame_signals(self, fr: ViewerFrame):
        viewer = getattr(fr, "viewer", None)
        if viewer is None:
            return
        if viewer not in self._level_handlers:
            try:
                handler = partial(self._viewer_levels_changed, viewer)
                viewer.sigLevelsChanged.connect(handler)
                self._level_handlers[viewer] = handler
            except Exception:
                self._level_handlers.pop(viewer, None)
        if viewer not in self._view_handlers:
            try:
                handler = partial(self._viewer_view_changed, viewer)
                viewer.sigViewChanged.connect(handler)
                self._view_handlers[viewer] = handler
            except Exception:
                self._view_handlers.pop(viewer, None)
        if viewer not in self._cursor_handlers:
            try:
                viewer.sigCursorMoved.connect(self._viewer_cursor_moved)
                self._cursor_handlers[viewer] = True
            except Exception:
                self._cursor_handlers.pop(viewer, None)

    def _on_frame_mode_changed(self, *_):
        self._update_apply_button_state()

    def _disconnect_frame_signals(self, fr: ViewerFrame):
        viewer = getattr(fr, "viewer", None)
        if viewer is None:
            return
        handler = self._level_handlers.pop(viewer, None)
        if handler is not None:
            try:
                viewer.sigLevelsChanged.disconnect(handler)
            except Exception:
                pass
        handler = self._view_handlers.pop(viewer, None)
        if handler is not None:
            try:
                viewer.sigViewChanged.disconnect(handler)
            except Exception:
                pass
        if viewer in self._cursor_handlers:
            try:
                viewer.sigCursorMoved.disconnect(self._viewer_cursor_moved)
            except Exception:
                pass
            self._cursor_handlers.pop(viewer, None)
        try:
            viewer.clear_mirrored_crosshair()
        except Exception:
            pass

    def _viewer_levels_changed(self, viewer, levels):
        if not self.chk_link_levels.isChecked() or self._syncing_levels:
            return
        self._syncing_levels = True
        try:
            try:
                lo, hi = (float(levels[0]), float(levels[1]))
            except Exception:
                lo, hi = viewer.get_levels()
            for fr in self.frames:
                if fr.viewer is viewer:
                    continue
                fr.viewer.set_levels(lo, hi)
        finally:
            self._syncing_levels = False

    def _viewer_view_changed(self, viewer, xr, yr):
        if not self.chk_link_panzoom.isChecked() or self._syncing_views:
            return
        self._syncing_views = True
        try:
            xr_vals = tuple(xr) if xr is not None else viewer.get_view_range()[0]
            yr_vals = tuple(yr) if yr is not None else viewer.get_view_range()[1]
            for fr in self.frames:
                if fr.viewer is viewer:
                    continue
                fr.viewer.set_view_range(xr=xr_vals, yr=yr_vals)
        finally:
            self._syncing_views = False

    def _on_link_levels_toggled(self, on: bool):
        if on:
            self._sync_all_levels()

    def _on_link_panzoom_toggled(self, on: bool):
        if on:
            self._sync_all_views()

    def _on_link_cursor_toggled(self, on: bool):
        if not on:
            self._clear_mirrored_crosshairs()

    def _sync_all_levels(self):
        if not self.frames:
            return
        ref = self.frames[0].viewer
        try:
            lo, hi = ref.get_levels()
        except Exception:
            return
        self._syncing_levels = True
        try:
            for fr in self.frames[1:]:
                fr.viewer.set_levels(lo, hi)
        finally:
            self._syncing_levels = False

    def _sync_all_views(self):
        if not self.frames:
            return
        ref = self.frames[0].viewer
        try:
            xr, yr = ref.get_view_range()
        except Exception:
            return
        self._syncing_views = True
        try:
            for fr in self.frames[1:]:
                fr.viewer.set_view_range(xr=xr, yr=yr)
        finally:
            self._syncing_views = False

    def _viewer_cursor_moved(self, viewer, x, y, value, inside, label):
        if not inside or not self.chk_cursor_mirror.isChecked():
            self._clear_mirrored_crosshairs(exclude=viewer)
            return
        for fr in self.frames:
            other = fr.viewer
            if other is viewer:
                continue
            try:
                local_value = other.value_at(x, y) if hasattr(other, "value_at") else None
            except Exception:
                local_value = None
            try:
                other.show_crosshair(x, y, value=local_value, mirrored=True)
            except Exception:
                pass

    def _clear_mirrored_crosshairs(self, exclude=None):
        for fr in self.frames:
            viewer = fr.viewer
            if exclude is not None and viewer is exclude:
                continue
            try:
                viewer.clear_mirrored_crosshair()
            except Exception:
                pass

    def _sync_new_frame_to_links(self, fr: ViewerFrame):
        others = [f for f in self.frames if f is not fr]
        if not others:
            return
        ref = others[0].viewer
        if self.chk_link_levels.isChecked():
            try:
                lo, hi = ref.get_levels()
                fr.viewer.set_levels(lo, hi)
            except Exception:
                pass
        if self.chk_link_panzoom.isChecked():
            try:
                xr, yr = ref.get_view_range()
                fr.viewer.set_view_range(xr=xr, yr=yr)
            except Exception:
                pass

    def _autoscale_colors(self):
        if self.chk_link_levels.isChecked():
            self.chk_link_levels.setChecked(False)
        for fr in self.frames:
            try:
                fr.viewer.autoscale_levels()
            except Exception:
                pass

    def _auto_panzoom(self):
        for fr in self.frames:
            try:
                fr.viewer.auto_view_range()
            except Exception:
                pass

    def equalize_columns(self):
        for splitter in self._iter_row_splitters():
            count = splitter.count()
            if count <= 0:
                continue
            splitter.setSizes([1] * count)

    def equalize_rows(self):
        count = self.vsplit.count()
        if count <= 0:
            return
        self.vsplit.setSizes([1] * count)

    def _iter_row_splitters(self):
        for i in range(self.vsplit.count()):
            w = self.vsplit.widget(i)
            if isinstance(w, QtWidgets.QSplitter):
                yield w
