#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from functools import partial

import numpy as np
import xarray as xr

from PySide2 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
try:
    import pyqtgraph.opengl as gl
except Exception:  # pragma: no cover - optional dependency
    gl = None

from xr_plot_widget import CentralPlotWidget, ScientificAxisItem
from xr_coords import guess_phys_coords
from data_processing import (
    ParameterDefinition,
    ProcessingPipeline,
    ProcessingStep,
    apply_processing_step,
    get_processing_function,
    list_processing_functions,
    summarize_parameters,
)

# ---------------------------------------------------------------------------
# Helper: open_dataset
# ---------------------------------------------------------------------------
_FORCE_ENGINE = None  # override to force a specific xarray engine if desired

def open_dataset(path: Path) -> xr.Dataset:
    if path.suffix.lower() == '.zarr' or path.name.lower().endswith('.zarr'):
        return xr.open_zarr(str(path))
    if _FORCE_ENGINE:
        return xr.open_dataset(str(path), engine=_FORCE_ENGINE)
    return xr.open_dataset(str(path))


def _nan_aware_reducer(func):
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
                if ax < 0:
                    normalized.append((ax + ndim) % ndim)
                else:
                    normalized.append(ax)
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


# ---------------------------------------------------------------------------
# Dataset / variable references used for drag & drop
# ---------------------------------------------------------------------------
class DataSetRef(QtCore.QObject):
    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)

    def to_mime(self) -> str:
        return "DatasetRef:" + json.dumps({"path": str(self.path)})

    @staticmethod
    def from_mime(txt: str) -> Optional["DataSetRef"]:
        if not txt or not txt.startswith("DatasetRef:"):
            return None
        try:
            data = json.loads(txt.split(":", 1)[1])
            return DataSetRef(Path(data["path"]))
        except Exception:
            return None

    def load(self) -> xr.Dataset:
        return open_dataset(self.path)


class VarRef(QtCore.QObject):
    def __init__(self, path: Path, var: str, hint: str = ""):
        super().__init__()
        self.path = Path(path)
        self.var = var
        self.hint = hint

    def to_mime(self) -> str:
        return "VarRef:" + json.dumps({"path": str(self.path), "var": self.var, "hint": self.hint})

    @staticmethod
    def from_mime(txt: str) -> Optional["VarRef"]:
        if not txt or not txt.startswith("VarRef:"):
            return None
        try:
            data = json.loads(txt.split(":", 1)[1])
            return VarRef(Path(data["path"]), data["var"], data.get("hint", ""))
        except Exception:
            return None

    def load(self):
        ds = open_dataset(self.path)
        if self.var not in ds.data_vars or ds[self.var].ndim != 2:
            raise RuntimeError(f"{self.var!r} is not a 2D variable in {self.path}")
        da = ds[self.var]
        coords = guess_phys_coords(da)
        return da, coords


# ---------------------------------------------------------------------------
# Parameter editing helpers
# ---------------------------------------------------------------------------


class ParameterForm(QtWidgets.QWidget):
    parametersChanged = QtCore.Signal()

    def __init__(self, parameters: Iterable[ParameterDefinition], parent=None):
        super().__init__(parent)
        self._definitions = list(parameters)
        self._widgets: Dict[str, QtWidgets.QWidget] = {}

        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        if not self._definitions:
            lbl = QtWidgets.QLabel("No parameters")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            layout.addRow(lbl)
            return

        for definition in self._definitions:
            widget: Optional[QtWidgets.QWidget] = None
            if definition.kind == "float":
                spin = QtWidgets.QDoubleSpinBox()
                spin.setDecimals(6)
                lo = float(definition.minimum) if definition.minimum is not None else -1e9
                hi = float(definition.maximum) if definition.maximum is not None else 1e9
                spin.setRange(lo, hi)
                if definition.step is not None:
                    spin.setSingleStep(float(definition.step))
                spin.setValue(float(definition.default))
                spin.valueChanged.connect(lambda *_: self.parametersChanged.emit())
                widget = spin
            elif definition.kind == "int":
                spin_i = QtWidgets.QSpinBox()
                lo = int(definition.minimum) if definition.minimum is not None else -1_000_000
                hi = int(definition.maximum) if definition.maximum is not None else 1_000_000
                spin_i.setRange(lo, hi)
                if definition.step is not None:
                    spin_i.setSingleStep(int(definition.step))
                spin_i.setValue(int(definition.default))
                spin_i.valueChanged.connect(lambda *_: self.parametersChanged.emit())
                widget = spin_i
            elif definition.kind == "enum":
                combo = QtWidgets.QComboBox()
                if definition.choices:
                    for label, value in definition.choices:
                        combo.addItem(label, value)
                combo.setCurrentIndex(max(combo.findData(definition.default), 0))
                combo.currentIndexChanged.connect(lambda *_: self.parametersChanged.emit())
                widget = combo
            else:
                line = QtWidgets.QLineEdit(str(definition.default))
                line.textChanged.connect(lambda *_: self.parametersChanged.emit())
                widget = line

            self._widgets[definition.name] = widget
            layout.addRow(definition.label, widget)

    def values(self) -> Dict[str, object]:
        values: Dict[str, object] = {}
        for definition in self._definitions:
            widget = self._widgets.get(definition.name)
            if widget is None:
                continue
            if isinstance(widget, QtWidgets.QDoubleSpinBox):
                values[definition.name] = float(widget.value())
            elif isinstance(widget, QtWidgets.QSpinBox):
                values[definition.name] = int(widget.value())
            elif isinstance(widget, QtWidgets.QComboBox):
                data = widget.currentData()
                values[definition.name] = data if data is not None else widget.currentText()
            elif isinstance(widget, QtWidgets.QLineEdit):
                values[definition.name] = widget.text()
        return values

    def set_values(self, params: Dict[str, object]):
        for definition in self._definitions:
            widget = self._widgets.get(definition.name)
            if widget is None:
                continue
            value = params.get(definition.name, definition.default)
            block = widget.blockSignals(True)
            try:
                if isinstance(widget, QtWidgets.QDoubleSpinBox):
                    widget.setValue(float(value))
                elif isinstance(widget, QtWidgets.QSpinBox):
                    widget.setValue(int(value))
                elif isinstance(widget, QtWidgets.QComboBox):
                    idx = widget.findData(value)
                    if idx < 0:
                        idx = widget.findText(str(value))
                    widget.setCurrentIndex(max(idx, 0))
                elif isinstance(widget, QtWidgets.QLineEdit):
                    widget.setText(str(value))
            finally:
                widget.blockSignals(block)


# ---------------------------------------------------------------------------
# Processing manager and dialogs
# ---------------------------------------------------------------------------


class ProcessingManager(QtCore.QObject):
    pipelines_changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self._pipelines: Dict[str, ProcessingPipeline] = {}

    def list_pipelines(self) -> List[ProcessingPipeline]:
        return [self._clone_pipeline(p) for p in self._pipelines.values()]

    def pipeline_names(self) -> List[str]:
        return sorted(self._pipelines.keys())

    def get_pipeline(self, name: str) -> Optional[ProcessingPipeline]:
        if name not in self._pipelines:
            return None
        return self._clone_pipeline(self._pipelines[name])

    def save_pipeline(self, pipeline: ProcessingPipeline):
        name = pipeline.name.strip()
        if not name:
            raise ValueError("Pipeline name cannot be empty")
        self._pipelines[name] = self._clone_pipeline(pipeline)
        self.pipelines_changed.emit()

    def delete_pipeline(self, name: str):
        if name in self._pipelines:
            del self._pipelines[name]
            self.pipelines_changed.emit()

    def _clone_pipeline(self, pipeline: ProcessingPipeline) -> ProcessingPipeline:
        return ProcessingPipeline(
            name=pipeline.name,
            steps=[ProcessingStep(step.key, dict(step.params)) for step in pipeline.steps],
        )


class PipelineEditorDialog(QtWidgets.QDialog):
    def __init__(self, manager: ProcessingManager, pipeline: ProcessingPipeline, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.pipeline = ProcessingPipeline(
            name=pipeline.name,
            steps=[ProcessingStep(step.key, dict(step.params)) for step in pipeline.steps],
        )
        self._raw_data: Optional[np.ndarray] = None
        self._processed_data: Optional[np.ndarray] = None
        self.setWindowTitle(f"Edit Pipeline – {self.pipeline.name or 'Untitled'}")
        self.resize(800, 700)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        top_row = QtWidgets.QHBoxLayout()
        self.lbl_data = QtWidgets.QLabel("No data loaded")
        self.lbl_data.setStyleSheet("color: #555;")
        top_row.addWidget(self.lbl_data, 1)
        self.btn_load_data = QtWidgets.QPushButton("Load data…")
        self.btn_load_data.clicked.connect(self._load_data)
        top_row.addWidget(self.btn_load_data, 0)
        cmap_label = QtWidgets.QLabel("Color map:")
        cmap_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        top_row.addWidget(cmap_label, 0)
        self.cmb_colormap = QtWidgets.QComboBox()
        self.cmb_colormap.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        candidate_maps = [
            "gray",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "thermal",
        ]
        for name in candidate_maps:
            try:
                pg.colormap.get(name)
            except Exception:
                continue
            label = name.title()
            self.cmb_colormap.addItem(label, name)
        if self.cmb_colormap.count() == 0:
            self.cmb_colormap.addItem("Default", "default")
        self.cmb_colormap.currentIndexChanged.connect(self._on_colormap_changed)
        top_row.addWidget(self.cmb_colormap, 0)
        layout.addLayout(top_row)

        self.image_view = pg.ImageView()
        try:
            self.image_view.getView().invertY(True)
        except Exception:
            pass
        self.roi = pg.RectROI([10, 10], [40, 40], pen=pg.mkPen('#ffaa00', width=2))
        self.roi.addScaleHandle((1, 1), (0, 0))
        self.roi.addScaleHandle((0, 0), (1, 1))
        try:
            self.roi.setVisible(False)
        except Exception:
            pass

        roi_button = getattr(self.image_view.ui, "roiBtn", None)
        if roi_button is not None:
            try:
                roi_button.clicked.disconnect()
            except Exception:
                pass
            roi_button.setCheckable(True)
            roi_button.setChecked(False)
            roi_button.toggled.connect(self._on_roi_button_toggled)
            roi_button.setToolTip("Toggle ROI preview (right-click the ROI plot for reduction options)")
        roi_plot_widget = getattr(self.image_view.ui, "roiPlot", None)
        if roi_plot_widget is not None:
            roi_plot_widget.hide()
            roi_plot_widget.setMaximumHeight(0)
        layout.addWidget(self.image_view, 2)

        self.roi_box = QtWidgets.QGroupBox("ROI preview")
        roi_layout = QtWidgets.QVBoxLayout(self.roi_box)
        roi_layout.setContentsMargins(8, 8, 8, 8)
        roi_layout.setSpacing(6)

        self._roi_axis_options: List[tuple[str, int, str, str]] = [
            ("Collapse rows (Y) → profile across X", 0, "rows (Y)", "X"),
            ("Collapse columns (X) → profile across Y", 1, "columns (X)", "Y"),
        ]
        self._roi_axis_index: int = 0
        self._roi_reducers = {
            "mean": ("Mean", _nan_aware_reducer(lambda arr, axis=None: np.nanmean(arr, axis=axis))),
            "median": ("Median", _nan_aware_reducer(lambda arr, axis=None: np.nanmedian(arr, axis=axis))),
            "min": ("Minimum", _nan_aware_reducer(lambda arr, axis=None: np.nanmin(arr, axis=axis))),
            "max": ("Maximum", _nan_aware_reducer(lambda arr, axis=None: np.nanmax(arr, axis=axis))),
            "std": ("Std. dev", _nan_aware_reducer(lambda arr, axis=None: np.nanstd(arr, axis=axis))),
            "ptp": (
                "Peak-to-peak",
                _nan_aware_reducer(
                    lambda arr, axis=None: np.nanmax(arr, axis=axis) - np.nanmin(arr, axis=axis)
                ),
            ),
        }
        self._roi_method_key: str = "mean"

        self.lbl_roi_axis = QtWidgets.QLabel()
        self.lbl_roi_axis.setStyleSheet("color: #555;")
        roi_layout.addWidget(self.lbl_roi_axis)

        hint = QtWidgets.QLabel("Right-click the ROI plot to change the reduction axis or statistic.")
        hint.setStyleSheet("color: #777;")
        roi_layout.addWidget(hint)

        self.roi_plot = pg.PlotWidget()
        self.roi_plot.showGrid(x=True, y=True, alpha=0.3)
        self.roi_plot.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.roi_plot.customContextMenuRequested.connect(self._show_roi_context_menu)
        self.roi_curve = self.roi_plot.plot([], [], pen=pg.mkPen('#ffaa00', width=2))
        roi_layout.addWidget(self.roi_plot, 1)

        self.roi_box.hide()
        layout.addWidget(self.roi_box, 1)

        self.steps_scroll = QtWidgets.QScrollArea()
        self.steps_scroll.setWidgetResizable(True)
        self.steps_container = QtWidgets.QWidget()
        self.steps_layout = QtWidgets.QVBoxLayout(self.steps_container)
        self.steps_layout.setContentsMargins(0, 0, 0, 0)
        self.steps_layout.setSpacing(8)
        self.steps_scroll.setWidget(self.steps_container)
        layout.addWidget(self.steps_scroll, 1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._forms: List[tuple[ProcessingStep, ParameterForm]] = []
        self._roi_enabled = False
        self._rebuild_forms()
        self._update_roi_axis_label()
        try:
            self.roi.sigRegionChanged.connect(self._update_roi_preview)
        except Exception:
            pass

    # ----- helpers -----
    def result_pipeline(self) -> ProcessingPipeline:
        return ProcessingPipeline(
            name=self.pipeline.name,
            steps=[ProcessingStep(step.key, dict(step.params)) for step in self.pipeline.steps],
        )

    def _rebuild_forms(self):
        while self.steps_layout.count():
            item = self.steps_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._forms.clear()

        if not self.pipeline.steps:
            lbl = QtWidgets.QLabel("Pipeline has no steps.")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet("color: #777;")
            self.steps_layout.addWidget(lbl)
            return

        for idx, step in enumerate(self.pipeline.steps, start=1):
            spec = get_processing_function(step.key)
            title = spec.label if spec else step.key
            box = QtWidgets.QGroupBox(f"Step {idx}: {title}")
            vbox = QtWidgets.QVBoxLayout(box)
            if spec is None:
                lbl = QtWidgets.QLabel("Unknown processing function")
                lbl.setAlignment(QtCore.Qt.AlignLeft)
                vbox.addWidget(lbl)
            else:
                form = ParameterForm(spec.parameters)
                form.set_values(step.params)
                form.parametersChanged.connect(self._on_parameters_changed)
                vbox.addWidget(form)
                self._forms.append((step, form))
            self.steps_layout.addWidget(box)
        self.steps_layout.addStretch(1)
        self._apply_pipeline()

    def _update_steps_from_forms(self):
        for step, form in self._forms:
            step.params = form.values()

    def _apply_pipeline(self):
        if self._raw_data is None:
            return
        data = np.asarray(self._raw_data, float)
        try:
            for step in self.pipeline.steps:
                data = apply_processing_step(step.key, data, step.params)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Processing failed", str(e))
            return
        self._processed_data = data
        try:
            self.image_view.setImage(data, autoLevels=True)
            self._apply_selected_colormap()
        except Exception:
            pass
        self._update_roi_preview()

    def _apply_selected_colormap(self):
        if not hasattr(self, "cmb_colormap"):
            return
        name = self.cmb_colormap.currentData()
        if not name or name == "default":
            return
        try:
            cmap = pg.colormap.get(str(name))
        except Exception:
            return
        try:
            self.image_view.setColorMap(cmap)
        except Exception:
            pass

    def _update_roi_axis_label(self):
        if not self._roi_axis_options:
            self.roi_box.setTitle("ROI preview")
            self.lbl_roi_axis.setText("")
            return
        index = max(0, min(self._roi_axis_index, len(self._roi_axis_options) - 1))
        axis, collapsed, remaining = self._roi_axis_options[index][1:]
        self.roi_box.setTitle(f"ROI preview – reducing {collapsed}")
        self.lbl_roi_axis.setText(f"Reducing over {collapsed} to plot along {remaining}.")
        self.roi_plot.setLabel("bottom", f"{remaining} index")
        self.roi_plot.setTitle(f"ROI profile along {remaining}")

    def _current_roi_axis(self) -> int:
        if not self._roi_axis_options:
            return 0
        index = max(0, min(self._roi_axis_index, len(self._roi_axis_options) - 1))
        axis = self._roi_axis_options[index][1]
        return int(axis)

    def _on_colormap_changed(self):
        self._apply_selected_colormap()

    def _show_roi_context_menu(self, pos: QtCore.QPoint):
        if not self._roi_axis_options:
            return
        menu = QtWidgets.QMenu(self.roi_plot)

        axis_menu = menu.addMenu("Reduce over")
        for idx, (label, *_rest) in enumerate(self._roi_axis_options):
            action = axis_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(idx == self._roi_axis_index)
            action.triggered.connect(partial(self._set_roi_axis_index, idx))

        stat_menu = menu.addMenu("Statistic")
        for key, (label, _) in self._roi_reducers.items():
            action = stat_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(key == self._roi_method_key)
            action.triggered.connect(partial(self._set_roi_method, key))

        menu.exec_(self.roi_plot.mapToGlobal(pos))

    def _set_roi_axis_index(self, index: int):
        if index == self._roi_axis_index:
            return
        self._roi_axis_index = max(0, min(index, len(self._roi_axis_options) - 1))
        self._update_roi_axis_label()
        self._update_roi_preview()

    def _set_roi_method(self, key: str):
        if key not in self._roi_reducers or key == self._roi_method_key:
            return
        self._roi_method_key = key
        self._update_roi_preview()

    def _on_roi_button_toggled(self, checked: bool):
        view = getattr(self.image_view, "view", None)
        if view is None:
            try:
                view = self.image_view.getView()
            except Exception:
                view = None
        if view is not None:
            try:
                if checked:
                    if self.roi.scene() is None:
                        view.addItem(self.roi)
                else:
                    view.removeItem(self.roi)
            except Exception:
                pass
        try:
            self.roi.setVisible(checked)
        except Exception:
            pass
        self._roi_enabled = bool(checked)
        self.roi_box.setVisible(self._roi_enabled)
        if self._roi_enabled:
            self._reset_roi_to_image()
            self._update_roi_preview()
        else:
            self.roi_curve.setData([], [])

    def _extract_roi_array(self) -> Optional[np.ndarray]:
        if self._processed_data is None:
            return None
        if not hasattr(self, "roi") or self.roi is None:
            return None
        image_item = getattr(self.image_view, "imageItem", None)
        if image_item is None:
            return None
        try:
            roi_data = self.roi.getArrayRegion(self._processed_data, image_item)
        except Exception:
            try:
                roi_data = self.roi.getArraySlice(self._processed_data, image_item)
                if isinstance(roi_data, tuple):
                    roi_data = roi_data[0]
            except Exception:
                return None
        if roi_data is None:
            return None
        return np.asarray(roi_data)

    def _update_roi_preview(self):
        if not self._roi_enabled or self._processed_data is None:
            self.roi_curve.setData([], [])
            return
        roi_array = self._extract_roi_array()
        if roi_array is None or roi_array.size == 0:
            self.roi_curve.setData([], [])
            return
        method_key = self._roi_method_key
        reducer_entry = self._roi_reducers.get(method_key)
        if reducer_entry is None:
            self.roi_curve.setData([], [])
            return
        _, reducer = reducer_entry
        axis = self._current_roi_axis()
        axis = max(0, min(axis, roi_array.ndim - 1))
        with np.errstate(all="ignore"):
            profile = reducer(roi_array, axis=axis)
        if profile is None:
            self.roi_curve.setData([], [])
            return
        profile = np.asarray(profile).ravel()
        if profile.size == 0:
            self.roi_curve.setData([], [])
            return
        x = np.arange(profile.size)
        self.roi_curve.setData(x, profile)
        self.roi_plot.enableAutoRange()

    def _reset_roi_to_image(self, shape: Optional[Tuple[int, int]] = None):
        if not self._roi_enabled or not hasattr(self, "roi") or self.roi is None:
            return
        if shape is None:
            if self._processed_data is None:
                return
            shape = self._processed_data.shape
        if not shape or len(shape) < 2:
            return
        height, width = int(shape[0]), int(shape[1])
        if width <= 0 or height <= 0:
            return
        rect_width = max(2, width // 2)
        rect_height = max(2, height // 2)
        pos_x = max(0, (width - rect_width) // 2)
        pos_y = max(0, (height - rect_height) // 2)
        try:
            self.roi.blockSignals(True)
            self.roi.setPos((pos_x, pos_y))
            self.roi.setSize((rect_width, rect_height))
        finally:
            try:
                self.roi.blockSignals(False)
            except Exception:
                pass
        self._update_roi_preview()

    # ----- slots -----
    def _on_parameters_changed(self):
        self._update_steps_from_forms()
        self._apply_pipeline()

    def _load_data(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select dataset",
            "",
            "NetCDF / Zarr (*.nc *.zarr);;All files (*)",
        )
        if not path:
            return
        p = Path(path)
        try:
            ds = open_dataset(p)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(e))
            return
        choices = [var for var in ds.data_vars if ds[var].ndim == 2]
        if not choices:
            QtWidgets.QMessageBox.information(self, "No 2D variables", "The dataset has no 2D variables to preview.")
            return
        var, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Select variable",
            "Variable:",
            choices,
            0,
            False,
        )
        if not ok or not var:
            return
        da = ds[var]
        self._raw_data = np.asarray(da.values, float)
        self._processed_data = np.asarray(self._raw_data)
        self.lbl_data.setText(f"{p.name}:{var}")
        self._reset_roi_to_image(self._raw_data.shape if hasattr(self._raw_data, "shape") else None)
        self._apply_pipeline()

# ---------------------------------------------------------------------------
# Processing dock widget
# ---------------------------------------------------------------------------


class ProcessingDockContainer(QtWidgets.QWidget):
    def __init__(self, title: str, widget: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        self._title = title
        self._content_widget = widget
        self._floating_window: Optional[QtWidgets.QDialog] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(6, 6, 6, 6)
        header_layout.setSpacing(6)
        lbl = QtWidgets.QLabel(title)
        font = lbl.font()
        font.setBold(True)
        lbl.setFont(font)
        header_layout.addWidget(lbl)
        header_layout.addStretch(1)

        self.btn_float = QtWidgets.QToolButton()
        self.btn_float.setText("Float")
        self.btn_float.setAutoRaise(True)
        self.btn_float.setToolTip("Undock processing pane to a floating window")
        self.btn_float.clicked.connect(self._on_float_clicked)
        header_layout.addWidget(self.btn_float)

        self.btn_toggle = QtWidgets.QToolButton()
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(True)
        self.btn_toggle.setAutoRaise(True)
        self.btn_toggle.setArrowType(QtCore.Qt.DownArrow)
        self.btn_toggle.setToolTip("Hide processing pane")
        self.btn_toggle.toggled.connect(self._on_toggle_toggled)
        header_layout.addWidget(self.btn_toggle)

        layout.addWidget(header)

        self._content_frame = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._content_frame)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        self._content_layout.addWidget(self._content_widget)
        layout.addWidget(self._content_frame, 1)

        self._placeholder = QtWidgets.QLabel(
            "Processing pane is undocked. Click Dock to return it to the sidebar."
        )
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.hide()
        layout.addWidget(self._placeholder, 1)

        self._update_toggle_visuals()
        self._update_content_visibility()
        self._update_float_button()

    def _on_toggle_toggled(self, checked: bool):
        del checked
        self._update_toggle_visuals()
        self._update_content_visibility()

    def _on_float_clicked(self):
        if self._floating_window is not None:
            self.dock()
        else:
            self.undock()

    def undock(self):
        if self._floating_window is not None:
            try:
                self._floating_window.raise_()
                self._floating_window.activateWindow()
            except Exception:
                pass
            return
        self.btn_toggle.setChecked(True)
        self._update_toggle_visuals()
        self._update_content_visibility()
        self._content_layout.removeWidget(self._content_widget)
        self._content_widget.setParent(None)
        dialog = QtWidgets.QDialog(self.window())
        dialog.setWindowTitle(self._title)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        dlg_layout = QtWidgets.QVBoxLayout(dialog)
        dlg_layout.setContentsMargins(0, 0, 0, 0)
        dlg_layout.setSpacing(0)
        dlg_layout.addWidget(self._content_widget)
        dialog.finished.connect(self._on_floating_closed)
        dialog.resize(self.width(), self.height())
        dialog.show()
        self._floating_window = dialog
        self._update_float_button()
        self._update_content_visibility()

    def dock(self):
        floating = self._floating_window
        if floating is None:
            return
        self._floating_window = None
        try:
            floating.finished.disconnect(self._on_floating_closed)
        except Exception:
            pass
        layout = floating.layout()
        if layout is not None:
            layout.removeWidget(self._content_widget)
        self._content_widget.setParent(self._content_frame)
        self._content_layout.addWidget(self._content_widget)
        self._content_widget.show()
        floating.hide()
        floating.deleteLater()
        self._update_float_button()
        self._update_content_visibility()

    def _on_floating_closed(self, *_):
        if self._floating_window is None:
            return
        self.dock()

    def _update_toggle_visuals(self):
        if self.btn_toggle.isChecked():
            self.btn_toggle.setArrowType(QtCore.Qt.DownArrow)
            self.btn_toggle.setToolTip("Hide processing pane")
        else:
            self.btn_toggle.setArrowType(QtCore.Qt.RightArrow)
            self.btn_toggle.setToolTip("Show processing pane")

    def _update_float_button(self):
        if self._floating_window is not None:
            self.btn_float.setText("Dock")
            self.btn_float.setToolTip("Dock processing pane back to the sidebar")
        else:
            self.btn_float.setText("Float")
            self.btn_float.setToolTip("Undock processing pane to a floating window")

    def _update_content_visibility(self):
        if self._floating_window is not None:
            self._content_frame.hide()
            self._placeholder.show()
            self.btn_toggle.setEnabled(False)
        else:
            self._placeholder.hide()
            self.btn_toggle.setEnabled(True)
            self._content_frame.setVisible(self.btn_toggle.isChecked())


class ProcessingDockWidget(QtWidgets.QWidget):
    def __init__(self, manager: ProcessingManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.steps: List[ProcessingStep] = []

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        # --- Step builder ---
        builder = QtWidgets.QGroupBox("Build step")
        builder_layout = QtWidgets.QVBoxLayout(builder)
        builder_layout.setContentsMargins(6, 6, 6, 6)
        builder_layout.setSpacing(6)

        func_row = QtWidgets.QHBoxLayout()
        func_row.addWidget(QtWidgets.QLabel("Function:"))
        self.cmb_function = QtWidgets.QComboBox()
        self.cmb_function.addItem("Select…", "")
        self._stack_indices: Dict[str, int] = {}
        self.param_stack = QtWidgets.QStackedWidget()
        self.param_stack.addWidget(QtWidgets.QWidget())
        for spec in list_processing_functions():
            self.cmb_function.addItem(spec.label, spec.key)
            form = ParameterForm(spec.parameters)
            form.parametersChanged.connect(self._on_function_params_changed)
            idx = self.param_stack.addWidget(form)
            self._stack_indices[spec.key] = idx
        self.cmb_function.currentIndexChanged.connect(self._on_function_changed)
        func_row.addWidget(self.cmb_function, 1)
        builder_layout.addLayout(func_row)
        builder_layout.addWidget(self.param_stack)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_step = QtWidgets.QPushButton("Add step")
        self.btn_add_step.clicked.connect(self._add_step)
        btn_row.addWidget(self.btn_add_step)
        self.btn_update_step = QtWidgets.QPushButton("Update selected")
        self.btn_update_step.clicked.connect(self._update_step)
        btn_row.addWidget(self.btn_update_step)
        builder_layout.addLayout(btn_row)

        outer.addWidget(builder)

        # --- Steps list ---
        self.list_steps = QtWidgets.QListWidget()
        self.list_steps.currentRowChanged.connect(self._on_step_selected)
        outer.addWidget(self.list_steps, 1)

        step_btns = QtWidgets.QHBoxLayout()
        self.btn_move_up = QtWidgets.QPushButton("Move up")
        self.btn_move_up.clicked.connect(lambda: self._move_step(-1))
        step_btns.addWidget(self.btn_move_up)
        self.btn_move_down = QtWidgets.QPushButton("Move down")
        self.btn_move_down.clicked.connect(lambda: self._move_step(1))
        step_btns.addWidget(self.btn_move_down)
        self.btn_remove_step = QtWidgets.QPushButton("Remove")
        self.btn_remove_step.clicked.connect(self._remove_step)
        step_btns.addWidget(self.btn_remove_step)
        self.btn_clear_steps = QtWidgets.QPushButton("Clear")
        self.btn_clear_steps.clicked.connect(self._clear_steps)
        step_btns.addWidget(self.btn_clear_steps)
        outer.addLayout(step_btns)

        # --- Pipeline info ---
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Pipeline name:"))
        self.edit_name = QtWidgets.QLineEdit()
        self.edit_name.textChanged.connect(lambda _: self._update_buttons())
        name_row.addWidget(self.edit_name, 1)
        outer.addLayout(name_row)

        save_row = QtWidgets.QHBoxLayout()
        self.btn_save_pipeline = QtWidgets.QPushButton("Save pipeline")
        self.btn_save_pipeline.clicked.connect(self._save_pipeline)
        save_row.addWidget(self.btn_save_pipeline)
        self.btn_interactive = QtWidgets.QPushButton("Interactive edit…")
        self.btn_interactive.clicked.connect(self._interactive_edit)
        save_row.addWidget(self.btn_interactive)
        outer.addLayout(save_row)

        # --- Saved pipelines ---
        saved_box = QtWidgets.QGroupBox("Saved pipelines")
        saved_layout = QtWidgets.QVBoxLayout(saved_box)
        saved_layout.setContentsMargins(6, 6, 6, 6)
        saved_layout.setSpacing(6)
        self.list_saved = QtWidgets.QListWidget()
        self.list_saved.itemSelectionChanged.connect(self._update_buttons)
        self.list_saved.itemDoubleClicked.connect(lambda _: self._load_saved())
        saved_layout.addWidget(self.list_saved)

        saved_btns = QtWidgets.QHBoxLayout()
        self.btn_load_saved = QtWidgets.QPushButton("Load")
        self.btn_load_saved.clicked.connect(self._load_saved)
        saved_btns.addWidget(self.btn_load_saved)
        self.btn_delete_saved = QtWidgets.QPushButton("Delete")
        self.btn_delete_saved.clicked.connect(self._delete_saved)
        saved_btns.addWidget(self.btn_delete_saved)
        self.btn_export_saved = QtWidgets.QPushButton("Export…")
        self.btn_export_saved.clicked.connect(self._export_saved)
        saved_btns.addWidget(self.btn_export_saved)
        self.btn_import_saved = QtWidgets.QPushButton("Import…")
        self.btn_import_saved.clicked.connect(self._import_saved)
        saved_btns.addWidget(self.btn_import_saved)
        saved_layout.addLayout(saved_btns)

        outer.addWidget(saved_box)
        outer.addStretch(1)

        self.manager.pipelines_changed.connect(self._refresh_saved)
        self._refresh_saved()
        self._update_buttons()

    # ----- helpers -----
    def _current_spec(self):
        key = self.cmb_function.currentData()
        if not key:
            return None
        return get_processing_function(str(key))

    def _current_form(self) -> Optional[ParameterForm]:
        key = self.cmb_function.currentData()
        if not key:
            return None
        idx = self._stack_indices.get(str(key))
        if idx is None:
            return None
        widget = self.param_stack.widget(idx)
        return widget if isinstance(widget, ParameterForm) else None

    def _selected_step_index(self) -> int:
        return self.list_steps.currentRow()

    def _selected_saved_name(self) -> Optional[str]:
        items = self.list_saved.selectedItems()
        if not items:
            return None
        return str(items[0].data(QtCore.Qt.UserRole) or items[0].text())

    def _refresh_step_list(self):
        self.list_steps.clear()
        for step in self.steps:
            spec = get_processing_function(step.key)
            label = spec.label if spec else step.key
            summary = summarize_parameters(step.key, step.params)
            text = label
            if summary:
                text += f" ({summary})"
            self.list_steps.addItem(text)
        self._update_buttons()

    def _set_function_selection(self, key: str):
        block = self.cmb_function.blockSignals(True)
        idx = self.cmb_function.findData(key)
        if idx >= 0:
            self.cmb_function.setCurrentIndex(idx)
        self.cmb_function.blockSignals(block)
        self._on_function_changed()

    def _build_pipeline(self) -> ProcessingPipeline:
        name = self.edit_name.text().strip() or "Untitled"
        return ProcessingPipeline(name=name, steps=[ProcessingStep(step.key, dict(step.params)) for step in self.steps])

    # ----- button state -----
    def _update_buttons(self):
        idx = self._selected_step_index()
        has_selection = 0 <= idx < len(self.steps)
        has_steps = bool(self.steps)
        self.btn_update_step.setEnabled(has_selection)
        self.btn_move_up.setEnabled(has_selection and idx > 0)
        self.btn_move_down.setEnabled(has_selection and idx < len(self.steps) - 1)
        self.btn_remove_step.setEnabled(has_selection)
        self.btn_clear_steps.setEnabled(has_steps)
        self.btn_interactive.setEnabled(has_steps)
        self.btn_save_pipeline.setEnabled(has_steps and bool(self.edit_name.text().strip()))
        has_saved = self._selected_saved_name() is not None
        self.btn_load_saved.setEnabled(has_saved)
        self.btn_delete_saved.setEnabled(has_saved)
        self.btn_export_saved.setEnabled(has_saved)

    # ----- slots -----
    def _on_function_changed(self):
        key = self.cmb_function.currentData()
        if key and key in self._stack_indices:
            self.param_stack.setCurrentIndex(self._stack_indices[str(key)])
        else:
            self.param_stack.setCurrentIndex(0)
        self._update_buttons()

    def _on_function_params_changed(self):
        self._update_buttons()

    def _add_step(self):
        spec = self._current_spec()
        form = self._current_form()
        if spec is None or form is None:
            return
        self.steps.append(ProcessingStep(spec.key, form.values()))
        self._refresh_step_list()
        self.list_steps.setCurrentRow(len(self.steps) - 1)

    def _update_step(self):
        idx = self._selected_step_index()
        spec = self._current_spec()
        form = self._current_form()
        if idx < 0 or idx >= len(self.steps) or spec is None or form is None:
            return
        self.steps[idx] = ProcessingStep(spec.key, form.values())
        self._refresh_step_list()
        self.list_steps.setCurrentRow(idx)

    def _on_step_selected(self, index: int):
        if index < 0 or index >= len(self.steps):
            self._update_buttons()
            return
        step = self.steps[index]
        self._set_function_selection(step.key)
        form = self._current_form()
        if form:
            form.set_values(step.params)
        self._update_buttons()

    def _move_step(self, delta: int):
        idx = self._selected_step_index()
        new_idx = idx + delta
        if idx < 0 or new_idx < 0 or new_idx >= len(self.steps):
            return
        self.steps[idx], self.steps[new_idx] = self.steps[new_idx], self.steps[idx]
        self._refresh_step_list()
        self.list_steps.setCurrentRow(new_idx)

    def _remove_step(self):
        idx = self._selected_step_index()
        if idx < 0 or idx >= len(self.steps):
            return
        del self.steps[idx]
        self._refresh_step_list()
        if self.steps:
            self.list_steps.setCurrentRow(min(idx, len(self.steps) - 1))

    def _clear_steps(self):
        if QtWidgets.QMessageBox.question(self, "Clear steps", "Remove all steps from the pipeline?") != QtWidgets.QMessageBox.Yes:
            return
        self.steps.clear()
        self._refresh_step_list()

    def _save_pipeline(self):
        if not self.steps:
            return
        name = self.edit_name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Missing name", "Please enter a name for the pipeline.")
            return
        pipeline = ProcessingPipeline(name=name, steps=[ProcessingStep(step.key, dict(step.params)) for step in self.steps])
        try:
            self.manager.save_pipeline(pipeline)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(e))
            return
        self._refresh_saved()
        items = self.list_saved.findItems(name, QtCore.Qt.MatchExactly)
        if items:
            self.list_saved.setCurrentItem(items[0])
        QtWidgets.QMessageBox.information(self, "Pipeline saved", f"Pipeline '{name}' saved.")

    def _interactive_edit(self):
        if not self.steps:
            return
        pipeline = self._build_pipeline()
        dlg = PipelineEditorDialog(self.manager, pipeline, self)
        dlg.exec_()
        updated = dlg.result_pipeline()
        self.steps = [ProcessingStep(step.key, dict(step.params)) for step in updated.steps]
        self._refresh_step_list()

    def _refresh_saved(self):
        selected = self._selected_saved_name()
        self.list_saved.blockSignals(True)
        self.list_saved.clear()
        for name in self.manager.pipeline_names():
            item = QtWidgets.QListWidgetItem(name)
            item.setData(QtCore.Qt.UserRole, name)
            self.list_saved.addItem(item)
            if selected and name == selected:
                item.setSelected(True)
        if selected is None and self.list_saved.count() > 0:
            self.list_saved.setCurrentRow(0)
        self.list_saved.blockSignals(False)
        self._update_buttons()

    def _load_saved(self):
        name = self._selected_saved_name()
        if not name:
            return
        pipeline = self.manager.get_pipeline(name)
        if pipeline is None:
            return
        self.edit_name.setText(pipeline.name)
        self.steps = [ProcessingStep(step.key, dict(step.params)) for step in pipeline.steps]
        self._refresh_step_list()
        if self.steps:
            self.list_steps.setCurrentRow(0)

    def _delete_saved(self):
        name = self._selected_saved_name()
        if not name:
            return
        if QtWidgets.QMessageBox.question(self, "Delete pipeline", f"Delete pipeline '{name}'?") != QtWidgets.QMessageBox.Yes:
            return
        self.manager.delete_pipeline(name)
        self._refresh_saved()

    def _export_saved(self):
        name = self._selected_saved_name()
        if not name:
            return
        pipeline = self.manager.get_pipeline(name)
        if pipeline is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export pipeline", f"{name}.json", "Pipeline JSON (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(pipeline.to_dict(), fh, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))

    def _import_saved(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import pipeline", "", "Pipeline JSON (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            pipeline = ProcessingPipeline.from_dict(data)
            if not pipeline.name:
                pipeline.name = Path(path).stem
            self.manager.save_pipeline(pipeline)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Import failed", str(e))
            return
        self._refresh_saved()
        items = self.list_saved.findItems(pipeline.name, QtCore.Qt.MatchExactly)
        if items:
            self.list_saved.setCurrentItem(items[0])


class ProcessingSelectionDialog(QtWidgets.QDialog):
    def __init__(self, manager: Optional[ProcessingManager], parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Apply Processing")
        self.resize(420, 360)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(QtWidgets.QLabel("Choose a processing function or pipeline:"))

        self.cmb_mode = QtWidgets.QComboBox()
        layout.addWidget(self.cmb_mode)

        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)

        self._function_forms: Dict[str, ParameterForm] = {}

        none_widget = QtWidgets.QWidget()
        self._none_index = self.stack.addWidget(none_widget)
        self._add_mode_item("No processing", {"type": "none", "stack": self._none_index})

        for spec in list_processing_functions():
            form = ParameterForm(spec.parameters)
            idx = self.stack.addWidget(form)
            self._function_forms[spec.key] = form
            self._add_mode_item(
                f"Function: {spec.label}",
                {"type": "function", "key": spec.key, "stack": idx},
            )

        pipelines = self.manager.list_pipelines() if self.manager else []
        if pipelines:
            self.cmb_mode.insertSeparator(self.cmb_mode.count())
            for pipeline in pipelines:
                summary = QtWidgets.QPlainTextEdit()
                summary.setReadOnly(True)
                summary.setPlainText(self._summarize_pipeline(pipeline))
                summary.setMinimumHeight(160)
                idx = self.stack.addWidget(summary)
                self._add_mode_item(
                    f"Pipeline: {pipeline.name}",
                    {"type": "pipeline", "name": pipeline.name, "stack": idx},
                )

        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.stack.setCurrentIndex(self._none_index)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _add_mode_item(self, label: str, data: Dict[str, object]):
        self.cmb_mode.addItem(label)
        index = self.cmb_mode.count() - 1
        self.cmb_mode.setItemData(index, data)

    def _on_mode_changed(self, index: int):
        data = self.cmb_mode.itemData(index) or {}
        stack_index = data.get("stack", self._none_index)
        try:
            self.stack.setCurrentIndex(int(stack_index))
        except Exception:
            self.stack.setCurrentIndex(self._none_index)

    def _summarize_pipeline(self, pipeline: ProcessingPipeline) -> str:
        if not pipeline.steps:
            return "(No steps)"
        lines: List[str] = []
        for i, step in enumerate(pipeline.steps, 1):
            spec = get_processing_function(step.key)
            label = spec.label if spec else step.key
            summary = summarize_parameters(step.key, step.params)
            text = f"{i}. {label}"
            if summary:
                text += f" — {summary}"
            lines.append(text)
        return "\n".join(lines)

    def selected_processing(self) -> Tuple[str, Dict[str, object]]:
        index = self.cmb_mode.currentIndex()
        data = self.cmb_mode.itemData(index) or {}
        mode_type = data.get("type")
        if mode_type == "function":
            key = str(data.get("key", ""))
            form = self._function_forms.get(key)
            params = form.values() if form else {}
            return key, params
        if mode_type == "pipeline":
            name = str(data.get("name", ""))
            return f"pipeline:{name}", {}
        return "none", {}

# ---------------------------------------------------------------------------
# Datasets pane (left): load a dataset and list 2D variables (drag from here)
# ---------------------------------------------------------------------------
class DatasetsPane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(6,6,6,6)
        self.btn_open = QtWidgets.QPushButton("Load Dataset…")
        self.btn_open.clicked.connect(self._open)
        lay.addWidget(self.btn_open)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Datasets / Variables"])
        self.tree.setDragEnabled(True)
        self.tree.setDefaultDropAction(QtCore.Qt.CopyAction)
        self.tree.setDropIndicatorShown(False)
        self.tree.mimeData = self._mimeData  # override to pack VarRef
        lay.addWidget(self.tree, 1)

        self._populate_examples()

    def _mimeData(self, _items):
        md = QtCore.QMimeData()
        sel = self.tree.selectedItems()
        if sel:
            txt = sel[0].data(0, QtCore.Qt.UserRole)
            if txt:
                md.setText(txt)
        return md

    def _populate_examples(self):
        candidates = [
            Path(__file__).resolve().parent / "example_dataset.nc",
            Path(__file__).resolve().parent / "example_3d_dataset.nc",
            Path(__file__).resolve().parent / "example_rect_warp.nc",
        ]
        for path in candidates:
            if path.exists():
                self._register_dataset(path, quiet=True)

    def _register_dataset(self, path: Path, quiet: bool = False):
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if Path(item.text(0)) == path:
                return

        try:
            ds = open_dataset(path)
        except Exception as e:
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "Open failed", str(e))
            return

        root = QtWidgets.QTreeWidgetItem([str(path)])
        root.setExpanded(True)
        root.setData(0, QtCore.Qt.UserRole, DataSetRef(path).to_mime())
        self.tree.addTopLevelItem(root)

        for var in ds.data_vars:
            if ds[var].ndim != 2:
                continue
            da = ds[var]
            coords = guess_phys_coords(da)
            if "X" in coords and "Y" in coords:
                hint = "(X,Y)"
            elif "x" in coords and "y" in coords:
                hint = "(x,y)"
            else:
                hint = "(pixel)"
            child = QtWidgets.QTreeWidgetItem([f"{var}  {hint}"])
            child.setData(0, QtCore.Qt.UserRole, VarRef(path, var, hint).to_mime())
            root.addChild(child)

        try:
            ds.close()
        except Exception:
            pass

    def _open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open xarray Dataset", "", "NetCDF / Zarr (*.nc *.zarr);;All files (*)"
        )
        if not path:
            return
        self._register_dataset(Path(path))


# ---------------------------------------------------------------------------
# ViewerFrame: one tile with the image + optional histogram on the right
# ---------------------------------------------------------------------------
class ViewerFrame(QtWidgets.QFrame):
    request_close = QtCore.Signal(object)

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
        self._display_mode: str = "image"
        self._current_processing: str = "none"
        self._processing_params: Dict[str, object] = {}
        self._selected: bool = False
        self._dataset_path: Optional[Path] = None
        self._dataset: Optional[xr.Dataset] = None
        self._available_variables: List[str] = []
        self._variable_hints: Dict[str, str] = {}
        self._current_variable: Optional[str] = None

        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(2,2,2,2); lay.setSpacing(2)
        # Header
        hdr = QtWidgets.QFrame(); hl = QtWidgets.QHBoxLayout(hdr); hl.setContentsMargins(6,3,6,3)
        self.lbl = QtWidgets.QLabel(title); hl.addWidget(self.lbl, 1)
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
            if getattr(da, "ndim", 0) != 2:
                continue
            self._available_variables.append(var)
            dims = []
            try:
                dims = [f"{dim}[{getattr(da, 'sizes', {}).get(dim, '?')}]" for dim in da.dims[:2]]
            except Exception:
                dims = []
            if dims:
                self._variable_hints[var] = " (" + " × ".join(dims) + ")"

        self._available_variables.sort()
        self._current_variable = None
        self._update_variable_combo()
        self._clear_display()
        if not self._available_variables:
            self._set_header_text(None)
            self._set_header_text(None, custom=f"{self._dataset_display_name()} — No 2D variables")
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
        if getattr(da, "ndim", 0) != 2:
            return False
        coords = guess_phys_coords(da)
        self.set_data(da, coords)
        self._current_variable = var_name
        self._set_header_text(var_name)
        return True

    def set_data(self, da, coords):
        Z = np.asarray(getattr(da, "values", da), float)
        self._raw_data = np.asarray(Z, float)
        self._processed_data = np.asarray(Z, float)
        self._coords = {}
        coords = dict(coords or {})
        if "X" in coords and "Y" in coords:
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
        try:
            self.viewer.img_item.setVisible(True)
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
        if self._display_mode == "warped" and "X" in self._coords and "Y" in self._coords:
            self.viewer.set_warped(self._coords["X"], self._coords["Y"], data, autorange=autorange)
        elif self._display_mode == "rectilinear" and "x" in self._coords and "y" in self._coords:
            self.viewer.set_rectilinear(self._coords["x"], self._coords["y"], data, autorange=autorange)
        else:
            self.viewer.set_image(data, autorange=autorange)
        try:
            self.viewer.img_item.setVisible(True)
        except Exception:
            pass

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
        if not preserve_header:
            self._set_header_text(None)


# ---------------------------------------------------------------------------
# MultiView grid: drag vars to create tiles; master toggle for histograms
# ---------------------------------------------------------------------------
class MultiViewGrid(QtWidgets.QWidget):
    """
    A splitter-based grid of ViewerFrame tiles.
    - Drag a dataset or variable reference from the DatasetsPane onto this widget to add a tile.
    - 'Columns' spinbox controls how many tiles per row.
    - 'Show histograms' toggles the classic HistogramLUTItem to the right of each tile.
    """
    def __init__(self, processing_manager: Optional[ProcessingManager] = None, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.frames: List[ViewerFrame] = []
        self.processing_manager = processing_manager
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

        # Toolbar
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Columns:"))
        self.col_spin = QtWidgets.QSpinBox()
        self.col_spin.setRange(1, 12)
        self.col_spin.setValue(3)
        self.col_spin.valueChanged.connect(self._reflow)
        bar.addWidget(self.col_spin)

        self.chk_show_hist = QtWidgets.QCheckBox("Show histograms")
        self.chk_show_hist.setChecked(True)        # set False if you prefer off-by-default
        self.chk_show_hist.toggled.connect(self._apply_histogram_visibility)
        bar.addWidget(self.chk_show_hist)

        self.chk_link_levels = QtWidgets.QCheckBox("Lock colorscales")
        self.chk_link_levels.toggled.connect(self._on_link_levels_toggled)
        bar.addWidget(self.chk_link_levels)

        self.chk_link_panzoom = QtWidgets.QCheckBox("Lock pan/zoom")
        self.chk_link_panzoom.toggled.connect(self._on_link_panzoom_toggled)
        bar.addWidget(self.chk_link_panzoom)

        self.chk_cursor_mirror = QtWidgets.QCheckBox("Mirror cursor")
        self.chk_cursor_mirror.setChecked(False)
        self.chk_cursor_mirror.toggled.connect(self._on_link_cursor_toggled)
        bar.addWidget(self.chk_cursor_mirror)

        self.btn_autoscale = QtWidgets.QPushButton("Autoscale colors")
        self.btn_autoscale.clicked.connect(self._autoscale_colors)
        bar.addWidget(self.btn_autoscale)

        self.btn_autopan = QtWidgets.QPushButton("Auto pan/zoom")
        self.btn_autopan.clicked.connect(self._auto_panzoom)
        bar.addWidget(self.btn_autopan)

        self.btn_equalize_rows = QtWidgets.QPushButton("Equalize rows")
        self.btn_equalize_rows.clicked.connect(self.equalize_rows)
        bar.addWidget(self.btn_equalize_rows)

        self.btn_equalize_cols = QtWidgets.QPushButton("Equalize columns")
        self.btn_equalize_cols.clicked.connect(self.equalize_columns)
        bar.addWidget(self.btn_equalize_cols)

        self.btn_select_all = QtWidgets.QPushButton("Select All Plots")
        self.btn_select_all.setEnabled(False)
        self.btn_select_all.clicked.connect(self._select_all_frames)
        bar.addWidget(self.btn_select_all)

        self.btn_apply_processing = QtWidgets.QPushButton("Apply processing…")
        self.btn_apply_processing.setEnabled(False)
        self.btn_apply_processing.clicked.connect(self._on_apply_processing_clicked)
        bar.addWidget(self.btn_apply_processing)

        bar.addStretch(1)
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

    # ---------- Drag & Drop ----------
    def dragEnterEvent(self, ev: QtGui.QDragEnterEvent):
        ev.acceptProposedAction() if ev.mimeData().hasText() else ev.ignore()

    def dropEvent(self, ev: QtGui.QDropEvent):
        text = ev.mimeData().text()
        ds_ref = DataSetRef.from_mime(text)
        vr = None if ds_ref else VarRef.from_mime(text)

        if not ds_ref and not vr:
            ev.ignore()
            return

        dataset = None
        frame_title = ""
        try:
            if ds_ref:
                dataset = ds_ref.load()
                frame_title = ds_ref.path.name
            elif vr:
                dataset = open_dataset(vr.path)
                frame_title = vr.path.name
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(e))
            ev.ignore()
            return

        fr = ViewerFrame(title=frame_title, parent=self)
        fr.request_close.connect(self._remove_frame)
        try:
            if ds_ref:
                fr.set_dataset(dataset, ds_ref.path)
            elif vr:
                fr.set_dataset(dataset, vr.path, select=vr.var)
        except Exception as exc:
            try:
                close = getattr(dataset, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            ev.ignore()
            return

        self.frames.append(fr)
        self._connect_frame_signals(fr)
        self._sync_new_frame_to_links(fr)
        fr.set_selected(False)

        self._reflow()
        self._update_apply_button_state()
        ev.acceptProposedAction()

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

    def _on_apply_processing_clicked(self):
        frames = self.selected_frames()
        if not frames:
            return
        dialog = ProcessingSelectionDialog(self.processing_manager, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        mode, params = dialog.selected_processing()
        for frame in frames:
            try:
                frame.apply_processing(mode, params, self.processing_manager)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Processing failed", str(exc))
                break


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

# ---------------------------------------------------------------------------
# Sequential view helpers
# ---------------------------------------------------------------------------


class SequentialRoiWindow(QtWidgets.QWidget):
    axesChanged = QtCore.Signal(tuple)
    reducerChanged = QtCore.Signal(str)
    closed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Sequential ROI Inspector")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.setMinimumSize(360, 520)
        self.resize(420, 600)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)

        controls.addWidget(QtWidgets.QLabel("Reduce over:"))
        self.cmb_axes = QtWidgets.QComboBox()
        self.cmb_axes.currentIndexChanged.connect(self._emit_axes_changed)
        self.cmb_axes.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        controls.addWidget(self.cmb_axes, 1)

        controls.addWidget(QtWidgets.QLabel("Statistic:"))
        self.cmb_method = QtWidgets.QComboBox()
        self.cmb_method.currentIndexChanged.connect(self._emit_method_changed)
        self.cmb_method.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        controls.addWidget(self.cmb_method, 1)

        layout.addLayout(controls)

        self.lbl_hint = QtWidgets.QLabel()
        self.lbl_hint.setStyleSheet("color: #666;")
        layout.addWidget(self.lbl_hint)

        profile_axes = {
            "bottom": ScientificAxisItem("bottom"),
            "left": ScientificAxisItem("left"),
        }
        self.profile_plot = pg.PlotWidget(axisItems=profile_axes)
        self.profile_plot.setMinimumHeight(140)
        self.profile_plot.showGrid(x=True, y=True, alpha=0.3)
        self.profile_plot.setLabel("bottom", "Axis")
        self.profile_plot.setLabel("left", "Value")
        self.profile_curve = self.profile_plot.plot([], [], pen=pg.mkPen('#ffaa00', width=2))
        layout.addWidget(self.profile_plot, 1)

        slice_axes = {
            "bottom": ScientificAxisItem("bottom"),
            "left": ScientificAxisItem("left"),
        }
        self.slice_plot = pg.PlotWidget(axisItems=slice_axes)
        self.slice_plot.setMinimumHeight(160)
        self.slice_plot.showGrid(x=True, y=True, alpha=0.3)
        self.slice_plot.setLabel("bottom", "Slice coordinate")
        self.slice_plot.setLabel("left", "ROI statistic")
        self.slice_curve = self.slice_plot.plot([], [], pen=pg.mkPen('#66bbff', width=2))
        layout.addWidget(self.slice_plot, 1)

        self._updating = False

    def set_axis_options(
        self, options: List[Tuple[str, Tuple[int, ...], str, str, Optional[int]]], current_index: int
    ):
        self._updating = True
        self.cmb_axes.clear()
        for entry in options:
            if not entry:
                continue
            label = entry[0]
            axes = entry[1] if len(entry) > 1 else ()
            self.cmb_axes.addItem(label, tuple(int(a) for a in axes))
        self.cmb_axes.setEnabled(bool(options))
        if options:
            self.cmb_axes.setCurrentIndex(max(0, min(current_index, len(options) - 1)))
        self._updating = False

    def set_reducer_options(self, reducers: Dict[str, Tuple[str, object]], current_key: str):
        self._updating = True
        self.cmb_method.clear()
        for key, (label, _fn) in reducers.items():
            self.cmb_method.addItem(label, key)
        idx = max(0, self.cmb_method.findData(current_key))
        self.cmb_method.setCurrentIndex(idx)
        self.cmb_method.setEnabled(self.cmb_method.count() > 0)
        self._updating = False

    def set_hint(self, text: str):
        self.lbl_hint.setText(text)

    def update_profile(self, xs: List[float], ys: List[float], xlabel: str, ylabel: str, visible: bool):
        self.profile_plot.setVisible(visible)
        if not visible:
            self.profile_curve.setData([], [])
            return
        self.profile_plot.setLabel("bottom", xlabel)
        self.profile_plot.setLabel("left", ylabel)
        self.profile_curve.setData(xs, ys)
        self.profile_plot.enableAutoRange()

    def update_slice_curve(self, xs: List[float], ys: List[float], xlabel: str, ylabel: str):
        self.slice_plot.setLabel("bottom", xlabel)
        self.slice_plot.setLabel("left", ylabel)
        self.slice_curve.setData(xs, ys)
        self.slice_plot.enableAutoRange()

    def _emit_axes_changed(self):
        if self._updating:
            return
        data = self.cmb_axes.currentData()
        if data is None:
            return
        try:
            axes = tuple(int(a) for a in data)
        except Exception:
            axes = tuple()
        self.axesChanged.emit(axes)

    def _emit_method_changed(self):
        if self._updating:
            return
        key = self.cmb_method.currentData()
        if key:
            self.reducerChanged.emit(str(key))

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.closed.emit()
        except Exception:
            pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Sequential view: 3D volume visualization helper
# ---------------------------------------------------------------------------


class SequentialVolumeWindow(QtWidgets.QWidget):
    closed = QtCore.Signal()

    def __init__(self, parent=None):
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

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)

        controls.addWidget(QtWidgets.QLabel("Visible range:"))

        self.spin_min = QtWidgets.QDoubleSpinBox()
        self.spin_min.setDecimals(6)
        self.spin_min.valueChanged.connect(self._on_range_changed)
        controls.addWidget(self.spin_min)

        self.spin_max = QtWidgets.QDoubleSpinBox()
        self.spin_max.setDecimals(6)
        self.spin_max.valueChanged.connect(self._on_range_changed)
        controls.addWidget(self.spin_max)

        self.btn_reset_range = QtWidgets.QPushButton("Reset")
        self.btn_reset_range.clicked.connect(self._on_reset_range)
        controls.addWidget(self.btn_reset_range)

        controls.addStretch(1)
        layout.addLayout(controls)

        self.view = gl.GLViewWidget()
        self.view.opts["distance"] = 400
        self.view.setBackgroundColor(QtGui.QColor(20, 20, 20))
        layout.addWidget(self.view, 1)

        self._volume_item: Optional[gl.GLVolumeItem] = None

    # ----- public API -----
    def set_volume(self, data: Optional[np.ndarray]):
        if data is None or data.size == 0:
            self._data = None
            self._remove_volume()
            self._update_range_controls()
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
        self._update_range_controls(reset=True)
        self._ensure_volume_item()
        self._apply_volume_data()

    def set_colormap(self, name: Optional[str]):
        if name:
            self._colormap_name = str(name)
        else:
            self._colormap_name = "viridis"
        self._apply_transfer_function()

    def clear_volume(self):
        self._data = None
        self._remove_volume()
        self._update_range_controls()

    # ----- helpers -----
    def _ensure_volume_item(self):
        if self._volume_item is not None:
            return
        data = self._data
        if data is None:
            return
        volume = self._prepare_volume_array(data)
        self._volume_item = gl.GLVolumeItem(volume, smooth=False)
        self._volume_item.setGLOptions("additive")
        self.view.addItem(self._volume_item)
        self._apply_transfer_function()

    def _prepare_volume_array(self, data: np.ndarray) -> np.ndarray:
        if data.ndim != 3:
            return np.zeros((1, 1, 1), dtype=np.float32)
        transposed = np.transpose(data, (2, 1, 0))
        return np.ascontiguousarray(transposed, dtype=np.float32)

    def _apply_volume_data(self):
        if self._data is None or self._volume_item is None:
            return
        volume = self._prepare_volume_array(self._data)
        self._volume_item.setData(volume)
        self._volume_item.setLevels((self._data_min, self._data_max))
        self._apply_transfer_function()

    def _apply_transfer_function(self):
        if self._volume_item is None or self._data is None:
            return
        try:
            cmap = pg.colormap.get(self._colormap_name)
        except Exception:
            cmap = pg.colormap.get("viridis")
        lut = cmap.getLookupTable(nPts=512, alpha=True)
        if lut.shape[1] == 3:
            alpha = np.full((lut.shape[0], 1), 255, dtype=np.uint8)
            lut = np.hstack([lut, alpha])
        values = np.linspace(self._data_min, self._data_max, lut.shape[0])
        lo = min(self.spin_min.value(), self.spin_max.value())
        hi = max(self.spin_min.value(), self.spin_max.value())
        alpha_mask = ((values >= lo) & (values <= hi)).astype(float)
        lut[:, 3] = (alpha_mask * 255).astype(np.uint8)
        self._volume_item.setLookupTable(lut)
        self._volume_item.setLevels((self._data_min, self._data_max))

    def _remove_volume(self):
        if self._volume_item is None:
            return
        try:
            self.view.removeItem(self._volume_item)
        except Exception:
            pass
        self._volume_item = None

    def _update_range_controls(self, *, reset: bool = False):
        block_min = self.spin_min.blockSignals(True)
        block_max = self.spin_max.blockSignals(True)
        try:
            if self._data is None:
                self.spin_min.setEnabled(False)
                self.spin_max.setEnabled(False)
                self.btn_reset_range.setEnabled(False)
                self.spin_min.setValue(0.0)
                self.spin_max.setValue(0.0)
            else:
                self.spin_min.setEnabled(True)
                self.spin_max.setEnabled(True)
                self.btn_reset_range.setEnabled(True)
                self.spin_min.setRange(self._data_min, self._data_max)
                self.spin_max.setRange(self._data_min, self._data_max)
                if reset:
                    self.spin_min.setValue(self._data_min)
                    self.spin_max.setValue(self._data_max)
        finally:
            self.spin_min.blockSignals(block_min)
            self.spin_max.blockSignals(block_max)
        if reset:
            self._apply_transfer_function()

    def _on_range_changed(self):
        if self._data is None:
            return
        if self.spin_min.value() > self.spin_max.value():
            return
        self._apply_transfer_function()

    def _on_reset_range(self):
        if self._data is None:
            return
        with QtCore.QSignalBlocker(self.spin_min), QtCore.QSignalBlocker(self.spin_max):
            self.spin_min.setValue(self._data_min)
            self.spin_max.setValue(self._data_max)
        self._apply_transfer_function()

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.closed.emit()
        except Exception:
            pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Sequential view: explore 2D slices along an arbitrary axis
# ---------------------------------------------------------------------------


class SequentialView(QtWidgets.QWidget):
    def __init__(self, processing_manager: Optional[ProcessingManager] = None, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.processing_manager = processing_manager

        self._dataset: Optional[xr.Dataset] = None
        self._dataset_path: Optional[Path] = None
        self._current_variable: Optional[str] = None
        self._current_da: Optional[xr.DataArray] = None
        self._dims: List[str] = []
        self._slice_axis: Optional[str] = None
        self._row_axis: Optional[str] = None
        self._col_axis: Optional[str] = None
        self._fixed_indices: Dict[str, int] = {}
        self._fixed_dim_widgets: Dict[str, QtWidgets.QSpinBox] = {}
        self._processing_mode: str = "none"
        self._processing_params: Dict[str, object] = {}
        self._slice_index: int = 0
        self._slice_count: int = 0
        self._axis_coords: Optional[np.ndarray] = None
        self._current_processed_slice: Optional[np.ndarray] = None
        self._roi_last_shape: Optional[Tuple[int, int]] = None
        self._roi_last_bounds: Optional[Tuple[int, int, int, int]] = None

        self._roi_enabled: bool = False
        self._roi_reducers = {
            "mean": ("Mean", _nan_aware_reducer(lambda arr, axis=None: np.nanmean(arr, axis=axis))),
            "median": ("Median", _nan_aware_reducer(lambda arr, axis=None: np.nanmedian(arr, axis=axis))),
            "min": ("Minimum", _nan_aware_reducer(lambda arr, axis=None: np.nanmin(arr, axis=axis))),
            "max": ("Maximum", _nan_aware_reducer(lambda arr, axis=None: np.nanmax(arr, axis=axis))),
            "std": ("Std. dev", _nan_aware_reducer(lambda arr, axis=None: np.nanstd(arr, axis=axis))),
            "ptp": (
                "Peak-to-peak",
                _nan_aware_reducer(
                    lambda arr, axis=None: np.nanmax(arr, axis=axis) - np.nanmin(arr, axis=axis)
                ),
            ),
        }
        self._roi_method_key: str = "mean"
        self._roi_last_slices: Optional[Tuple[slice, slice]] = None
        self._roi_axis_options: List[Tuple[str, Tuple[int, ...], str, str, Optional[int]]] = []
        self._roi_axes_selection: Tuple[int, ...] = (0, 1)
        self._roi_axis_index: int = 0
        self._current_slice_coords: Dict[str, np.ndarray] = {}
        self._row_coord_1d: Optional[np.ndarray] = None
        self._row_coord_2d: Optional[np.ndarray] = None
        self._col_coord_1d: Optional[np.ndarray] = None
        self._col_coord_2d: Optional[np.ndarray] = None
        self._volume_cache: Optional[np.ndarray] = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        hint = QtWidgets.QLabel(
            "Drop a dataset here or use the Load button to explore sequential slices."
        )
        hint.setStyleSheet("color: #666;")
        outer.addWidget(hint)

        top = QtWidgets.QHBoxLayout()
        self.lbl_dataset = QtWidgets.QLabel("No dataset loaded")
        self.lbl_dataset.setStyleSheet("color: #555;")
        top.addWidget(self.lbl_dataset, 1)
        self.btn_load = QtWidgets.QPushButton("Load dataset…")
        self.btn_load.clicked.connect(self._load_dataset_dialog)
        top.addWidget(self.btn_load, 0)
        outer.addLayout(top)

        var_row = QtWidgets.QHBoxLayout()
        var_row.addWidget(QtWidgets.QLabel("Variable:"))
        self.cmb_variable = QtWidgets.QComboBox()
        self.cmb_variable.setEnabled(False)
        self.cmb_variable.currentIndexChanged.connect(self._on_variable_changed)
        var_row.addWidget(self.cmb_variable, 1)
        outer.addLayout(var_row)

        axis_group = QtWidgets.QGroupBox("Slice configuration")
        axis_form = QtWidgets.QFormLayout(axis_group)
        axis_form.setContentsMargins(6, 6, 6, 6)
        axis_form.setSpacing(6)

        self.cmb_slice_axis = QtWidgets.QComboBox()
        self.cmb_slice_axis.currentIndexChanged.connect(self._on_axes_changed)
        axis_form.addRow("Slice axis", self.cmb_slice_axis)

        self.cmb_row_axis = QtWidgets.QComboBox()
        self.cmb_row_axis.currentIndexChanged.connect(self._on_axes_changed)
        axis_form.addRow("Rows", self.cmb_row_axis)

        self.cmb_col_axis = QtWidgets.QComboBox()
        self.cmb_col_axis.currentIndexChanged.connect(self._on_axes_changed)
        axis_form.addRow("Columns", self.cmb_col_axis)

        self.fixed_dims_container = QtWidgets.QWidget()
        self.fixed_dims_layout = QtWidgets.QFormLayout(self.fixed_dims_container)
        self.fixed_dims_layout.setContentsMargins(0, 0, 0, 0)
        self.fixed_dims_layout.setSpacing(4)
        axis_form.addRow("Fixed indices", self.fixed_dims_container)

        outer.addWidget(axis_group)

        slider_row = QtWidgets.QHBoxLayout()
        self.lbl_slice = QtWidgets.QLabel("Slice: –")
        slider_row.addWidget(self.lbl_slice, 0)
        self.sld_slice = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_slice.setEnabled(False)
        self.sld_slice.valueChanged.connect(self._on_slice_changed)
        slider_row.addWidget(self.sld_slice, 1)
        self.spin_slice = QtWidgets.QSpinBox()
        self.spin_slice.setEnabled(False)
        self.spin_slice.valueChanged.connect(self._on_slice_spin_changed)
        slider_row.addWidget(self.spin_slice, 0)
        outer.addLayout(slider_row)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_apply_processing = QtWidgets.QPushButton("Apply processing…")
        self.btn_apply_processing.setEnabled(False)
        self.btn_apply_processing.clicked.connect(self._choose_processing)
        btn_row.addWidget(self.btn_apply_processing)

        self.btn_reset_processing = QtWidgets.QPushButton("Reset processing")
        self.btn_reset_processing.setEnabled(False)
        self.btn_reset_processing.clicked.connect(self._reset_processing)
        btn_row.addWidget(self.btn_reset_processing)

        self.btn_autoscale = QtWidgets.QPushButton("Autoscale colors")
        self.btn_autoscale.setEnabled(False)
        self.btn_autoscale.clicked.connect(self._on_autoscale_clicked)
        btn_row.addWidget(self.btn_autoscale)

        self.btn_autorange = QtWidgets.QPushButton("Auto view")
        self.btn_autorange.setEnabled(False)
        self.btn_autorange.clicked.connect(self._on_autorange_clicked)
        btn_row.addWidget(self.btn_autorange)

        btn_row.addSpacing(12)

        cmap_label = QtWidgets.QLabel("Color map:")
        cmap_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        btn_row.addWidget(cmap_label, 0)

        self.cmb_colormap = QtWidgets.QComboBox()
        self.cmb_colormap.setEnabled(False)
        self.cmb_colormap.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cmb_colormap.currentIndexChanged.connect(self._on_colormap_changed)
        btn_row.addWidget(self.cmb_colormap, 0)
        self._populate_colormap_choices()

        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self.viewer_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.viewer_split.setChildrenCollapsible(False)
        self.viewer_split.setHandleWidth(6)
        outer.addWidget(self.viewer_split, 1)

        self.viewer = CentralPlotWidget(self)
        self.viewer_split.addWidget(self.viewer)
        hist = self.viewer.histogram_widget()
        if hist is not None and self.viewer_split.indexOf(hist) == -1:
            hist.setSizePolicy(
                QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
            )
            hist.setMinimumWidth(140)
            hist.setMaximumWidth(180)
            self.viewer_split.addWidget(hist)
            try:
                self.viewer_split.setStretchFactor(0, 1)
                self.viewer_split.setStretchFactor(1, 0)
            except Exception:
                pass
            QtCore.QTimer.singleShot(0, lambda: self.viewer_split.setSizes([600, 150]))

        roi_row = QtWidgets.QHBoxLayout()
        self.btn_toggle_roi = QtWidgets.QPushButton("Enable ROI")
        self.btn_toggle_roi.setCheckable(True)
        self.btn_toggle_roi.setEnabled(False)
        self.btn_toggle_roi.toggled.connect(self._on_roi_toggled)
        roi_row.addWidget(self.btn_toggle_roi)

        self.btn_volume_view = QtWidgets.QPushButton("Open volume view…")
        self.btn_volume_view.setEnabled(False)
        self.btn_volume_view.clicked.connect(self._open_volume_view)
        if gl is None:
            self.btn_volume_view.setToolTip(
                "3D volume rendering requires the optional pyqtgraph.opengl module"
            )
        roi_row.addWidget(self.btn_volume_view)

        self.lbl_roi_status = QtWidgets.QLabel("ROI disabled")
        self.lbl_roi_status.setStyleSheet("color: #666;")
        roi_row.addWidget(self.lbl_roi_status, 1)
        outer.addLayout(roi_row)

        self.roi = pg.RectROI([10, 10], [40, 40], pen=pg.mkPen('#ffaa00', width=2))
        self.roi.addScaleHandle((1, 1), (0, 0))
        self.roi.addScaleHandle((0, 0), (1, 1))
        self.roi.hide()
        self.roi.sigRegionChanged.connect(self._on_roi_region_changed)
        try:
            self.roi.sigRegionChangeFinished.connect(self._on_roi_region_changed)
        except Exception:
            pass

        self._roi_window: Optional[SequentialRoiWindow] = None
        self._volume_window: Optional[SequentialVolumeWindow] = None

    # ---------- dataset helpers ----------
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
            self.cmb_colormap.addItem("Default", "default")
        self.cmb_colormap.blockSignals(False)
        if self.cmb_colormap.count():
            self.cmb_colormap.setCurrentIndex(0)

    def _on_colormap_changed(self):
        if not hasattr(self, "viewer"):
            return
        self._apply_selected_colormap()
        self._update_volume_window_colormap()

    def _apply_selected_colormap(self):
        if not hasattr(self, "viewer"):
            return
        name = self.cmb_colormap.currentData()
        if not name or name == "default":
            target = "viridis"
        else:
            target = str(name)
        try:
            cmap = pg.colormap.get(target)
        except Exception:
            return
        try:
            self.viewer.lut.gradient.setColorMap(cmap)
        except Exception:
            return
        try:
            self.viewer.lut.rehide_stops()
        except Exception:
            pass
        self._update_volume_window_colormap()

    def set_processing_manager(self, manager: Optional[ProcessingManager]):
        self.processing_manager = manager

    def dragEnterEvent(self, ev: QtGui.QDragEnterEvent):
        if ev.mimeData().hasText():
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dropEvent(self, ev: QtGui.QDropEvent):
        text = ev.mimeData().text()
        ref = DataSetRef.from_mime(text)
        if not ref:
            ev.ignore()
            return
        try:
            ds = ref.load()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            ev.ignore()
            return
        self._set_dataset(ds, ref.path)
        ev.acceptProposedAction()

    def _load_dataset_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open dataset",
            "",
            "NetCDF / Zarr (*.nc *.zarr);;All files (*)",
        )
        if not path:
            return
        p = Path(path)
        try:
            ds = open_dataset(p)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            return
        self._set_dataset(ds, p)

    def _clear_view(self):
        self._reset_current_state()
        self._clear_fixed_dim_widgets()
        self.cmb_variable.blockSignals(True)
        self.cmb_variable.clear()
        self.cmb_variable.blockSignals(False)
        for combo in (self.cmb_slice_axis, self.cmb_row_axis, self.cmb_col_axis):
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)
            combo.setEnabled(False)
        self.cmb_variable.setEnabled(False)
        self._clear_display()

    def _clear_display(self):
        if self.roi.scene() is not None:
            try:
                self.viewer.plot.removeItem(self.roi)
            except Exception:
                pass
        self.roi.hide()
        self.btn_toggle_roi.blockSignals(True)
        self.btn_toggle_roi.setChecked(False)
        self.btn_toggle_roi.blockSignals(False)
        self.btn_toggle_roi.setEnabled(False)
        self.btn_volume_view.setEnabled(False)
        if self._roi_window is not None:
            try:
                self._roi_window.hide()
                self._roi_window.update_slice_curve([], [], "Slice coordinate", "ROI statistic")
                self._roi_window.update_profile([], [], "", "", False)
            except Exception:
                pass
        if self._volume_window is not None:
            try:
                self._volume_window.clear_volume()
                self._volume_window.hide()
            except Exception:
                pass
        self.lbl_roi_status.setText("ROI disabled")
        self._roi_enabled = False
        self._roi_last_slices = None
        self._roi_last_bounds = None
        self._roi_last_shape = None
        self._cache_slice_coords({})
        self._roi_axis_options = []
        self._roi_axes_selection = (0, 1)
        self._roi_axis_index = 0
        self._volume_cache = None
        self.viewer.set_image(np.zeros((1, 1)), autorange=True)
        self.sld_slice.blockSignals(True)
        self.spin_slice.blockSignals(True)
        self.sld_slice.setRange(0, 0)
        self.spin_slice.setRange(0, 0)
        self.sld_slice.setValue(0)
        self.spin_slice.setValue(0)
        self.sld_slice.blockSignals(False)
        self.spin_slice.blockSignals(False)
        self.sld_slice.setEnabled(False)
        self.spin_slice.setEnabled(False)
        self.btn_apply_processing.setEnabled(False)
        self.btn_reset_processing.setEnabled(False)
        self.btn_autoscale.setEnabled(False)
        self.btn_autorange.setEnabled(False)
        self.cmb_colormap.setEnabled(False)
        self.lbl_slice.setText("Slice: –")

    def _set_dataset(self, ds: xr.Dataset, path: Optional[Path]):
        if self._dataset is not None and self._dataset is not ds:
            try:
                self._dataset.close()
            except Exception:
                pass
        self._dataset = ds
        self._dataset_path = Path(path) if path else None
        self.lbl_dataset.setText(self._dataset_path.name if self._dataset_path else "(in-memory dataset)")
        self.lbl_dataset.setStyleSheet("")
        self._clear_view()
        vars_with_dims = [var for var in ds.data_vars if ds[var].ndim >= 3]
        if not vars_with_dims:
            self.cmb_variable.addItem("No 3D variables available", None)
            return
        self.cmb_variable.blockSignals(True)
        for var in vars_with_dims:
            dims = " × ".join(str(d) for d in ds[var].dims)
            self.cmb_variable.addItem(f"{var}  ({dims})", var)
        self.cmb_variable.setEnabled(True)
        self.cmb_variable.setCurrentIndex(0)
        self.cmb_variable.blockSignals(False)
        self._on_variable_changed()

    def _reset_current_state(self):
        self._current_variable = None
        self._current_da = None
        self._dims = []
        self._slice_axis = None
        self._row_axis = None
        self._col_axis = None
        self._fixed_indices = {}
        self._fixed_dim_widgets = {}
        self._processing_mode = "none"
        self._processing_params = {}
        self._slice_index = 0
        self._slice_count = 0
        self._axis_coords = None
        self._current_processed_slice = None
        self._roi_last_shape = None
        self._roi_last_slices = None
        self._roi_last_bounds = None
        self._roi_axis_options = []
        self._roi_axis_index = 0
        self._roi_axes_selection = (0, 1)
        self._volume_cache = None
        self._update_roi_window_options()

    def _clear_fixed_dim_widgets(self):
        while self.fixed_dims_layout.rowCount():
            self.fixed_dims_layout.removeRow(0)
        self._fixed_dim_widgets.clear()
        self._fixed_indices.clear()

    # ---------- configuration ----------
    def _on_variable_changed(self):
        if self._dataset is None:
            return
        self._reset_current_state()
        self._clear_fixed_dim_widgets()
        index = self.cmb_variable.currentIndex()
        var = self.cmb_variable.itemData(index)
        if not var:
            self._clear_display()
            return
        da = self._dataset[var]
        self._current_variable = var
        self._current_da = da
        self._dims = list(da.dims)
        self._processing_mode = "none"
        self._processing_params = {}
        self._slice_index = 0
        self._slice_count = 0
        self._axis_coords = None
        self._current_processed_slice = None
        self._rebuild_axis_controls()
        self._update_slice_widgets()
        self._update_slice_display(autorange=True)
        self._update_roi_axis_options()

    def _rebuild_axis_controls(self):
        dims = self._dims
        combos = (self.cmb_slice_axis, self.cmb_row_axis, self.cmb_col_axis)
        for combo in combos:
            combo.blockSignals(True)
            combo.clear()
            for dim in dims:
                combo.addItem(dim, dim)
            combo.blockSignals(False)
            combo.setEnabled(bool(dims))
        if len(dims) >= 3:
            self.cmb_slice_axis.setCurrentIndex(0)
            self.cmb_row_axis.setCurrentIndex(1)
            self.cmb_col_axis.setCurrentIndex(2)
        elif len(dims) >= 2:
            self.cmb_slice_axis.setCurrentIndex(0)
            self.cmb_row_axis.setCurrentIndex(1)
            self.cmb_col_axis.setCurrentIndex(0)
        self._ensure_unique_axes()
        self._rebuild_fixed_indices()

    def _ensure_unique_axes(self):
        dims = self._dims
        combos = (self.cmb_slice_axis, self.cmb_row_axis, self.cmb_col_axis)
        seen: List[str] = []
        for combo in combos:
            idx = combo.currentIndex()
            dim = combo.itemData(idx)
            if dim is None:
                continue
            if dim in seen:
                for alt in dims:
                    if alt not in seen:
                        block = combo.blockSignals(True)
                        combo.setCurrentIndex(combo.findData(alt))
                        combo.blockSignals(block)
                        dim = alt
                        break
            seen.append(dim)
        self._slice_axis = self.cmb_slice_axis.currentData()
        self._row_axis = self.cmb_row_axis.currentData()
        self._col_axis = self.cmb_col_axis.currentData()
        self._update_axis_coords()

    def _update_axis_coords(self):
        axis = self._slice_axis
        if self._current_da is None or axis is None:
            self._axis_coords = None
            return
        coord = self._current_da.coords.get(axis)
        if coord is None:
            self._axis_coords = None
            return
        try:
            self._axis_coords = np.asarray(coord.values)
        except Exception:
            self._axis_coords = None

    def _axis_display_name(self, axis: Optional[str]) -> str:
        if not axis:
            return "axis"
        text = str(axis).replace("_", " ").strip()
        return text or "axis"

    def _cache_slice_coords(self, coords: Optional[Dict[str, np.ndarray]]):
        cache = dict(coords or {})
        self._current_slice_coords = cache

        def _extract(key: str, allowed_ndim: Tuple[int, ...]) -> Optional[np.ndarray]:
            arr = cache.get(key)
            if arr is None:
                return None
            try:
                arr = np.asarray(arr, float)
            except Exception:
                return None
            if arr.size == 0:
                return None
            if allowed_ndim and arr.ndim not in allowed_ndim:
                return None
            return arr

        def _first_valid(keys: Iterable[str], allowed_ndim: Tuple[int, ...]) -> Optional[np.ndarray]:
            for key in keys:
                if not key:
                    continue
                arr = _extract(key, allowed_ndim)
                if arr is not None:
                    return arr
            return None

        self._row_coord_1d = _first_valid(
            ("row_values", "y", self._row_axis or ""), (1,)
        )
        self._col_coord_1d = _first_valid(
            ("col_values", "x", self._col_axis or ""), (1,)
        )
        self._row_coord_2d = _first_valid(("row_grid", "Y"), (2,))
        self._col_coord_2d = _first_valid(("col_grid", "X"), (2,))

    def _column_coordinates(self, start: int, stop: int) -> Optional[np.ndarray]:
        if self._col_coord_1d is not None and self._col_coord_1d.size >= stop:
            return np.asarray(self._col_coord_1d[start:stop], float)
        if self._col_coord_2d is not None and self._col_coord_2d.shape[1] >= stop:
            subset = self._col_coord_2d[:, start:stop]
            with np.errstate(all="ignore"):
                vals = np.nanmean(subset, axis=0)
            return np.asarray(vals, float)
        return None

    def _row_coordinates(self, start: int, stop: int) -> Optional[np.ndarray]:
        if self._row_coord_1d is not None and self._row_coord_1d.size >= stop:
            return np.asarray(self._row_coord_1d[start:stop], float)
        if self._row_coord_2d is not None and self._row_coord_2d.shape[0] >= stop:
            subset = self._row_coord_2d[start:stop, :]
            with np.errstate(all="ignore"):
                vals = np.nanmean(subset, axis=1)
            return np.asarray(vals, float)
        return None

    def _roi_profile_coordinates(self, profile_axis: int, length: int) -> List[float]:
        if length <= 0:
            return []
        bounds = self._roi_last_bounds or self._roi_bounds_from_geometry()
        if bounds is None:
            start = 0
        else:
            y0, y1, x0, x1 = bounds
            start = x0 if profile_axis == 1 else y0
        stop = start + length
        coords = (
            self._column_coordinates(start, stop)
            if profile_axis == 1
            else self._row_coordinates(start, stop)
        )
        if coords is None or coords.size != length:
            coords = np.arange(start, start + length, dtype=float)
        return [float(v) if np.isfinite(v) else np.nan for v in np.asarray(coords, float)]

    def _rebuild_fixed_indices(self):
        self._clear_fixed_dim_widgets()
        if self._current_da is None:
            return
        for dim in self._current_da.dims:
            if dim in (self._slice_axis, self._row_axis, self._col_axis):
                continue
            size = int(self._current_da.sizes.get(dim, 1))
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, max(0, size - 1))
            spin.setValue(0)
            spin.valueChanged.connect(partial(self._on_fixed_index_changed, dim))
            self.fixed_dims_layout.addRow(dim, spin)
            self._fixed_dim_widgets[dim] = spin
            self._fixed_indices[dim] = 0

    def _on_axes_changed(self):
        if not self._dims:
            return
        self._ensure_unique_axes()
        self._invalidate_volume_cache()
        self._slice_index = 0
        self._rebuild_fixed_indices()
        self._update_slice_widgets()
        self._update_slice_display(autorange=True)
        self._update_roi_axis_options()

    def _on_fixed_index_changed(self, dim: str, value: int):
        self._fixed_indices[dim] = int(value)
        self._invalidate_volume_cache()
        self._update_slice_display()

    def _update_slice_widgets(self):
        axis = self._slice_axis
        if not axis or self._current_da is None:
            self._clear_display()
            return
        size = int(self._current_da.sizes.get(axis, 0))
        self._slice_count = size
        self._slice_index = min(self._slice_index, max(0, size - 1))
        self.sld_slice.blockSignals(True)
        self.spin_slice.blockSignals(True)
        self.sld_slice.setRange(0, max(0, size - 1))
        self.spin_slice.setRange(0, max(0, size - 1))
        self.sld_slice.setValue(self._slice_index)
        self.spin_slice.setValue(self._slice_index)
        self.sld_slice.blockSignals(False)
        self.spin_slice.blockSignals(False)
        enabled = size > 0
        self.sld_slice.setEnabled(enabled)
        self.spin_slice.setEnabled(enabled)
        self.btn_apply_processing.setEnabled(enabled)
        self.btn_reset_processing.setEnabled(enabled)
        self.btn_autoscale.setEnabled(enabled)
        self.btn_autorange.setEnabled(enabled)
        self.btn_toggle_roi.setEnabled(enabled)
        self._update_volume_button_state()
        self._update_slice_label()

    def _update_slice_label(self):
        if not self._slice_axis:
            self.lbl_slice.setText("Slice: –")
            return
        coord_text = ""
        coords = self._axis_coords
        if coords is not None and 0 <= self._slice_index < coords.size:
            coord = coords[self._slice_index]
            coord_text = f" ({coord})"
        self.lbl_slice.setText(f"Slice: {self._slice_axis} = {self._slice_index}{coord_text}")

    def _update_volume_button_state(self):
        enabled = (
            gl is not None
            and self._current_da is not None
            and self._slice_axis is not None
            and self._row_axis is not None
            and self._col_axis is not None
            and self._slice_count > 0
        )
        self.btn_volume_view.setEnabled(enabled)

    # ---------- slice navigation ----------
    def _on_slice_changed(self, value: int):
        self._slice_index = int(value)
        block = self.spin_slice.blockSignals(True)
        self.spin_slice.setValue(self._slice_index)
        self.spin_slice.blockSignals(block)
        self._update_slice_label()
        self._update_slice_display()

    def _on_slice_spin_changed(self, value: int):
        self._slice_index = int(value)
        block = self.sld_slice.blockSignals(True)
        self.sld_slice.setValue(self._slice_index)
        self.sld_slice.blockSignals(block)
        self._update_slice_label()
        self._update_slice_display()

    def _gather_selection(self, slice_index: Optional[int] = None) -> Dict[str, int]:
        idx = dict(self._fixed_indices)
        axis = self._slice_axis
        if axis:
            idx[axis] = int(self._slice_index if slice_index is None else slice_index)
        return idx

    def _extract_slice(self, slice_index: Optional[int] = None) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        if self._current_da is None or self._row_axis is None or self._col_axis is None:
            return None, {}
        if self._slice_axis is None:
            return None, {}
        select = self._gather_selection(slice_index)
        try:
            slice_da = self._current_da.isel(select)
        except Exception:
            return None, {}
        for dim in (self._row_axis, self._col_axis):
            if dim not in slice_da.dims:
                return None, {}
        try:
            slice_da = slice_da.transpose(self._row_axis, self._col_axis)
        except Exception:
            return None, {}
        data = np.asarray(slice_da.values, float)
        coords = guess_phys_coords(slice_da)
        try:
            row_coord = slice_da.coords.get(self._row_axis)
            if row_coord is not None:
                values = np.asarray(row_coord.values)
                if values.ndim == 1:
                    coords["row_values"] = np.asarray(values, float)
                elif values.ndim >= 2:
                    coords["row_grid"] = np.asarray(values, float)
        except Exception:
            pass
        try:
            col_coord = slice_da.coords.get(self._col_axis)
            if col_coord is not None:
                values = np.asarray(col_coord.values)
                if values.ndim == 1:
                    coords["col_values"] = np.asarray(values, float)
                elif values.ndim >= 2:
                    coords["col_grid"] = np.asarray(values, float)
        except Exception:
            pass
        return data, coords

    def _apply_processing(self, data: np.ndarray) -> np.ndarray:
        mode = self._processing_mode or "none"
        params = dict(self._processing_params or {})
        processed = np.asarray(data, float)
        if mode.startswith("pipeline:"):
            if not self.processing_manager:
                raise RuntimeError("No processing manager is available for pipelines.")
            name = mode.split(":", 1)[1]
            pipeline = self.processing_manager.get_pipeline(name)
            if pipeline is None:
                raise RuntimeError(f"Pipeline '{name}' is not available.")
            processed = pipeline.apply(processed)
        elif mode != "none":
            processed = apply_processing_step(mode, processed, params)
        return np.asarray(processed, float)

    def _update_slice_display(self, *, autorange: bool = False):
        data, coords = self._extract_slice()
        if data is None:
            self._current_processed_slice = None
            self._cache_slice_coords({})
            self.viewer.set_image(np.zeros((1, 1)), autorange=True)
            self._roi_last_shape = None
            if self._roi_enabled:
                self._update_roi_curve()
            self._refresh_volume_window()
            return
        try:
            processed = self._apply_processing(data)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Processing failed", str(exc))
            self._processing_mode = "none"
            self._processing_params = {}
            processed = np.asarray(data, float)
        self._current_processed_slice = np.asarray(processed, float)
        self._cache_slice_coords(coords)
        shape = self._current_processed_slice.shape
        if self._roi_enabled and shape != self._roi_last_shape:
            self._reset_roi_to_image(shape)
        self._roi_last_shape = shape
        if "X" in coords and "Y" in coords:
            self.viewer.set_warped(coords["X"], coords["Y"], processed, autorange=autorange)
        elif "x" in coords and "y" in coords:
            self.viewer.set_rectilinear(coords["x"], coords["y"], processed, autorange=autorange)
        else:
            self.viewer.set_image(processed, autorange=autorange)
        self.cmb_colormap.setEnabled(True)
        self._apply_selected_colormap()
        if autorange:
            try:
                self.viewer.autoscale_levels()
            except Exception:
                pass
            try:
                self.viewer.auto_view_range()
            except Exception:
                pass
        self._update_slice_label()
        if self._roi_enabled:
            self._update_roi_slice_reference()
            self._update_roi_curve()
        self._refresh_volume_window()

    def _on_autoscale_clicked(self):
        self.viewer.autoscale_levels()

    def _on_autorange_clicked(self):
        self.viewer.auto_view_range()

    # ---------- processing ----------
    def _choose_processing(self):
        dialog = ProcessingSelectionDialog(self.processing_manager, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        mode, params = dialog.selected_processing()
        self._processing_mode = mode
        self._processing_params = dict(params)
        self._invalidate_volume_cache()
        self._update_slice_display()

    def _reset_processing(self):
        self._processing_mode = "none"
        self._processing_params = {}
        self._invalidate_volume_cache()
        self._update_slice_display(autorange=True)

    # ---------- volume viewer ----------
    def _invalidate_volume_cache(self):
        self._volume_cache = None

    def _ensure_volume_window(self) -> SequentialVolumeWindow:
        if gl is None:
            raise RuntimeError("pyqtgraph.opengl is not available")
        window = self._volume_window
        if window is None:
            window = SequentialVolumeWindow(self)
            window.closed.connect(self._on_volume_window_closed)
            self._volume_window = window
        return window

    def _on_volume_window_closed(self):
        self._volume_window = None

    def _open_volume_view(self):
        if gl is None:
            QtWidgets.QMessageBox.information(
                self,
                "Volume rendering unavailable",
                "3D volume rendering requires the optional pyqtgraph.opengl package.",
            )
            return
        volume = self._collect_volume_data()
        if volume is None:
            QtWidgets.QMessageBox.information(
                self,
                "Volume unavailable",
                "Unable to assemble a 3D volume with the current axis and index selection.",
            )
            return
        try:
            window = self._ensure_volume_window()
        except RuntimeError as exc:
            QtWidgets.QMessageBox.warning(self, "Volume rendering error", str(exc))
            return
        window.set_volume(volume)
        cmap_name = self.cmb_colormap.currentData() or "viridis"
        window.set_colormap(cmap_name)
        window.show()
        window.raise_()
        window.activateWindow()

    def _collect_volume_data(self) -> Optional[np.ndarray]:
        if self._volume_cache is not None:
            return self._volume_cache
        if (
            self._current_da is None
            or self._slice_axis is None
            or self._row_axis is None
            or self._col_axis is None
        ):
            return None
        for axis in (self._slice_axis, self._row_axis, self._col_axis):
            if axis not in self._current_da.dims:
                return None
        select: Dict[str, int] = {}
        for dim in self._current_da.dims:
            if dim in (self._slice_axis, self._row_axis, self._col_axis):
                continue
            select[dim] = int(self._fixed_indices.get(dim, 0))
        try:
            subset = self._current_da.isel(select)
        except Exception:
            return None
        try:
            subset = subset.transpose(self._slice_axis, self._row_axis, self._col_axis)
        except Exception:
            return None
        data = np.asarray(subset.values, float)
        if data.ndim != 3:
            return None
        frames: List[np.ndarray] = []
        for idx in range(data.shape[0]):
            frame = np.asarray(data[idx], float)
            try:
                processed = self._apply_processing(frame)
            except Exception:
                processed = frame
            frames.append(np.asarray(processed, float))
        try:
            volume = np.stack(frames, axis=0)
        except Exception:
            return None
        self._volume_cache = volume
        return volume

    def _refresh_volume_window(self):
        if self._volume_window is None or not self._volume_window.isVisible():
            return
        volume = self._collect_volume_data()
        if volume is None:
            self._volume_window.clear_volume()
            return
        self._volume_window.set_volume(volume)
        cmap_name = self.cmb_colormap.currentData() or "viridis"
        self._volume_window.set_colormap(cmap_name)

    def _update_volume_window_colormap(self):
        if self._volume_window is None or not self._volume_window.isVisible():
            return
        cmap_name = self.cmb_colormap.currentData() or "viridis"
        self._volume_window.set_colormap(cmap_name)

    # ---------- ROI controls ----------
    def _ensure_roi_window(self) -> SequentialRoiWindow:
        window = self._roi_window
        if window is None:
            window = SequentialRoiWindow(self)
            window.axesChanged.connect(self._on_roi_axes_changed_from_window)
            window.reducerChanged.connect(self._set_roi_method)
            window.closed.connect(self._on_roi_window_closed)
            self._roi_window = window
        self._update_roi_window_options()
        return window

    def _on_roi_window_closed(self):
        if self._roi_enabled:
            self.btn_toggle_roi.blockSignals(True)
            self.btn_toggle_roi.setChecked(False)
            self.btn_toggle_roi.blockSignals(False)
            self._on_roi_toggled(False)

    def _update_roi_axis_options(self):
        options: List[Tuple[str, Tuple[int, ...], str, str, Optional[int]]] = []
        if self._row_axis and self._col_axis:
            row_label = self._axis_display_name(self._row_axis)
            col_label = self._axis_display_name(self._col_axis)
            slice_label = self._axis_display_name(self._slice_axis)
            options.append(
                (
                    f"Reduce {row_label} & {col_label} → curve along {slice_label}",
                    (0, 1),
                    f"{row_label} & {col_label}",
                    slice_label,
                    None,
                )
            )
            options.append(
                (
                    f"Reduce {row_label} → profile across {col_label}",
                    (0,),
                    row_label,
                    col_label,
                    1,
                )
            )
            options.append(
                (
                    f"Reduce {col_label} → profile across {row_label}",
                    (1,),
                    col_label,
                    row_label,
                    0,
                )
            )
        self._roi_axis_options = options
        if not options:
            self._roi_axis_index = 0
            self._roi_axes_selection = (0, 1)
        else:
            self._roi_axis_index = max(0, min(self._roi_axis_index, len(options) - 1))
            self._roi_axes_selection = tuple(options[self._roi_axis_index][1])
        self._update_roi_window_options()

    def _on_roi_axes_changed_from_window(self, axes: Tuple[int, ...]):
        axes = tuple(int(a) for a in axes) if axes else (0, 1)
        matched = False
        for idx, option in enumerate(self._roi_axis_options):
            if tuple(option[1]) == axes:
                self._roi_axis_index = idx
                matched = True
                break
        if not matched and self._roi_axis_options:
            self._roi_axis_index = 0
            axes = tuple(self._roi_axis_options[0][1])
        self._roi_axes_selection = axes if axes else (0, 1)
        self._update_roi_window_hint()
        if self._roi_enabled:
            self._update_roi_curve()

    def _current_roi_option(self) -> Optional[Tuple[str, Tuple[int, ...], str, str, Optional[int]]]:
        if not self._roi_axis_options:
            return None
        idx = max(0, min(self._roi_axis_index, len(self._roi_axis_options) - 1))
        return self._roi_axis_options[idx]

    def _update_roi_window_options(self):
        if self._roi_window is None:
            return
        self._roi_window.set_axis_options(self._roi_axis_options, self._roi_axis_index)
        self._roi_window.set_reducer_options(self._roi_reducers, self._roi_method_key)
        self._update_roi_window_hint()

    def _update_roi_window_hint(self):
        if self._roi_window is None:
            return
        option = self._current_roi_option()
        if option is None:
            self._roi_window.set_hint("")
            return
        axes = option[1] if len(option) > 1 else ()
        collapsed = option[2] if len(option) > 2 else "region"
        remaining = option[3] if len(option) > 3 else ""
        axes = tuple(int(a) for a in axes)
        if set(axes) == {0, 1}:
            slice_label = remaining or self._axis_display_name(self._slice_axis)
            hint = f"Collapsing {collapsed} to track statistics along {slice_label}."
        elif axes == (0,):
            target = remaining or self._axis_display_name(self._col_axis)
            hint = f"Collapsing {collapsed} to profile across {target} for the active slice."
        elif axes == (1,):
            target = remaining or self._axis_display_name(self._row_axis)
            hint = f"Collapsing {collapsed} to profile across {target} for the active slice."
        else:
            hint = ""
        self._roi_window.set_hint(hint)

    def _current_roi_array(self) -> Optional[np.ndarray]:
        if self._current_processed_slice is None:
            return None
        return self._roi_extract_region(self._current_processed_slice)

    def _on_roi_toggled(self, checked: bool):
        checked = bool(checked)
        view = getattr(self.viewer, "plot", None)
        if checked and self._current_processed_slice is not None:
            if view is not None and self.roi.scene() is None:
                view.addItem(self.roi)
            self.roi.show()
            self._roi_enabled = True
            self.lbl_roi_status.setText(self._describe_roi())
            self._reset_roi_to_image(self._current_processed_slice.shape)
            self._update_roi_slice_reference()
            self._update_roi_axis_options()
            window = self._ensure_roi_window()
            window.show()
            window.raise_()
            self._update_roi_curve()
        else:
            if view is not None and self.roi.scene() is not None:
                try:
                    view.removeItem(self.roi)
                except Exception:
                    pass
            self.roi.hide()
            self._roi_enabled = False
            self._roi_last_slices = None
            self._roi_last_bounds = None
            self._roi_last_shape = None
            if self._roi_window is not None:
                try:
                    self._roi_window.hide()
                    self._roi_window.update_slice_curve([], [], "Slice coordinate", "ROI statistic")
                    self._roi_window.update_profile([], [], "", "", False)
                except Exception:
                    pass
            self.lbl_roi_status.setText("ROI disabled")

    def _reset_roi_to_image(self, shape: Optional[Tuple[int, int]] = None):
        if not self._roi_enabled:
            return
        self._roi_last_bounds = None
        if shape is None:
            if self._current_processed_slice is None:
                return
            shape = self._current_processed_slice.shape
        if not shape or len(shape) < 2:
            return
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            return
        rect_w = max(2, width // 2)
        rect_h = max(2, height // 2)
        pos_x = max(0, (width - rect_w) // 2)
        pos_y = max(0, (height - rect_h) // 2)
        try:
            self.roi.blockSignals(True)
            self.roi.setPos((pos_x, pos_y))
            self.roi.setSize((rect_w, rect_h))
        finally:
            try:
                self.roi.blockSignals(False)
            except Exception:
                pass
        self._update_roi_slice_reference()

    def _on_roi_region_changed(self, *_args):
        if not self._roi_enabled:
            return
        self._update_roi_slice_reference()
        self._update_roi_curve()

    def _update_roi_slice_reference(self):
        if not self._roi_enabled or self._current_processed_slice is None:
            self._roi_last_slices = None
            self._roi_last_bounds = None
            return
        img_item = getattr(self.viewer, "img_item", None)
        if img_item is None:
            self._roi_last_slices = None
            self._roi_last_bounds = None
            return
        slices = None
        try:
            try:
                _, slc = self.roi.getArraySlice(
                    self._current_processed_slice,
                    img_item,
                    returnSlice=True,
                )
            except TypeError:
                _, slc = self.roi.getArraySlice(self._current_processed_slice, img_item)
            if isinstance(slc, tuple):
                slices = slc
        except Exception:
            slices = None

        if (
            isinstance(slices, tuple)
            and len(slices) >= 2
            and all(isinstance(s, slice) for s in slices[:2])
        ):
            sy, sx = slices[0], slices[1]
            self._roi_last_slices = (sy, sx)
            self._roi_last_bounds = self._normalize_roi_bounds(sy, sx)
        else:
            self._roi_last_slices = None
            self._roi_last_bounds = self._roi_bounds_from_geometry()

    def _normalize_roi_bounds(self, sy: slice, sx: slice) -> Optional[Tuple[int, int, int, int]]:
        if self._current_processed_slice is None:
            return None
        height, width = self._current_processed_slice.shape[:2]

        def _bounds(sl: slice, limit: int) -> Tuple[int, int]:
            start = float(sl.start) if sl.start is not None else 0.0
            stop = float(sl.stop) if sl.stop is not None else float(limit)
            step = sl.step
            if step is not None and step < 0:
                start, stop = stop, start
            a = int(np.floor(start))
            b = int(np.ceil(stop))
            a = max(0, min(limit, a))
            b = max(a, min(limit, b))
            return a, b

        y0, y1 = _bounds(sy, height)
        x0, x1 = _bounds(sx, width)
        if y1 <= y0 or x1 <= x0:
            return None
        return (y0, y1, x0, x1)

    def _roi_bounds_from_geometry(self) -> Optional[Tuple[int, int, int, int]]:
        if self._current_processed_slice is None:
            return None
        img_item = getattr(self.viewer, "img_item", None)
        if img_item is None:
            return None
        try:
            rect = self.roi.boundingRect()
            top_left_scene = self.roi.mapToScene(rect.topLeft())
            bottom_right_scene = self.roi.mapToScene(rect.bottomRight())
            top_left_item = img_item.mapFromScene(top_left_scene)
            bottom_right_item = img_item.mapFromScene(bottom_right_scene)
        except Exception:
            return None

        xs = [float(top_left_item.x()), float(bottom_right_item.x())]
        ys = [float(top_left_item.y()), float(bottom_right_item.y())]
        x0 = int(np.floor(min(xs)))
        x1 = int(np.ceil(max(xs)))
        y0 = int(np.floor(min(ys)))
        y1 = int(np.ceil(max(ys)))
        height, width = self._current_processed_slice.shape[:2]
        x0 = max(0, min(width, x0))
        x1 = max(x0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(y0, min(height, y1))
        if y1 <= y0 or x1 <= x0:
            return None
        return (y0, y1, x0, x1)

    def _roi_extract_region(self, data: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(data, float)
        bounds = self._roi_last_bounds
        if bounds is None:
            bounds = self._roi_bounds_from_geometry()
            if bounds is None:
                return None
            self._roi_last_bounds = bounds
        if arr.ndim < 2:
            return arr
        height, width = arr.shape[:2]
        y0, y1, x0, x1 = bounds
        y0 = max(0, min(height, y0))
        y1 = max(y0, min(height, y1))
        x0 = max(0, min(width, x0))
        x1 = max(x0, min(width, x1))
        if y1 <= y0 or x1 <= x0:
            return None
        region = arr[y0:y1, x0:x1]
        if region.size == 0:
            return None
        return np.asarray(region, float)

    def _compute_roi_value(self, data: np.ndarray) -> float:
        reducer_entry = self._roi_reducers.get(self._roi_method_key)
        if reducer_entry is None:
            return float("nan")
        _, reducer = reducer_entry
        roi_data = self._roi_extract_region(data)
        if roi_data is None:
            roi_data = np.asarray(data, float)
        axes = tuple(int(a) for a in self._roi_axes_selection) or (0, 1)
        axes = tuple(sorted(set(axes)))
        with np.errstate(all="ignore"):
            if not axes:
                result = reducer(roi_data, axis=None)
            else:
                axis_param = axes[0] if len(axes) == 1 else axes
                result = reducer(roi_data, axis=axis_param)
            while isinstance(result, np.ndarray) and result.ndim > 0:
                if result.ndim == 1:
                    result = reducer(result, axis=0)
                else:
                    result = reducer(result, axis=tuple(range(result.ndim)))
        try:
            return float(np.asarray(result).item())
        except Exception:
            try:
                return float(result)
            except Exception:
                return float("nan")

    def _update_roi_curve(self):
        if self._roi_window is not None and not self._roi_window.isVisible():
            self._roi_window.update_slice_curve([], [], "Slice coordinate", "ROI statistic")
            self._roi_window.update_profile([], [], "", "", False)
        if not self._roi_enabled or self._current_da is None:
            if self._roi_window is not None:
                self._roi_window.update_slice_curve([], [], "Slice coordinate", "ROI statistic")
                self._roi_window.update_profile([], [], "", "", False)
            return
        count = max(0, self._slice_count)
        if count == 0:
            if self._roi_window is not None:
                self._roi_window.update_slice_curve([], [], "Slice coordinate", "ROI statistic")
                self._roi_window.update_profile([], [], "", "", False)
            return
        self._update_roi_slice_reference()
        values: List[float] = []
        xs: List[float] = []
        coords = self._axis_coords
        for idx in range(count):
            data, _ = self._extract_slice(slice_index=idx)
            if data is None:
                values.append(np.nan)
            else:
                try:
                    processed = self._apply_processing(data)
                except Exception:
                    processed = np.asarray(data, float)
                values.append(self._compute_roi_value(processed))
            if coords is not None and idx < coords.size:
                xs.append(float(coords[idx]))
            else:
                xs.append(float(idx))
        name = self._roi_reducers.get(self._roi_method_key, ("ROI statistic",))[0]
        axis_label = self._axis_display_name(self._slice_axis)
        if self._roi_window is not None:
            self._roi_window.update_slice_curve(xs, values, axis_label, name)
            self._update_roi_profile_plot()
        self.lbl_roi_status.setText(self._describe_roi())

    def _update_roi_profile_plot(self):
        if self._roi_window is None or not self._roi_enabled:
            return
        option = self._current_roi_option()
        if option is None:
            self._roi_window.update_profile([], [], "", "", False)
            return
        axes = option[1] if len(option) > 1 else ()
        remaining = option[3] if len(option) > 3 else ""
        profile_axis = option[4] if len(option) > 4 else None
        axes = tuple(int(a) for a in axes)
        if set(axes) == {0, 1} or profile_axis is None:
            self._roi_window.update_profile([], [], "", "", False)
            return
        data = self._current_roi_array()
        reducer_entry = self._roi_reducers.get(self._roi_method_key)
        if data is None or reducer_entry is None:
            self._roi_window.update_profile([], [], "", "", False)
            return
        _, reducer = reducer_entry
        axis = axes[0] if axes else 0
        with np.errstate(all="ignore"):
            profile = reducer(data, axis=axis)
        try:
            prof_arr = np.asarray(profile, float)
        except Exception:
            self._roi_window.update_profile([], [], "", "", False)
            return
        if prof_arr.ndim > 1:
            prof_arr = np.asarray(prof_arr).ravel()
        xs = list(range(int(prof_arr.size)))
        ys = [float(val) if np.isfinite(val) else np.nan for val in prof_arr]
        coords = self._roi_profile_coordinates(int(profile_axis), len(xs))
        if len(coords) == len(xs):
            xs = coords
        xlabel = remaining or (
            self._axis_display_name(self._col_axis) if profile_axis == 1 else self._axis_display_name(self._row_axis)
        )
        ylabel = self._roi_reducers.get(self._roi_method_key, ("Value",))[0]
        self._roi_window.update_profile(xs, ys, xlabel, ylabel, True)

    def _describe_roi(self) -> str:
        name = self._roi_reducers.get(self._roi_method_key, ("statistic",))[0]
        option = self._current_roi_option()
        collapsed = option[2] if option else "region"
        axis = option[3] if option and len(option) > 3 else self._axis_display_name(self._slice_axis)
        axis = axis or self._axis_display_name(self._slice_axis)
        return f"ROI {name.lower()} of {collapsed} across {axis}"

    def _set_roi_method(self, key: str):
        if key not in self._roi_reducers or key == self._roi_method_key:
            return
        self._roi_method_key = key
        self._update_roi_window_options()
        if self._roi_enabled:
            self._update_roi_curve()
        self.lbl_roi_status.setText(self._describe_roi())

# ---------------------------------------------------------------------------
# Overlay view: stack multiple layers with per-layer controls
# ---------------------------------------------------------------------------
class OverlayLayer(QtCore.QObject):
    def __init__(self, view: "OverlayView", title: str, data: np.ndarray, rect: QtCore.QRectF | None):
        super().__init__(view)
        self.view = view
        self.title = title
        self.base_data = np.asarray(data, float)
        self.processed_data = np.array(self.base_data, copy=True)
        self.rect = rect
        self.image_item = pg.ImageItem()
        self.image_item.setImage(self.processed_data, autoLevels=False)
        if rect is not None:
            try:
                self.image_item.setRect(rect)
            except Exception:
                pass
        self.image_item.setOpacity(1.0)
        self.image_item.setVisible(True)
        self._levels = self._compute_levels(self.processed_data)
        try:
            self.image_item.setLevels(self._levels)
        except Exception:
            pass
        self.colormap_name = "viridis"
        self.visible = True
        self.opacity = 1.0
        self.current_processing = "none"
        self.processing_params: dict = {}
        self.widget: Optional["OverlayLayerWidget"] = None
        self.set_colormap(self.colormap_name)

    # ---------- helpers ----------
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
            self.image_item.setVisible(self.visible)
        except Exception:
            pass

    def set_opacity(self, alpha: float):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self.opacity = alpha
        try:
            self.image_item.setOpacity(alpha)
        except Exception:
            pass
        if self.widget:
            self.widget.update_opacity_label(alpha)

    def set_colormap(self, name: str):
        self.colormap_name = name or "viridis"
        try:
            cmap = pg.colormap.get(self.colormap_name)
            if hasattr(cmap, "getLookupTable"):
                lut = cmap.getLookupTable(0.0, 1.0, 256)
                self.image_item.setLookupTable(lut)
        except Exception:
            pass

    def set_levels(self, lo: float, hi: float, *, update_widget: bool = True):
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        if hi <= lo:
            hi = lo + max(abs(lo) * 1e-6, 1e-6)
        self._levels = (float(lo), float(hi))
        try:
            self.image_item.setLevels(self._levels)
        except Exception:
            pass
        if update_widget and self.widget:
            self.widget.update_level_spins(self._levels[0], self._levels[1])

    def auto_levels(self):
        lo, hi = self._compute_levels(self.processed_data)
        self.set_levels(lo, hi, update_widget=True)

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
        self.processed_data = np.asarray(data, float)
        try:
            self.image_item.setImage(self.processed_data, autoLevels=False)
        except Exception:
            pass
        self.auto_levels()

    def get_display_rect(self):
        rect = self.rect
        if rect is None:
            try:
                rect = self.image_item.mapRectToParent(self.image_item.boundingRect())
            except Exception:
                rect = None
        return rect


class OverlayLayerWidget(QtWidgets.QGroupBox):
    def __init__(self, view: "OverlayView", layer: OverlayLayer):
        super().__init__(layer.title)
        self.view = view
        self.layer = layer
        self._ready = False

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(6)

        # Visibility / remove
        header = QtWidgets.QHBoxLayout()
        self.chk_visible = QtWidgets.QCheckBox("Visible")
        self.chk_visible.setChecked(True)
        self.chk_visible.toggled.connect(self._on_visibility)
        header.addWidget(self.chk_visible)
        header.addStretch(1)
        self.btn_remove = QtWidgets.QToolButton()
        self.btn_remove.setText("✕")
        self.btn_remove.setToolTip("Remove layer")
        self.btn_remove.clicked.connect(self._on_remove)
        header.addWidget(self.btn_remove)
        lay.addLayout(header)

        # Colormap selection
        cmap_row = QtWidgets.QHBoxLayout()
        cmap_row.addWidget(QtWidgets.QLabel("Colormap:"))
        self.cmb_colormap = QtWidgets.QComboBox()
        try:
            cmaps = sorted(pg.colormap.listMaps())
        except Exception:
            cmaps = ["viridis", "plasma", "magma", "cividis", "gray"]
        for name in cmaps:
            self.cmb_colormap.addItem(name)
        self.cmb_colormap.currentTextChanged.connect(self._on_colormap)
        cmap_row.addWidget(self.cmb_colormap, 1)
        lay.addLayout(cmap_row)

        # Levels controls
        lvl_row = QtWidgets.QHBoxLayout()
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
        lay.addLayout(lvl_row)

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
        proc_layout = QtWidgets.QVBoxLayout(proc_box)
        proc_layout.setContentsMargins(6, 6, 6, 6)
        proc_layout.setSpacing(4)

        proc_row = QtWidgets.QHBoxLayout()
        proc_row.addWidget(QtWidgets.QLabel("Operation:"))
        self.cmb_processing = QtWidgets.QComboBox()
        self.cmb_processing.currentIndexChanged.connect(self._on_processing_mode_changed)
        proc_row.addWidget(self.cmb_processing, 1)
        proc_layout.addLayout(proc_row)

        self.param_stack = QtWidgets.QStackedWidget()
        self.param_stack.addWidget(QtWidgets.QWidget())  # None placeholder
        self._function_forms: Dict[str, ParameterForm] = {}
        self._function_indices: Dict[str, int] = {}
        for spec in list_processing_functions():
            form = ParameterForm(spec.parameters)
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
        self._set_colormap_selection(self.layer.colormap_name)
        lo, hi = getattr(self.layer, "_levels", (0.0, 1.0))
        self.update_level_spins(lo, hi)
        self.update_opacity_label(self.layer.opacity)
        self.sld_opacity.setValue(int(round(self.layer.opacity * 100)))
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
        idx = self.cmb_colormap.findText(name, QtCore.Qt.MatchFixedString)
        if idx < 0:
            idx = self.cmb_colormap.findText("viridis", QtCore.Qt.MatchFixedString)
        if idx >= 0:
            block = self.cmb_colormap.blockSignals(True)
            self.cmb_colormap.setCurrentIndex(idx)
            self.cmb_colormap.blockSignals(block)

    def update_level_spins(self, lo: float, hi: float):
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

    def _refresh_processing_options(self):
        current = self._current_processing_data()
        block = self.cmb_processing.blockSignals(True)
        self.cmb_processing.clear()
        self.cmb_processing.addItem("None", {"type": "none"})
        for spec in list_processing_functions():
            self.cmb_processing.addItem(spec.label, {"type": "function", "key": spec.key})
        if self.manager:
            for name in self.manager.pipeline_names():
                self.cmb_processing.addItem(f"Pipeline: {name}", {"type": "pipeline", "name": name})
        idx = self._find_data_index(current)
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
        if not self._ready:
            return
        self.layer.set_colormap(name)

    def _on_levels_changed(self):
        if not self._ready:
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
        self.layer.auto_levels()

    def _on_opacity(self, value: int):
        alpha = float(value) / 100.0
        self.layer.set_opacity(alpha)

    def _on_processing_mode_changed(self):
        data = self._current_processing_data()
        if data.get("type") == "function":
            idx = self._function_indices.get(data.get("key", ""), 0)
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
                self.param_stack.setCurrentIndex(0)
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
    def __init__(self, processing_manager: Optional[ProcessingManager] = None, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.layers: List[OverlayLayer] = []
        self.processing_manager: Optional[ProcessingManager] = processing_manager

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        toolbar = QtWidgets.QHBoxLayout()
        self.btn_auto_view = QtWidgets.QPushButton("Auto view")
        self.btn_auto_view.clicked.connect(self.auto_view_range)
        toolbar.addWidget(self.btn_auto_view)
        self.btn_clear = QtWidgets.QPushButton("Clear layers")
        self.btn_clear.clicked.connect(self.clear_layers)
        toolbar.addWidget(self.btn_clear)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # Layer controls panel
        panel = QtWidgets.QWidget()
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
        self.plot.invertY(True)
        self.plot.showGrid(x=False, y=False)
        self.plot.setLabel("left", "Y")
        self.plot.setLabel("bottom", "X")
        splitter.addWidget(self.glw)
        splitter.setStretchFactor(1, 1)

    # ---------- drag & drop ----------
    def dragEnterEvent(self, ev):
        ev.acceptProposedAction() if ev.mimeData().hasText() else ev.ignore()

    def dropEvent(self, ev):
        vr = VarRef.from_mime(ev.mimeData().text())
        if not vr:
            ev.ignore()
            return
        try:
            da, coords = vr.load()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(e))
            return
        data, rect = self._prepare_image(da, coords)
        layer = OverlayLayer(self, f"{vr.path.name}:{vr.var}", data, rect)
        self.plot.addItem(layer.image_item)
        widget = OverlayLayerWidget(self, layer)
        widget.set_processing_manager(self.processing_manager)
        layer.set_widget(widget)
        self.layers.append(layer)
        self._insert_layer_widget(widget)
        self._update_hint()
        self.auto_view_range()
        ev.acceptProposedAction()

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
            self.plot.removeItem(layer.image_item)
        except Exception:
            pass
        if layer.widget:
            w = layer.widget
            layer.widget = None
            w.setParent(None)
            w.deleteLater()
        self._update_hint()
        self.auto_view_range()

    def clear_layers(self):
        for layer in list(self.layers):
            self.remove_layer(layer)

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

    # ---------- data prep ----------
    def _prepare_image(self, da, coords):
        Z = np.asarray(da.values, float)
        if "X" in coords and "Y" in coords:
            return self._resample_warped(coords["X"], coords["Y"], Z)
        if "x" in coords and "y" in coords:
            return self._resample_rectilinear(coords["x"], coords["y"], Z)
        Ny, Nx = Z.shape
        rect = self._rect_to_qrectf(0.0, float(Nx), 0.0, float(Ny))
        return np.asarray(Z, float), rect

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


# ---------------------------------------------------------------------------
# Main window with tabs
# ---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Multi-Viewer")

        main = QtWidgets.QSplitter()
        self.setCentralWidget(main)

        self.processing_manager = ProcessingManager()

        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_splitter.setChildrenCollapsible(False)
        left_splitter.setHandleWidth(8)
        self.datasets = DatasetsPane()
        left_splitter.addWidget(self.datasets)
        self.processing_dock = ProcessingDockWidget(self.processing_manager)
        self.processing_panel = ProcessingDockContainer("Processing Pipelines", self.processing_dock)
        left_splitter.addWidget(self.processing_panel)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 1)
        main.addWidget(left_splitter)

        QtCore.QTimer.singleShot(0, lambda: left_splitter.setSizes([1, 1]))

        self.tabs = QtWidgets.QTabWidget()
        main.addWidget(self.tabs)
        main.setStretchFactor(1, 1)

        self.tab_multiview = MultiViewGrid(self.processing_manager)
        self.tabs.addTab(self.tab_multiview, "MultiView")
        self.tab_sequential = SequentialView(self.processing_manager)
        self.tabs.addTab(self.tab_sequential, "Sequential View")
        self.tab_overlay = OverlayView(self.processing_manager)
        self.tabs.addTab(self.tab_overlay, "Overlay")
        self.tab_overlay.set_processing_manager(self.processing_manager)

        self.resize(1500, 900)


def main():
    app = QtWidgets.QApplication([])
    pg.setConfigOptions(imageAxisOrder='row-major')
    win = MainWindow(); win.show()
    app.exec_()


if __name__ == "__main__":
    main()
