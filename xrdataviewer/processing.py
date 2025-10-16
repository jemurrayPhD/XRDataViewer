"""Processing tools for XRDataViewer.

This module contains the dialogs and helper classes that allow end-users to
build, preview, and persist data processing pipelines.  It also centralises the
logic for rendering previews, so the UI components can focus on presentation.

The module historically grew organically which lead to duplicated widget
initialisation code and implicit behaviour.  The refactor performed here
deduplicates the parameter form handling logic and documents the intent of the
key building blocks to make future maintenance predictable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import json

import numpy as np
import pyqtgraph as pg
from PySide2 import QtCore, QtGui, QtWidgets

from app_logging import log_action
from data_processing import (
    ParameterDefinition,
    ProcessingPipeline,
    ProcessingStep,
    apply_processing_step,
    get_processing_function,
    list_processing_functions,
    summarize_parameters,
)
from xr_plot_widget import (
    CentralPlotWidget,
    PlotAnnotationConfig,
    ScientificAxisItem,
    apply_plotitem_annotation,
    plotitem_annotation_state,
)

from .annotations import PlotAnnotationDialog
from .colormaps import available_colormap_names, get_colormap, is_scientific_colormap
from .utils import _nan_aware_reducer
from .utils import open_dataset


@dataclass(frozen=True)
class _WidgetBinding:
    """Container tying a widget to its value accessors."""

    widget: QtWidgets.QWidget
    reader: Callable[[QtWidgets.QWidget], object]
    writer: Callable[[QtWidgets.QWidget, object], None]


def _connect_signal(signal: Any, callback: Callable[[], None]):
    """Connect *signal* to *callback* while tolerating signature mismatches."""

    try:
        signal.connect(lambda *_args, **_kwargs: callback())
    except Exception:
        # Some custom widgets expose read-only signals.  Failing silently keeps
        # the default value intact without breaking the form.
        pass


class ParameterForm(QtWidgets.QWidget):
    """Widget that exposes processing parameter definitions as form controls.

    The form is responsible for building the appropriate Qt widget for each
    :class:`~data_processing.ParameterDefinition` and providing a uniform API to
    read or write their values.  Historically the widget construction logic was
    duplicated across the class; the refactor uses :class:`_WidgetBinding`
    instances to centralise the behaviour and make the intent explicit.
    """

    parametersChanged = QtCore.Signal()

    def __init__(self, parameters: Iterable[ParameterDefinition], parent=None):
        super().__init__(parent)
        self._definitions = list(parameters)
        self._bindings: Dict[str, _WidgetBinding] = {}

        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        if not self._definitions:
            lbl = QtWidgets.QLabel("No parameters")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            layout.addRow(lbl)
            return

        for definition in self._definitions:
            binding = self._create_binding(definition)
            self._bindings[definition.name] = binding
            layout.addRow(definition.label, binding.widget)

    def values(self) -> Dict[str, object]:
        """Return the current parameter values keyed by definition name."""

        values: Dict[str, object] = {}
        for definition in self._definitions:
            binding = self._bindings.get(definition.name)
            if binding is None:
                continue
            values[definition.name] = binding.reader(binding.widget)
        return values

    def set_values(self, params: Dict[str, object]):
        """Apply *params* to the form without emitting change signals."""

        for definition in self._definitions:
            binding = self._bindings.get(definition.name)
            if binding is None:
                continue
            value = params.get(definition.name, definition.default)
            widget = binding.widget
            block = widget.blockSignals(True)
            try:
                binding.writer(widget, value)
            finally:
                widget.blockSignals(block)

    # ----- private helpers -------------------------------------------------

    def _create_binding(self, definition: ParameterDefinition) -> _WidgetBinding:
        """Create a :class:`_WidgetBinding` for *definition*."""

        kind = (definition.kind or "").lower()
        if kind == "float":
            return self._build_double_spinbox(definition)
        if kind == "int":
            return self._build_int_spinbox(definition)
        if kind == "enum":
            return self._build_enum_combobox(definition)
        return self._build_line_edit(definition)

    def _build_double_spinbox(self, definition: ParameterDefinition) -> _WidgetBinding:
        """Return a binding configured for floating-point parameters."""

        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(6)
        lo = float(definition.minimum) if definition.minimum is not None else -1e9
        hi = float(definition.maximum) if definition.maximum is not None else 1e9
        spin.setRange(lo, hi)
        if definition.step is not None:
            spin.setSingleStep(float(definition.step))
        spin.setValue(float(definition.default))
        _connect_signal(spin.valueChanged, self.parametersChanged.emit)
        return _WidgetBinding(
            widget=spin,
            reader=lambda w: float(cast(QtWidgets.QDoubleSpinBox, w).value()),
            writer=lambda w, value: cast(QtWidgets.QDoubleSpinBox, w).setValue(float(value)),
        )

    def _build_int_spinbox(self, definition: ParameterDefinition) -> _WidgetBinding:
        """Return a binding configured for integer parameters."""

        spin = QtWidgets.QSpinBox()
        lo = int(definition.minimum) if definition.minimum is not None else -1_000_000
        hi = int(definition.maximum) if definition.maximum is not None else 1_000_000
        spin.setRange(lo, hi)
        if definition.step is not None:
            spin.setSingleStep(int(definition.step))
        spin.setValue(int(definition.default))
        _connect_signal(spin.valueChanged, self.parametersChanged.emit)
        return _WidgetBinding(
            widget=spin,
            reader=lambda w: int(cast(QtWidgets.QSpinBox, w).value()),
            writer=lambda w, value: cast(QtWidgets.QSpinBox, w).setValue(int(value)),
        )

    def _build_enum_combobox(self, definition: ParameterDefinition) -> _WidgetBinding:
        """Return a binding configured for enumerated options."""

        combo = QtWidgets.QComboBox()
        if definition.choices:
            for label, value in definition.choices:
                combo.addItem(label, value)
        combo.setCurrentIndex(max(combo.findData(definition.default), 0))
        _connect_signal(combo.currentIndexChanged, self.parametersChanged.emit)
        return _WidgetBinding(
            widget=combo,
            reader=self._read_combobox,
            writer=lambda w, value: self._write_combobox(
                cast(QtWidgets.QComboBox, w), value
            ),
        )

    def _build_line_edit(self, definition: ParameterDefinition) -> _WidgetBinding:
        """Return a binding configured for free-form text values."""

        line = QtWidgets.QLineEdit(str(definition.default))
        _connect_signal(line.textChanged, self.parametersChanged.emit)
        return _WidgetBinding(
            widget=line,
            reader=lambda w: cast(QtWidgets.QLineEdit, w).text(),
            writer=lambda w, value: cast(QtWidgets.QLineEdit, w).setText(str(value)),
        )

    @staticmethod
    def _read_combobox(widget: QtWidgets.QWidget) -> object:
        """Return the active value of *widget* taking custom data into account."""

        combo = cast(QtWidgets.QComboBox, widget)
        data = combo.currentData()
        return data if data is not None else combo.currentText()

    @staticmethod
    def _write_combobox(widget: QtWidgets.QComboBox, value: object) -> None:
        """Update *widget* to select *value* by data first, then text."""

        idx = widget.findData(value)
        if idx < 0:
            idx = widget.findText(str(value))
        widget.setCurrentIndex(max(idx, 0))

class ProcessingManager(QtCore.QObject):
    """In-memory registry for named :class:`ProcessingPipeline` objects."""

    pipelines_changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self._pipelines: Dict[str, ProcessingPipeline] = {}

    def list_pipelines(self) -> List[ProcessingPipeline]:
        """Return all known pipelines as defensive copies."""

        return [self._clone_pipeline(p) for p in self._pipelines.values()]

    def pipeline_names(self) -> List[str]:
        """Return the pipeline names sorted alphabetically."""

        return sorted(self._pipelines.keys())

    def get_pipeline(self, name: str) -> Optional[ProcessingPipeline]:
        """Return a copy of the pipeline identified by *name* if present."""

        if name not in self._pipelines:
            return None
        return self._clone_pipeline(self._pipelines[name])

    def save_pipeline(self, pipeline: ProcessingPipeline):
        """Persist *pipeline* (overwriting any existing entry with the same name)."""

        name = pipeline.name.strip()
        if not name:
            raise ValueError("Pipeline name cannot be empty")
        self._pipelines[name] = self._clone_pipeline(pipeline)
        self.pipelines_changed.emit()

    def delete_pipeline(self, name: str):
        """Remove the pipeline identified by *name* if it exists."""

        if name in self._pipelines:
            del self._pipelines[name]
            self.pipelines_changed.emit()

    def _clone_pipeline(self, pipeline: ProcessingPipeline) -> ProcessingPipeline:
        """Create a deep-ish copy of *pipeline* suitable for external use."""

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
        for name in available_colormap_names():
            label = name.replace("_", " ").title()
            if is_scientific_colormap(name):
                label = f"{label} (Scientific)"
            self.cmb_colormap.addItem(label, name)
        if self.cmb_colormap.count() == 0:
            self.cmb_colormap.addItem("Default", "default")
        self.cmb_colormap.currentIndexChanged.connect(self._on_colormap_changed)
        top_row.addWidget(self.cmb_colormap, 0)
        layout.addLayout(top_row)

        self.image_view = pg.ImageView()
        try:
            self.image_view.getView().invertY(False)
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

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Close | QtWidgets.QDialogButtonBox.Apply
        )
        buttons.rejected.connect(self.reject)
        self._apply_button = buttons.button(QtWidgets.QDialogButtonBox.Apply)
        if self._apply_button is not None:
            self._apply_button.clicked.connect(self._on_apply_clicked)
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

        dims = self._current_dims()
        ndim = None if dims is None else len(dims)
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
                parameters = spec.parameters_for_data(dims) if spec.supports_ndim(ndim) else spec.parameters_for_data(None)
                form = ParameterForm(parameters)
                form.set_values(step.params)
                form.parametersChanged.connect(self._on_parameters_changed)
                vbox.addWidget(form)
                self._forms.append((step, form))
            self.steps_layout.addWidget(box)
        self.steps_layout.addStretch(1)
        self._apply_pipeline()

    def _current_dims(self) -> Optional[Tuple[str, ...]]:
        if self._raw_data is None:
            return None
        ndim = int(np.ndim(self._raw_data))
        return tuple(f"axis{i}" for i in range(ndim))

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

    def _on_apply_clicked(self):
        self._update_steps_from_forms()
        self._apply_pipeline()

    def _apply_selected_colormap(self):
        if not hasattr(self, "cmb_colormap"):
            return
        name = self.cmb_colormap.currentData()
        if not name or name == "default":
            return
        cmap = get_colormap(str(name))
        if cmap is None:
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

class ProcessingDockContainer(QtWidgets.QWidget):
    def __init__(self, title: str, widget: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        self._title = title
        self._content_widget = widget
        self._floating_window: Optional[QtWidgets.QDialog] = None

        self.setObjectName("processingContainer")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QtWidgets.QWidget()
        header.setObjectName("processingHeader")
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
        self.btn_float.setProperty("headerAction", True)
        header_layout.addWidget(self.btn_float)

        self.btn_toggle = QtWidgets.QToolButton()
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(True)
        self.btn_toggle.setAutoRaise(True)
        self.btn_toggle.setArrowType(QtCore.Qt.DownArrow)
        self.btn_toggle.setToolTip("Hide processing pane")
        self.btn_toggle.toggled.connect(self._on_toggle_toggled)
        self.btn_toggle.setProperty("headerAction", True)
        header_layout.addWidget(self.btn_toggle)

        layout.addWidget(header)

        self._content_frame = QtWidgets.QWidget()
        self._content_frame.setObjectName("processingContent")
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
        self._placeholder.setObjectName("processingPlaceholder")
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


class PipelineBuilderDialog(QtWidgets.QDialog):
    def __init__(
        self,
        manager: ProcessingManager,
        pipeline: Optional[ProcessingPipeline] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.manager = manager
        self.steps: List[ProcessingStep] = []
        self._stack_indices: Dict[str, int] = {}
        self._result: Optional[ProcessingPipeline] = None

        self.setWindowTitle("Pipeline builder")
        self.resize(520, 640)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        builder = QtWidgets.QGroupBox("Build step")
        builder_layout = QtWidgets.QVBoxLayout(builder)
        builder_layout.setContentsMargins(6, 6, 6, 6)
        builder_layout.setSpacing(6)

        func_row = QtWidgets.QHBoxLayout()
        func_row.addWidget(QtWidgets.QLabel("Function:"))
        self.cmb_function = QtWidgets.QComboBox()
        self.cmb_function.addItem("Select…", "")
        self.param_stack = QtWidgets.QStackedWidget()
        self.param_stack.addWidget(QtWidgets.QWidget())
        for spec in list_processing_functions():
            self.cmb_function.addItem(spec.label, spec.key)
            form = ParameterForm(spec.parameters_for_data(None))
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

        layout.addWidget(builder)

        self.list_steps = QtWidgets.QListWidget()
        self.list_steps.currentRowChanged.connect(self._on_step_selected)
        layout.addWidget(self.list_steps, 1)

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
        layout.addLayout(step_btns)

        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Pipeline name:"))
        self.edit_name = QtWidgets.QLineEdit()
        self.edit_name.textChanged.connect(lambda _: self._update_buttons())
        name_row.addWidget(self.edit_name, 1)
        layout.addLayout(name_row)

        action_row = QtWidgets.QHBoxLayout()
        self.btn_interactive = QtWidgets.QPushButton("Interactive edit…")
        self.btn_interactive.clicked.connect(self._interactive_edit)
        action_row.addWidget(self.btn_interactive)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        initial = pipeline or ProcessingPipeline(name="", steps=[])
        self.edit_name.setText(initial.name)
        self.steps = [ProcessingStep(step.key, dict(step.params)) for step in initial.steps]
        self._refresh_step_list()
        if self.steps:
            self.list_steps.setCurrentRow(0)
        self._update_buttons()

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
        return ProcessingPipeline(
            name=name,
            steps=[ProcessingStep(step.key, dict(step.params)) for step in self.steps],
        )

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
        if not self.steps:
            return
        if (
            QtWidgets.QMessageBox.question(
                self, "Clear steps", "Remove all steps from the pipeline?"
            )
            != QtWidgets.QMessageBox.Yes
        ):
            return
        self.steps.clear()
        self._refresh_step_list()

    def _interactive_edit(self):
        if not self.steps:
            return
        pipeline = self._build_pipeline()
        dlg = PipelineEditorDialog(self.manager, pipeline, self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        updated = dlg.result_pipeline()
        self.edit_name.setText(updated.name)
        self.steps = [ProcessingStep(step.key, dict(step.params)) for step in updated.steps]
        self._refresh_step_list()
        if self.steps:
            self.list_steps.setCurrentRow(0)

    def accept(self):
        if self._commit_pipeline(close=True):
            super().accept()

    def result_pipeline(self) -> Optional[ProcessingPipeline]:
        return self._result

    def _commit_pipeline(self, *, close: bool) -> bool:
        if not self.steps:
            QtWidgets.QMessageBox.warning(
                self, "Missing steps", "Add at least one processing step before saving."
            )
            return False
        name = self.edit_name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(
                self, "Missing name", "Please enter a name for the pipeline."
            )
            return False
        pipeline = ProcessingPipeline(
            name=name,
            steps=[ProcessingStep(step.key, dict(step.params)) for step in self.steps],
        )
        try:
            self.manager.save_pipeline(pipeline)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(e))
            return False
        self._result = pipeline
        log_action(f"Saved processing pipeline '{pipeline.name}' with {len(self.steps)} step(s)")
        return True


class ProcessingDockWidget(QtWidgets.QWidget):
    def __init__(self, manager: ProcessingManager, parent=None):
        super().__init__(parent)
        self.manager = manager

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        info = QtWidgets.QLabel(
            "Pipelines can be created or edited in a dedicated dialog."
        )
        info.setWordWrap(True)
        outer.addWidget(info)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_new_pipeline = QtWidgets.QPushButton("New pipeline…")
        self.btn_new_pipeline.clicked.connect(self._create_pipeline)
        btn_row.addWidget(self.btn_new_pipeline)
        self.btn_edit_pipeline = QtWidgets.QPushButton("Edit selected…")
        self.btn_edit_pipeline.clicked.connect(self._edit_selected)
        btn_row.addWidget(self.btn_edit_pipeline)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        saved_box = QtWidgets.QGroupBox("Saved pipelines")
        saved_layout = QtWidgets.QVBoxLayout(saved_box)
        saved_layout.setContentsMargins(6, 6, 6, 6)
        saved_layout.setSpacing(6)

        self.list_saved = QtWidgets.QListWidget()
        self.list_saved.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_saved.itemSelectionChanged.connect(self._update_buttons)
        saved_layout.addWidget(self.list_saved, 1)

        saved_btns = QtWidgets.QHBoxLayout()
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

    def _selected_saved_name(self) -> Optional[str]:
        items = self.list_saved.selectedItems()
        if not items:
            return None
        return str(items[0].data(QtCore.Qt.UserRole) or items[0].text())

    def _update_buttons(self):
        has_selection = self._selected_saved_name() is not None
        self.btn_edit_pipeline.setEnabled(has_selection)
        self.btn_delete_saved.setEnabled(has_selection)
        self.btn_export_saved.setEnabled(has_selection)

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

    def _create_pipeline(self):
        self._open_builder(None)

    def _edit_selected(self):
        name = self._selected_saved_name()
        if not name:
            return
        pipeline = self.manager.get_pipeline(name)
        if pipeline is None:
            return
        self._open_builder(pipeline)

    def _open_builder(self, pipeline: Optional[ProcessingPipeline]):
        dialog = PipelineBuilderDialog(self.manager, pipeline, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            result = dialog.result_pipeline()
            self._refresh_saved()
            if result:
                self._select_pipeline(result.name)

    def _select_pipeline(self, name: str):
        for row in range(self.list_saved.count()):
            item = self.list_saved.item(row)
            if item and item.text() == name:
                self.list_saved.setCurrentRow(row)
                break
        self._update_buttons()

    def _delete_saved(self):
        name = self._selected_saved_name()
        if not name:
            return
        if (
            QtWidgets.QMessageBox.question(
                self, "Delete pipeline", f"Delete pipeline '{name}'?"
            )
            != QtWidgets.QMessageBox.Yes
        ):
            return
        self.manager.delete_pipeline(name)
        log_action(f"Deleted processing pipeline '{name}'")
        self._refresh_saved()

    def _export_saved(self):
        name = self._selected_saved_name()
        if not name:
            return
        pipeline = self.manager.get_pipeline(name)
        if pipeline is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export pipeline",
            f"{name}.json",
            "Pipeline JSON (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(pipeline.to_dict(), fh, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))

    def _import_saved(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import pipeline",
            "",
            "Pipeline JSON (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            pipeline = ProcessingPipeline.from_dict(data)
            if not pipeline.name:
                pipeline.name = Path(path).stem
            self.manager.save_pipeline(pipeline)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Import failed", str(e))
            return
        self._refresh_saved()
        self._select_pipeline(pipeline.name)

class ProcessingSelectionDialog(QtWidgets.QDialog):
    applied = QtCore.Signal(str, dict)

    def __init__(
        self,
        manager: Optional[ProcessingManager],
        parent=None,
        *,
        dims: Optional[Sequence[str]] = None,
    ):
        super().__init__(parent)
        self.manager = manager
        self._dims: Optional[Tuple[str, ...]] = tuple(dims) if dims else None
        self._ndim: Optional[int] = len(self._dims) if self._dims is not None else None
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

        for spec in list_processing_functions(self._ndim):
            form = ParameterForm(spec.parameters_for_data(self._dims))
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

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Apply
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        apply_btn = btns.button(QtWidgets.QDialogButtonBox.Apply)
        if apply_btn is not None:
            apply_btn.clicked.connect(self._on_apply_clicked)
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

    def _on_apply_clicked(self):
        mode, params = self.selected_processing()
        self.applied.emit(mode, params)
