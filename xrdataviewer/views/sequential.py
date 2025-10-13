from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
import xarray as xr
from PySide2 import QtCore, QtGui, QtWidgets

try:  # Optional dependency for volume rendering
    from pyqtgraph import opengl as gl  # type: ignore
except Exception:  # pragma: no cover - pyqtgraph.opengl may be missing
    gl = None  # type: ignore

try:  # Optional dependency for movie export
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 may be unavailable
    cv2 = None  # type: ignore

from app_logging import log_action
from xr_coords import guess_phys_coords
from xr_plot_widget import CentralPlotWidget, PlotAnnotationConfig, ScientificAxisItem

from ..annotations import PlotAnnotationDialog
from ..datasets import DataSetRef, HighDimVarRef, MemoryDatasetRef, VarRef
from ..preferences import PreferencesManager
from ..processing import apply_processing_step, ProcessingManager, ProcessingSelectionDialog
from ..utils import (
    ask_layout_label,
    ensure_extension,
    image_with_label,
    nan_aware_reducer,
    open_dataset,
    process_events,
    qimage_to_array,
    sanitize_filename,
    save_snapshot,
)
from .volume import SequentialVolumeWindow


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

class SequentialView(QtWidgets.QWidget):
    def __init__(
        self,
        processing_manager: Optional[ProcessingManager] = None,
        preferences: Optional[PreferencesManager] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.processing_manager = processing_manager
        self.preferences: Optional[PreferencesManager] = None

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
        self._line_mode: bool = False
        self._current_line_coords: Optional[np.ndarray] = None
        self._roi_last_shape: Optional[Tuple[int, int]] = None
        self._roi_last_bounds: Optional[Tuple[int, int, int, int]] = None

        self._roi_enabled: bool = False
        self._roi_reducers = {
            "mean": ("Mean", nan_aware_reducer(lambda arr, axis=None: np.nanmean(arr, axis=axis))),
            "median": ("Median", nan_aware_reducer(lambda arr, axis=None: np.nanmedian(arr, axis=axis))),
            "min": ("Minimum", nan_aware_reducer(lambda arr, axis=None: np.nanmin(arr, axis=axis))),
            "max": ("Maximum", nan_aware_reducer(lambda arr, axis=None: np.nanmax(arr, axis=axis))),
            "std": ("Std. dev", nan_aware_reducer(lambda arr, axis=None: np.nanstd(arr, axis=axis))),
            "ptp": (
                "Peak-to-peak",
                nan_aware_reducer(
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
        self._annotation_config: Optional[PlotAnnotationConfig] = None

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

        self.btn_export = QtWidgets.QToolButton()
        self.btn_export.setText("Export")
        self.btn_export.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        export_menu = QtWidgets.QMenu(self.btn_export)
        act_current = export_menu.addAction("Save current slice…")
        act_current.triggered.connect(self._export_current_slice)
        act_all = export_menu.addAction("Save all slices to folder…")
        act_all.triggered.connect(self._export_all_slices)
        act_movie = export_menu.addAction("Save slice movie…")
        act_movie.triggered.connect(self._export_slice_movie)
        act_grid = export_menu.addAction("Save slices as grid pages…")
        act_grid.triggered.connect(self._export_slice_grid)
        export_menu.addSeparator()
        act_layout = export_menu.addAction("Save view layout…")
        act_layout.triggered.connect(self._export_sequential_layout)
        self.btn_export.setMenu(export_menu)
        btn_row.addWidget(self.btn_export)

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

        self.btn_annotations = QtWidgets.QPushButton("Set annotations…")
        self.btn_annotations.setEnabled(False)
        self.btn_annotations.clicked.connect(self._open_annotation_dialog)
        roi_row.addWidget(self.btn_annotations)

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

        self.set_preferences(preferences)

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
        if self.preferences is not None:
            preferred = self.preferences.preferred_colormap(None)
            if preferred:
                idx = self.cmb_colormap.findData(preferred)
                if idx >= 0:
                    self.cmb_colormap.setCurrentIndex(idx)

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
        high_ref = HighDimVarRef.from_mime(text)
        if high_ref:
            try:
                dataset = high_ref.load_dataset()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
                ev.ignore()
                return
            label = high_ref.dataset_label()
            self._set_dataset(dataset, high_ref.path, label=label)
            index = self.cmb_variable.findData(high_ref.var)
            if index >= 0:
                self.cmb_variable.setCurrentIndex(index)
            ev.acceptProposedAction()
            return

        var_ref = VarRef.from_mime(text)
        if var_ref:
            try:
                dataset = open_dataset(var_ref.path)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
                ev.ignore()
                return
            self._set_dataset(dataset, var_ref.path, label=var_ref.path.name)
            index = self.cmb_variable.findData(var_ref.var)
            if index >= 0:
                self.cmb_variable.setCurrentIndex(index)
            ev.acceptProposedAction()
            return

        mem_ref = MemoryDatasetRef.from_mime(text)
        if mem_ref:
            try:
                dataset = mem_ref.load()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
                ev.ignore()
                return
            self._set_dataset(dataset, None, label=mem_ref.display_name())
            ev.acceptProposedAction()
            return

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
        self.btn_annotations.setEnabled(False)
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

    def _set_dataset(self, ds: xr.Dataset, path: Optional[Path], label: Optional[str] = None):
        if self._dataset is not None and self._dataset is not ds:
            try:
                self._dataset.close()
            except Exception:
                pass
        self._dataset = ds
        self._dataset_path = Path(path) if path else None
        if label:
            self.lbl_dataset.setText(label)
        elif self._dataset_path:
            self.lbl_dataset.setText(self._dataset_path.name)
        else:
            self.lbl_dataset.setText("(in-memory dataset)")
        self.lbl_dataset.setStyleSheet("")
        self._clear_view()
        vars_with_dims = [var for var in ds.data_vars if getattr(ds[var], "ndim", 0) >= 2]
        if not vars_with_dims:
            self.cmb_variable.addItem("No 2D or higher variables available", None)
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
        self._line_mode = False
        self._current_line_coords = None
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
        self.btn_annotations.setEnabled(True)
        self._apply_viewer_annotations()

    def _rebuild_axis_controls(self):
        dims = self._dims
        self.cmb_slice_axis.blockSignals(True)
        self.cmb_slice_axis.clear()
        for dim in dims:
            self.cmb_slice_axis.addItem(dim, dim)
        self.cmb_slice_axis.blockSignals(False)
        self.cmb_slice_axis.setEnabled(bool(dims))

        self.cmb_row_axis.blockSignals(True)
        self.cmb_row_axis.clear()
        for dim in dims:
            self.cmb_row_axis.addItem(dim, dim)
        self.cmb_row_axis.blockSignals(False)
        self.cmb_row_axis.setEnabled(bool(dims))

        self.cmb_col_axis.blockSignals(True)
        self.cmb_col_axis.clear()
        self.cmb_col_axis.addItem("1D profile (no column axis)", None)
        for dim in dims:
            self.cmb_col_axis.addItem(dim, dim)
        self.cmb_col_axis.blockSignals(False)
        self.cmb_col_axis.setEnabled(bool(dims))

        if dims:
            self.cmb_slice_axis.setCurrentIndex(0)
            if len(dims) >= 2:
                self.cmb_row_axis.setCurrentIndex(1)
            else:
                self.cmb_row_axis.setCurrentIndex(0)
        if len(dims) >= 3:
            idx = self.cmb_col_axis.findData(dims[2])
            if idx >= 0:
                self.cmb_col_axis.setCurrentIndex(idx)
        elif len(dims) >= 2:
            self.cmb_col_axis.setCurrentIndex(0)
        self._ensure_unique_axes()
        self._rebuild_fixed_indices()

    def _ensure_unique_axes(self):
        dims = self._dims
        combos = (self.cmb_slice_axis, self.cmb_row_axis, self.cmb_col_axis)
        seen: List[str] = []
        for combo in combos:
            if combo.count() == 0:
                continue
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
        self._line_mode = self._col_axis is None
        if self._line_mode and self._roi_enabled:
            self._on_roi_toggled(False)
        self._update_axis_coords()
        self._update_volume_window_axis_labels()

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
        self._current_line_coords = _extract("line_values", (1,))
        if self._current_line_coords is None and self._row_axis:
            self._current_line_coords = _extract(self._row_axis, (1,))

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
        self.btn_autoscale.setEnabled(enabled and not self._line_mode)
        self.btn_autorange.setEnabled(enabled)
        self.btn_toggle_roi.setEnabled(enabled and not self._line_mode)
        self._update_volume_button_state()
        self._update_slice_label()
        if self._line_mode:
            self.cmb_colormap.setEnabled(False)

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
        if self._current_da is None or self._row_axis is None:
            return None, {}
        select = self._gather_selection(slice_index)
        try:
            slice_da = self._current_da.isel(select)
        except Exception:
            return None, {}
        if self._slice_axis is None:
            return None, {}
        if self._col_axis is None:
            if self._row_axis not in slice_da.dims:
                return None, {}
            try:
                slice_da = slice_da.transpose(self._row_axis)
            except Exception:
                pass
            data = np.asarray(slice_da.values, float).reshape(-1)
            coords = guess_phys_coords(slice_da)
            try:
                coord = slice_da.coords.get(self._row_axis)
                if coord is not None:
                    values = np.asarray(coord.values)
                    if values.ndim == 1:
                        coords["line_values"] = np.asarray(values, float)
            except Exception:
                pass
            return data, coords
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
            if self._line_mode:
                self.viewer.set_line(np.array([], dtype=float), np.array([], dtype=float), autorange=True)
            else:
                self.viewer.set_image(np.zeros((1, 1)), autorange=True)
            self._roi_last_shape = None
            if self._roi_enabled:
                self._update_roi_curve()
            self.cmb_colormap.setEnabled(False)
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
        if self._line_mode or np.ndim(self._current_processed_slice) <= 1:
            xs = self._current_line_coords
            if xs is None or xs.size != self._current_processed_slice.size:
                xs = np.arange(self._current_processed_slice.size, dtype=float)
            self.viewer.set_line(xs, self._current_processed_slice, autorange=autorange)
            self.cmb_colormap.setEnabled(False)
        else:
            if "X" in coords and "Y" in coords:
                self.viewer.set_warped(coords["X"], coords["Y"], processed, autorange=autorange)
            elif "x" in coords and "y" in coords:
                self.viewer.set_rectilinear(coords["x"], coords["y"], processed, autorange=autorange)
            else:
                self.viewer.set_image(processed, autorange=autorange)
            self.cmb_colormap.setEnabled(True)
            self._apply_preference_colormap()
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
        if self._roi_enabled and not self._line_mode:
            self._update_roi_slice_reference()
            self._update_roi_curve()
        self._refresh_volume_window()
        self._apply_viewer_annotations()

    def _on_autoscale_clicked(self):
        self.viewer.autoscale_levels()

    def _on_autorange_clicked(self):
        self.viewer.auto_view_range()

    def _annotation_context(self) -> Dict[str, object]:
        idx = int(self._slice_index)
        coords = self._axis_coords
        value = None
        if coords is not None and 0 <= idx < coords.size:
            raw = coords[idx]
            if isinstance(raw, np.ndarray) and raw.size == 1:
                raw = raw.item()
            try:
                value = float(raw)
            except Exception:
                value = raw
        context: Dict[str, object] = {
            "slice_idx": idx,
            "slice_number": idx + 1,
            "n": idx,
            "slice_axis": self._slice_axis or "slice",
        }
        context["slice_val"] = value if value is not None else idx
        return context

    def _apply_viewer_annotations(self):
        if self._annotation_config is None:
            return
        try:
            context = self._annotation_context()
            self.viewer.apply_annotation(self._annotation_config, context=context)
        except Exception:
            pass

    def _open_annotation_dialog(self):
        initial = self.viewer.annotation_defaults()
        if self._annotation_config is not None:
            initial = replace(self._annotation_config, apply_to_all=False)
        hint = (
            "Tip: use Python f-string fields like {slice_idx}, {slice_number}, {slice_val:.3f}, and {slice_axis} "
            "to customise labels per slice."
        )
        dialog = PlotAnnotationDialog(
            self,
            initial=initial,
            allow_apply_all=False,
            template_hint=hint,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        config = dialog.annotation_config()
        if config is None:
            return
        config = replace(config, apply_to_all=False)
        self._annotation_config = config
        self._apply_viewer_annotations()

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
        self._apply_preference_colormap()
        self._apply_selected_colormap()
        if self._volume_window is not None:
            try:
                self._volume_window.set_preferences(preferences)
            except Exception:
                pass

    def _initial_path(self, filename: str) -> str:
        base = self._default_export_dir()
        if base:
            return str(Path(base) / filename)
        return filename

    def _apply_preference_colormap(self):
        if not self.preferences:
            return
        preferred = self.preferences.preferred_colormap(self._current_variable)
        if preferred:
            idx = self.cmb_colormap.findData(preferred)
            if idx >= 0 and idx != self.cmb_colormap.currentIndex():
                block = self.cmb_colormap.blockSignals(True)
                self.cmb_colormap.setCurrentIndex(idx)
                self.cmb_colormap.blockSignals(block)

    def _on_preferences_changed(self, _data):
        self._apply_preference_colormap()
        self._apply_selected_colormap()

    # ---------- export helpers ----------
    def _export_current_slice(self):
        if self._slice_count <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "No slice",
                "Load a dataset and choose a slice before exporting.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save current slice",
            self._initial_path("sequential-slice.png"),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        ok, label = ask_layout_label(self, "Slice label", self._default_layout_label())
        if not ok:
            return
        suffix = ".jpg" if path.lower().endswith((".jpg", ".jpeg")) else ".png"
        target = ensure_extension(path, suffix)
        process_events()
        image = self.viewer.grab().toImage()
        if label:
            image = image_with_label(image, label)
        if not image.save(str(target)):
            QtWidgets.QMessageBox.warning(self, "Save failed", "Unable to save the slice image.")
            return
        self._store_export_dir(str(Path(target).parent))
        log_action(f"Saved sequential slice {self._slice_index} to {target}")

    def _export_all_slices(self):
        if self._slice_count <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "No slices",
                "Load a dataset with at least one slice before exporting.",
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
        base_name = sanitize_filename(self.lbl_dataset.text()) or "slice"
        original = self._slice_index
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            count = 0
            for idx in range(self._slice_count):
                self._on_slice_changed(idx)
                process_events()
                image = self.viewer.grab().toImage()
                target = base / f"{base_name}_{idx:03d}.png"
                if image.save(str(target)):
                    count += 1
            if count == 0:
                QtWidgets.QMessageBox.warning(self, "Export failed", "No slices were exported.")
                return
        finally:
            if self._slice_count > 0:
                self._on_slice_changed(original)
            QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QMessageBox.information(
            self,
            "Export complete",
            f"Saved {count} slice image(s) to {base}",
        )
        log_action(f"Exported {count} sequential slices to {base}")

    def _export_slice_movie(self):
        if self._slice_count <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "No slices",
                "Load a dataset with at least one slice before exporting a movie.",
            )
            return
        if cv2 is None:
            QtWidgets.QMessageBox.warning(
                self,
                "OpenCV unavailable",
                "Saving movies requires OpenCV (cv2). Install it and try again.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save slice movie",
            self._initial_path("sequential-slices.mp4"),
            "MP4 video (*.mp4);;AVI video (*.avi)",
        )
        if not path:
            return
        suffix = ".avi" if path.lower().endswith(".avi") else ".mp4"
        target = ensure_extension(path, suffix)
        fps, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "Frames per second",
            "Frames per second:",
            10.0,
            0.1,
            120.0,
            1,
        )
        if not ok:
            return
        fourcc = cv2.VideoWriter_fourcc(*("MJPG" if suffix == ".avi" else "mp4v"))
        original = self._slice_index
        writer = None
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for idx in range(self._slice_count):
                self._on_slice_changed(idx)
                process_events()
                image = self.viewer.grab().toImage()
                frame = qimage_to_array(image)
                if frame.size == 0:
                    continue
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                height, width = bgr.shape[:2]
                if writer is None:
                    writer = cv2.VideoWriter(str(target), fourcc, float(fps), (width, height))
                    if not writer.isOpened():
                        writer = None
                        raise RuntimeError("Unable to open video writer")
                if writer is not None:
                    writer.write(bgr)
        except Exception as exc:
            if writer is not None:
                writer.release()
            QtWidgets.QApplication.restoreOverrideCursor()
            self._on_slice_changed(original)
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))
            return
        finally:
            if writer is not None:
                writer.release()
            if self._slice_count > 0:
                self._on_slice_changed(original)
            QtWidgets.QApplication.restoreOverrideCursor()
        self._store_export_dir(str(Path(target).parent))
        QtWidgets.QMessageBox.information(
            self,
            "Export complete",
            f"Saved movie to {target}",
        )
        log_action(f"Saved sequential slice movie to {target}")

    def _export_slice_grid(self):
        if self._slice_count <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "No slices",
                "Load a dataset with at least one slice before exporting grids.",
            )
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select export folder",
            self._default_export_dir(),
        )
        if not directory:
            return
        rows, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Grid rows",
            "Number of rows per page:",
            2,
            1,
            10,
        )
        if not ok:
            return
        cols, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Grid columns",
            "Number of columns per page:",
            2,
            1,
            10,
        )
        if not ok:
            return
        ok, label_prefix = ask_layout_label(self, "Grid label")
        if not ok:
            return
        base = Path(directory)
        self._store_export_dir(directory)
        images: List[QtGui.QImage] = []
        original = self._slice_index
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for idx in range(self._slice_count):
                self._on_slice_changed(idx)
                process_events()
                images.append(self.viewer.grab().toImage())
        finally:
            if self._slice_count > 0:
                self._on_slice_changed(original)
            QtWidgets.QApplication.restoreOverrideCursor()
        if not images:
            QtWidgets.QMessageBox.warning(self, "Export failed", "No slice images were captured.")
            return
        tile_width = images[0].width()
        tile_height = images[0].height()
        per_page = max(1, rows * cols)
        count = 0
        for page_idx in range(0, len(images), per_page):
            subset = images[page_idx : page_idx + per_page]
            if not subset:
                continue
            page_image = QtGui.QImage(
                cols * tile_width,
                rows * tile_height,
                QtGui.QImage.Format_ARGB32,
            )
            page_image.fill(QtGui.QColor("white"))
            painter = QtGui.QPainter(page_image)
            for idx, img in enumerate(subset):
                r = idx // cols
                c = idx % cols
                painter.drawImage(c * tile_width, r * tile_height, img)
            painter.end()
            label = label_prefix
            if label_prefix:
                label = f"{label_prefix} – page {count + 1}"
            if label:
                page_image = image_with_label(page_image, label)
            target = base / f"slice-grid_{count + 1:02d}.png"
            page_image.save(str(target))
            count += 1
        if count == 0:
            QtWidgets.QMessageBox.warning(self, "Export failed", "No grids were produced.")
            return
        QtWidgets.QMessageBox.information(
            self,
            "Export complete",
            f"Saved {count} grid page(s) to {base}",
        )
        log_action(f"Exported {count} sequential slice grids to {base}")

    def _export_sequential_layout(self):
        if self._slice_count <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "No slices",
                "Load a dataset before exporting the layout.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save sequential view layout",
            self._initial_path("sequential-layout.png"),
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
        log_action(f"Saved sequential view layout to {target}")

    # ---------- volume viewer ----------
    def _invalidate_volume_cache(self):
        self._volume_cache = None

    def _ensure_volume_window(self) -> SequentialVolumeWindow:
        if gl is None:
            raise RuntimeError("pyqtgraph.opengl is not available")
        window = self._volume_window
        if window is None:
            window = SequentialVolumeWindow(self, self.preferences)
            window.closed.connect(self._on_volume_window_closed)
            self._volume_window = window
        else:
            try:
                window.set_preferences(self.preferences)
            except Exception:
                pass
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
        slice_label = self._axis_display_name(self._slice_axis)
        row_label = self._axis_display_name(self._row_axis)
        col_label = self._axis_display_name(self._col_axis)
        window.set_axis_labels(slice_label, row_label, col_label)
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

    def _update_volume_window_axis_labels(self):
        if self._volume_window is None:
            return
        try:
            slice_label = self._axis_display_name(self._slice_axis)
            row_label = self._axis_display_name(self._row_axis)
            col_label = self._axis_display_name(self._col_axis)
            self._volume_window.set_axis_labels(slice_label, row_label, col_label)
        except Exception:
            pass

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
