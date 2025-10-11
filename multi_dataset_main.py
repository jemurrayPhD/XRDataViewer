#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Tuple
from functools import partial

import numpy as np
import xarray as xr

from PySide2 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

from xr_plot_widget import CentralPlotWidget
from xr_coords import guess_phys_coords
from data_processing import poly2d_detrend, gaussian_blur, median_filter, butterworth

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


# ---------------------------------------------------------------------------
# VarRef: drag/drop handle for a 2D variable inside a dataset on disk
# ---------------------------------------------------------------------------
class VarRef(QtCore.QObject):
    def __init__(self, path: Path, var: str, hint: str = ""):
        super().__init__()
        self.path = Path(path)
        self.var = var          # data variable name
        self.hint = hint        # "(X,Y)" / "(x,y)" / "(pixel)"

    def to_mime(self) -> str:
        return "VarRef:" + json.dumps({"path": str(self.path), "var": self.var, "hint": self.hint})

    @staticmethod
    def from_mime(txt: str) -> Optional["VarRef"]:
        if not txt or not txt.startswith("VarRef:"):
            return None
        try:
            d = json.loads(txt.split(":", 1)[1])
            return VarRef(Path(d["path"]), d["var"], d.get("hint", ""))
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

    def _mimeData(self, _items):
        md = QtCore.QMimeData()
        sel = self.tree.selectedItems()
        if sel:
            txt = sel[0].data(0, QtCore.Qt.UserRole)
            if txt:
                md.setText(txt)
        return md

    def _open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open xarray Dataset", "", "NetCDF / Zarr (*.nc *.zarr);;All files (*)"
        )
        if not path:
            return
        p = Path(path)
        try:
            ds = open_dataset(p)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Open failed", str(e))
            return

        root = QtWidgets.QTreeWidgetItem([str(p)])
        root.setExpanded(True)
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
            child.setData(0, QtCore.Qt.UserRole, VarRef(p, var, hint).to_mime())
            root.addChild(child)


# ---------------------------------------------------------------------------
# ViewerFrame: one tile with the image + optional histogram on the right
# ---------------------------------------------------------------------------
class ViewerFrame(QtWidgets.QFrame):
    request_close = QtCore.Signal(object)

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet("QFrame { border: 1px solid #888; }")

        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(2,2,2,2); lay.setSpacing(2)
        # Header
        hdr = QtWidgets.QFrame(); hl = QtWidgets.QHBoxLayout(hdr); hl.setContentsMargins(6,3,6,3)
        self.lbl = QtWidgets.QLabel(title); hl.addWidget(self.lbl, 1)
        btn_close = QtWidgets.QToolButton(); btn_close.setText("×")
        btn_close.clicked.connect(lambda: self.request_close.emit(self))
        hl.addWidget(btn_close, 0)
        lay.addWidget(hdr, 0)

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

    def set_data(self, da, coords):
        Z = np.asarray(da.values, float)
        if "X" in coords and "Y" in coords:
            self.viewer.set_warped(coords["X"], coords["Y"], Z, autorange=True)
        elif "x" in coords and "y" in coords:
            self.viewer.set_rectilinear(coords["x"], coords["y"], Z, autorange=True)
        else:
            self.viewer.set_image(Z, autorange=True)

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


# ---------------------------------------------------------------------------
# MultiView grid: drag vars to create tiles; master toggle for histograms
# ---------------------------------------------------------------------------
class MultiViewGrid(QtWidgets.QWidget):
    """
    A splitter-based grid of ViewerFrame tiles.
    - Drag a VarRef (from DatasetsPane) onto this widget to add a tile.
    - 'Columns' spinbox controls how many tiles per row.
    - 'Show histograms' toggles the classic HistogramLUTItem to the right of each tile.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.frames: List[ViewerFrame] = []

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
        vr = VarRef.from_mime(ev.mimeData().text())
        if not vr:
            ev.ignore()
            return
        try:
            da, coords = vr.load()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(e))
            ev.ignore()
            return

        fr = ViewerFrame(title=f"{vr.path.name}:{vr.var}", parent=self)
        fr.request_close.connect(self._remove_frame)
        fr.set_data(da, coords)
        self.frames.append(fr)
        self._connect_frame_signals(fr)
        self._sync_new_frame_to_links(fr)

        self._reflow()
        ev.acceptProposedAction()

    # ---------- Tile management ----------
    def _remove_frame(self, fr: ViewerFrame):
        self._disconnect_frame_signals(fr)
        if fr in self.frames:
            self.frames.remove(fr)
        try:
            fr.setParent(None)
        except Exception:
            pass
        self._reflow()

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
                other.show_crosshair(x, y, value=value, mirrored=True, label=label)
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
        try:
            if mode == "gaussian":
                sigma = float(max(params.get("sigma", 1.0), 0.0))
                data = gaussian_blur(data, sigma=sigma)
            elif mode == "median":
                ksize = int(max(params.get("ksize", 3), 1))
                if ksize % 2 == 0:
                    ksize += 1
                data = median_filter(data, ksize=ksize)
            elif mode == "poly":
                ox = int(max(params.get("order_x", 2), 0))
                oy = int(max(params.get("order_y", 2), 0))
                data = poly2d_detrend(data, order_x=ox, order_y=oy)
            elif mode == "butterworth":
                cutoff = float(params.get("cutoff", 0.1))
                order = int(max(params.get("order", 3), 1))
                btype = params.get("btype", "low") or "low"
                data = butterworth(data, cutoff=cutoff, order=order, btype=btype)
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
        proc_box = QtWidgets.QGroupBox("Processing")
        proc_layout = QtWidgets.QVBoxLayout(proc_box)
        proc_layout.setContentsMargins(6, 6, 6, 6)
        proc_layout.setSpacing(4)

        proc_row = QtWidgets.QHBoxLayout()
        proc_row.addWidget(QtWidgets.QLabel("Operation:"))
        self.cmb_processing = QtWidgets.QComboBox()
        self.cmb_processing.addItem("None", "none")
        self.cmb_processing.addItem("Gaussian blur", "gaussian")
        self.cmb_processing.addItem("Median filter", "median")
        self.cmb_processing.addItem("Poly detrend", "poly")
        self.cmb_processing.addItem("Butterworth", "butterworth")
        self.cmb_processing.currentIndexChanged.connect(self._on_processing_mode_changed)
        proc_row.addWidget(self.cmb_processing, 1)
        proc_layout.addLayout(proc_row)

        self.param_stack = QtWidgets.QStackedWidget()
        # None
        self.param_stack.addWidget(QtWidgets.QWidget())
        # Gaussian
        gauss_widget = QtWidgets.QWidget()
        gauss_layout = QtWidgets.QFormLayout(gauss_widget)
        self.spin_sigma = QtWidgets.QDoubleSpinBox()
        self.spin_sigma.setDecimals(2)
        self.spin_sigma.setRange(0.1, 50.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setValue(1.0)
        self.spin_sigma.valueChanged.connect(self._on_processing_params_changed)
        gauss_layout.addRow("Sigma", self.spin_sigma)
        self.param_stack.addWidget(gauss_widget)
        # Median
        median_widget = QtWidgets.QWidget()
        median_layout = QtWidgets.QFormLayout(median_widget)
        self.spin_kernel = QtWidgets.QSpinBox()
        self.spin_kernel.setRange(1, 99)
        self.spin_kernel.setSingleStep(2)
        self.spin_kernel.setValue(3)
        self.spin_kernel.valueChanged.connect(self._on_processing_params_changed)
        median_layout.addRow("Kernel", self.spin_kernel)
        self.param_stack.addWidget(median_widget)
        # Poly detrend
        poly_widget = QtWidgets.QWidget()
        poly_layout = QtWidgets.QFormLayout(poly_widget)
        self.spin_order_x = QtWidgets.QSpinBox(); self.spin_order_x.setRange(0, 6); self.spin_order_x.setValue(2)
        self.spin_order_y = QtWidgets.QSpinBox(); self.spin_order_y.setRange(0, 6); self.spin_order_y.setValue(2)
        self.spin_order_x.valueChanged.connect(self._on_processing_params_changed)
        self.spin_order_y.valueChanged.connect(self._on_processing_params_changed)
        poly_layout.addRow("Order X", self.spin_order_x)
        poly_layout.addRow("Order Y", self.spin_order_y)
        self.param_stack.addWidget(poly_widget)
        # Butterworth
        bw_widget = QtWidgets.QWidget()
        bw_layout = QtWidgets.QFormLayout(bw_widget)
        self.spin_cutoff = QtWidgets.QDoubleSpinBox(); self.spin_cutoff.setDecimals(3); self.spin_cutoff.setRange(0.001, 0.5); self.spin_cutoff.setSingleStep(0.01); self.spin_cutoff.setValue(0.1)
        self.spin_bw_order = QtWidgets.QSpinBox(); self.spin_bw_order.setRange(1, 10); self.spin_bw_order.setValue(3)
        self.spin_cutoff.valueChanged.connect(self._on_processing_params_changed)
        self.spin_bw_order.valueChanged.connect(self._on_processing_params_changed)
        bw_layout.addRow("Cutoff", self.spin_cutoff)
        bw_layout.addRow("Order", self.spin_bw_order)
        self.param_stack.addWidget(bw_widget)

        proc_layout.addWidget(self.param_stack)
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.clicked.connect(self._apply_processing)
        proc_layout.addWidget(self.btn_apply, alignment=QtCore.Qt.AlignRight)
        lay.addWidget(proc_box)

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
        self.cmb_processing.setCurrentText(self._processing_label(self.layer.current_processing))
        self._update_param_stack()
        self._ready = True
        self._apply_processing()

    def _processing_label(self, mode: str) -> str:
        mapping = {
            "none": "None",
            "gaussian": "Gaussian blur",
            "median": "Median filter",
            "poly": "Poly detrend",
            "butterworth": "Butterworth",
        }
        return mapping.get(mode, "None")

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
        mode = self.current_processing()
        if mode == "gaussian":
            return {"sigma": self.spin_sigma.value()}
        if mode == "median":
            return {"ksize": self.spin_kernel.value()}
        if mode == "poly":
            return {"order_x": self.spin_order_x.value(), "order_y": self.spin_order_y.value()}
        if mode == "butterworth":
            return {"cutoff": self.spin_cutoff.value(), "order": self.spin_bw_order.value(), "btype": "low"}
        return {}

    def current_processing(self) -> str:
        data = self.cmb_processing.currentData()
        return data or "none"

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
        self._update_param_stack()
        self._apply_processing()

    def _on_processing_params_changed(self, *_):
        self._apply_processing()

    def _apply_processing(self):
        if not self._ready:
            return
        mode = self.current_processing()
        params = self.processing_parameters()
        self.layer.apply_processing(mode, params)

    def _update_param_stack(self):
        mode = self.current_processing()
        mapping = {"none": 0, "gaussian": 1, "median": 2, "poly": 3, "butterworth": 4}
        idx = mapping.get(mode, 0)
        try:
            self.param_stack.setCurrentIndex(idx)
        except Exception:
            pass


class OverlayView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.layers: List[OverlayLayer] = []

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

        self.datasets = DatasetsPane(); main.addWidget(self.datasets)
        self.tabs = QtWidgets.QTabWidget(); main.addWidget(self.tabs)
        main.setStretchFactor(1, 1)

        self.tab_multiview = MultiViewGrid(); self.tabs.addTab(self.tab_multiview, "MultiView")
        self.tab_overlay = OverlayView();     self.tabs.addTab(self.tab_overlay, "Overlay")

        self.resize(1500, 900)


def main():
    app = QtWidgets.QApplication([])
    pg.setConfigOptions(imageAxisOrder='row-major')
    win = MainWindow(); win.show()
    app.exec_()


if __name__ == "__main__":
    main()
