#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List
from functools import partial

import numpy as np
import xarray as xr

from PySide2 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

from xr_plot_widget import CentralPlotWidget
from xr_coords import guess_phys_coords

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
# Overlay view: simple drag-to-layer
# ---------------------------------------------------------------------------
class OverlayView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(6,6,6,6)
        self.glw = pg.GraphicsLayoutWidget(); v.addWidget(self.glw, 1)
        self.plot = self.glw.addPlot(row=0, col=0); self.plot.invertY(True); self.plot.showGrid(x=False, y=False)
        self.setAcceptDrops(True); self.layers: List[pg.ImageItem] = []

    def dragEnterEvent(self, ev):
        ev.acceptProposedAction() if ev.mimeData().hasText() else ev.ignore()

    def dropEvent(self, ev):
        vr = VarRef.from_mime(ev.mimeData().text())
        if not vr: ev.ignore(); return
        try:
            da, coords = vr.load()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(e)); return
        Z = np.asarray(da.values, float)
        item = pg.ImageItem(); item.setImage(Z.T, autoLevels=True)
        self.plot.addItem(item); self.layers.append(item)
        ev.acceptProposedAction()


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
