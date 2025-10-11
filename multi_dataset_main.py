#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List

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
        lay.addWidget(self.center_split, 1)

        self.viewer = CentralPlotWidget(self)
        self.center_split.addWidget(self.viewer)

        # placeholder for histogram; replaced when enabled
        self._hist_placeholder = QtWidgets.QWidget()
        self._hist_placeholder.setMinimumWidth(0)
        self._hist_placeholder.setMaximumWidth(0)
        self.center_split.addWidget(self._hist_placeholder)

        # handles for histogram widgets
        self.hlut = None
        self._hist_glw = None

    def set_data(self, da, coords):
        Z = np.asarray(da.values, float)
        if "X" in coords and "Y" in coords:
            self.viewer.set_warped(coords["X"], coords["Y"], Z, autorange=True)
        elif "x" in coords and "y" in coords:
            self.viewer.set_rectilinear(coords["x"], coords["y"], Z, autorange=True)
        else:
            self.viewer.set_image(Z, autorange=True)

    def set_histogram_visible(self, on: bool):
        on = bool(on)

        # Ensure we always have a placeholder widget we can fall back to
        if self._hist_placeholder is None:
            self._hist_placeholder = QtWidgets.QWidget()
            self._hist_placeholder.setMinimumWidth(0)
            self._hist_placeholder.setMaximumWidth(0)
            if self.center_split.count() < 2:
                self.center_split.addWidget(self._hist_placeholder)

        if on:
            if self.hlut is None or self._hist_glw is None:
                try:
                    self._hist_glw = pg.GraphicsLayoutWidget()
                    self.hlut = pg.HistogramLUTItem()
                    try:
                        self.hlut.setImageItem(self.viewer.img_item)
                    except Exception:
                        pass
                    try:
                        self.hlut.gradient.setColorMap(pg.colormap.get("viridis"))
                    except Exception:
                        pass
                    self._hist_glw.addItem(self.hlut)
                except Exception:
                    self.hlut = None
                    self._hist_glw = None
            if self._hist_glw is None:
                return

            if self.center_split.count() < 2:
                self.center_split.addWidget(self._hist_glw)
            else:
                current = self.center_split.widget(1)
                if current is not self._hist_glw:
                    old = self.center_split.replaceWidget(1, self._hist_glw)
                    if old is not None and old is not self._hist_glw:
                        if old is self._hist_placeholder:
                            self._hist_placeholder = old
                        else:
                            try:
                                old.setParent(None)
                            except Exception:
                                pass

            self._hist_glw.setMinimumWidth(220)
            self._hist_glw.setMaximumWidth(16777215)
            try:
                self.center_split.setSizes([1, 0])
            except Exception:
                pass
        else:
            if self.center_split.count() < 2:
                self.center_split.addWidget(self._hist_placeholder)
            else:
                current = self.center_split.widget(1)
                if current is not self._hist_placeholder:
                    old = self.center_split.replaceWidget(1, self._hist_placeholder)
                    if old is not None and old is not self._hist_placeholder:
                        if old is self._hist_glw:
                            self._hist_glw = old
                        try:
                            old.setParent(None)
                        except Exception:
                            pass

            self._hist_placeholder.setMinimumWidth(0)
            self._hist_placeholder.setMaximumWidth(0)


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

        bar.addStretch(1)
        v.addLayout(bar)

        # A vertical splitter holds "rows" (each row is a horizontal splitter of tiles)
        self.vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.vsplit.setChildrenCollapsible(False)
        v.addWidget(self.vsplit, 1)

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

        self._reflow()
        ev.acceptProposedAction()

    # ---------- Tile management ----------
    def _remove_frame(self, fr: ViewerFrame):
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

    def _apply_histogram_visibility(self, on: bool):
        """Show/hide the classic HistogramLUTItem on every tile."""
        on = bool(on)
        for fr in self.frames:
            try:
                fr.set_histogram_visible(on)
            except Exception:
                pass

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
