#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Tuple
import numpy as np
from PySide2 import QtCore, QtWidgets
import pyqtgraph as pg


class GlobalLUTBar(QtWidgets.QWidget):
    """Horizontal colorbar with Colormap, Gamma, Reset.
    - Compact height but not clipped
    - Compatible across pyqtgraph versions
    - Binds to selected ImageItems
    """
    sigLevelsChanged = QtCore.Signal(float, float)
    sigGammaChanged = QtCore.Signal(float)
    sigColormapChanged = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bound_images: List[pg.ImageItem] = []
        self._cmap_name = 'viridis'

        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(10)

        self.cmb = QtWidgets.QComboBox()
        self.cmb.addItems(['viridis','magma','plasma','inferno','cividis','gray'])
        self.cmb.currentTextChanged.connect(self._on_cmap)
        h.addWidget(QtWidgets.QLabel('Colormap'))
        h.addWidget(self.cmb, 0)

        self.btn_reset = QtWidgets.QPushButton('Reset')
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_reset.setFixedHeight(22)
        h.addWidget(self.btn_reset, 0)

        h.addStretch(1)

        # Colorbar host; sized to avoid clipping
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setMinimumHeight(34)  # host height
        self.glw.setFixedHeight(34)
        self._plt = self.glw.addPlot()
        self._plt.hideAxis('left'); self._plt.hideAxis('bottom')

        try:
            self.cbar = pg.ColorBarItem(values=(0.0, 1.0),
                                        colorMap=pg.colormap.get(self._cmap_name),
                                        orientation='h')
        except Exception:
            self.cbar = pg.ColorBarItem(values=(0.0, 1.0),
                                        colorMap=pg.colormap.get(self._cmap_name))
        self.glw.addItem(self.cbar)
        h.addWidget(self.glw, 1)

        # Hide histogram; make interactive if supported
        for meth, arg in [('setHistogramVisible', False), ('setInteractive', True)]:
            try:
                getattr(self.cbar, meth)(arg)
            except Exception:
                pass

        # Robust levels signal signature
        try:
            self.cbar.sigLevelsChanged.connect(lambda *args: self._on_levels_change(*args))
        except Exception:
            pass

        # Gamma (small)
        self.lbl_gamma = QtWidgets.QLabel('γ=1.00')
        self.sld_gamma = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_gamma.setRange(10, 400)  # 0.1 .. 4.0
        self.sld_gamma.setValue(100)
        self.sld_gamma.setFixedHeight(16)
        self.sld_gamma.valueChanged.connect(self._on_gamma_slider)
        h.addWidget(self.lbl_gamma, 0)
        h.addWidget(self.sld_gamma, 0)

        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        self._apply_cmap()
        self._apply_lut_to_bound(force=True)

    # ----- public api -----
    def bind(self, image_items: List[pg.ImageItem]):
        self._bound_images = [im for im in image_items if im is not None]
        # Ensure valid levels on float images
        for im in self._bound_images:
            try:
                if getattr(im, 'levels', None) is None:
                    img = getattr(im, 'image', None)
                    if img is not None:
                        imgf = np.asarray(img, dtype=float)
                        zmin = float(np.nanmin(imgf)); zmax = float(np.nanmax(imgf))
                        if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax == zmin:
                            zmin, zmax = 0.0, 1.0
                        im.setLevels((zmin, zmax))
            except Exception:
                pass
        self._apply_lut_to_bound(force=True)

    def set_levels(self, lo: float, hi: float):
        try:
            self.cbar.setLevels((float(lo), float(hi)))
        except Exception:
            self._on_levels_change(float(lo), float(hi))

    def levels(self) -> Tuple[float, float]:
        try:
            return tuple(self.cbar.levels)
        except Exception:
            return (0.0, 1.0)

    def set_colormap(self, name: str):
        idx = self.cmb.findText(name)
        if idx >= 0:
            self.cmb.setCurrentIndex(idx)

    def set_gamma(self, g: float):
        sv = int(round(max(0.1, min(4.0, float(g))) * 100))
        self.sld_gamma.setValue(sv)

    def gamma(self) -> float:
        return float(self.sld_gamma.value()) / 100.0

    # ----- internals -----
    def _on_reset(self):
        self.set_colormap('viridis'); self.set_gamma(1.0); self.set_levels(0.0, 1.0)

    def _on_cmap(self, name: str):
        self._cmap_name = name; self._apply_cmap(); self._apply_lut_to_bound(force=True)
        self.sigColormapChanged.emit(name)

    def _on_gamma_slider(self, _v: int):
        g = self.gamma(); self.lbl_gamma.setText(f'γ={g:.2f}')
        self._apply_lut_to_bound(force=True); self.sigGammaChanged.emit(g)

    def _on_levels_change(self, *args):
        # Accept (lo,hi) or (levels,) tuple
        if len(args) == 1:
            a0 = args[0]
            try: lo, hi = float(a0[0]), float(a0[1])
            except Exception: return
        elif len(args) >= 2:
            try: lo, hi = float(args[0]), float(args[1])
            except Exception: return
        else:
            return
        for im in self._bound_images:
            try: im.setLevels((lo, hi))
            except Exception: pass
        self.sigLevelsChanged.emit(lo, hi)

    def _apply_cmap(self):
        try: self.cbar.setColorMap(pg.colormap.get(self._cmap_name))
        except Exception: pass

    def _apply_lut_to_bound(self, force=False):
        try: cm = pg.colormap.get(self._cmap_name)
        except Exception: cm = None
        lut = None
        if cm is not None:
            t = np.linspace(0.0, 1.0, 256); tg = np.power(t, 1.0 / max(0.1, self.gamma()))
            try: lut = cm.map(tg, mode='byte')
            except Exception: lut = None
        if lut is not None:
            for im in self._bound_images:
                try: im.setLookupTable(lut)
                except Exception: pass
