from __future__ import annotations
import numpy as np
from PySide2 import QtCore, QtWidgets
import pyqtgraph as pg

FORCE_SOFT_RENDER = False
if FORCE_SOFT_RENDER:
    pg.setConfigOptions(useOpenGL=False, antialias=False)

class MyHistogramLUT(pg.HistogramLUTItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        QtCore.QTimer.singleShot(0, self._suppress_all_stops)

    def mouseDoubleClickEvent(self, ev):
        ev.ignore()

    def contextMenuEvent(self, ev):
        ev.ignore()

    def _suppress_all_stops(self):
        try:
            g = self.gradient
            if hasattr(g, "ticks"):
                for t in list(getattr(g, "ticks", [])):
                    try:
                        getattr(t, "item", t).setVisible(False)
                    except Exception:
                        try:
                            t.setVisible(False)
                        except Exception:
                            pass
            if hasattr(g, "listTicks"):
                try:
                    for _, _, tick in g.listTicks():
                        try:
                            getattr(tick, "item", tick).setVisible(False)
                        except Exception:
                            try:
                                tick.setVisible(False)
                            except Exception:
                                pass
                except Exception:
                    pass
            try:
                for ch in g.childItems():
                    br = ch.boundingRect()
                    if br.width() < 15 and br.height() < 15:
                        ch.setVisible(False)
            except Exception:
                pass
        except Exception:
            pass

    def rehide_stops(self):
        QtCore.QTimer.singleShot(0, self._suppress_all_stops)

class CentralPlotWidget(QtWidgets.QWidget):
    sigInfoMessage = QtCore.Signal(str)
    sigLevelsChanged = QtCore.Signal(tuple)
    sigViewChanged = QtCore.Signal(tuple, tuple)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.invertY(True)
        self.plot.setMenuEnabled(False)
        self.plot.setLabel("left", "Y")
        self.plot.setLabel("bottom", "X")
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)
        self.lut = MyHistogramLUT()
        self.lut.setImageItem(self.img_item)

        self._hist_container = None
        self._block_levels_emit = False
        self._block_view_emit = False

        lay = QtWidgets.QHBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.addWidget(self.glw)
        try:
            cmap = pg.colormap.get("viridis"); self.lut.gradient.setColorMap(cmap)
        except Exception: pass

        try:
            self.lut.gradient.sigGradientChanged.connect(lambda *_: self.lut.rehide_stops())
        except Exception:
            pass
        try:
            self.lut.sigLevelsChanged.connect(self._on_levels_changed)
        except Exception:
            pass
        try:
            self.plot.vb.sigRangeChanged.connect(self._on_viewbox_range_changed)
        except Exception:
            pass
        try:
            self.lut.rehide_stops()
        except Exception:
            pass

        # sample grid items (on main plot)
        self._grid_items = []

    # ---------- public API ----------
    def set_labels(self, xlabel: str = "X", ylabel: str = "Y"):
        self.plot.setLabel("bottom", xlabel); self.plot.setLabel("left", ylabel)

    def get_levels(self):
        try: return self.lut.getLevels()
        except Exception: return (0.0, 1.0)

    def set_levels(self, lo, hi):
        self._block_levels_emit = True
        try:
            self.lut.setLevels(float(lo), float(hi))
        except Exception:
            pass
        finally:
            self._block_levels_emit = False

    def get_view_range(self):
        try: xr, yr = self.plot.vb.viewRange(); return (tuple(xr), tuple(yr))
        except Exception: return ((0,1),(0,1))

    def set_view_range(self, xr=None, yr=None):
        self._block_view_emit = True
        try:
            if xr is not None: self.plot.vb.setXRange(float(xr[0]), float(xr[1]), padding=0.0)
            if yr is not None: self.plot.vb.setYRange(float(yr[0]), float(yr[1]), padding=0.0)
        except Exception:
            pass
        finally:
            self._block_view_emit = False

    def autoscale_levels(self):
        img = getattr(self.img_item, 'image', None)
        if img is None:
            return
        try:
            data = np.asarray(img, float)
            finite = np.isfinite(data)
            if not finite.any():
                return
            lo = float(data[finite].min())
            hi = float(data[finite].max())
            if lo == hi:
                hi = lo + 1.0
            self.set_levels(lo, hi)
        except Exception:
            pass

    def auto_view_range(self):
        try:
            rect = self.img_item.mapRectToParent(self.img_item.boundingRect())
        except Exception:
            rect = None
        if not rect or rect.isNull():
            return
        try:
            self.plot.vb.setRange(rect=rect, padding=0.0)
        except Exception:
            pass

    def histogram_widget(self):
        if getattr(self, "lut", None) is None:
            return None
        if getattr(self, "_hist_container", None) is None:
            try:
                glw = pg.GraphicsLayoutWidget()
                glw.addItem(self.lut, row=0, col=0)
                glw.setObjectName("HistogramLUTContainer")
                glw.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
                self._hist_container = glw
            except Exception:
                self._hist_container = None
        return self._hist_container

    def _on_levels_changed(self, *_):
        try:
            self.lut.rehide_stops()
        except Exception:
            pass
        if self._block_levels_emit:
            return
        try:
            self.sigLevelsChanged.emit(self.get_levels())
        except Exception:
            pass

    def _on_viewbox_range_changed(self, *_args):
        if self._block_view_emit:
            return
        try:
            xr, yr = self.get_view_range()
            self.sigViewChanged.emit(xr, yr)
        except Exception:
            pass

    # ---------- data display ----------
    def set_image(self, Z: np.ndarray, autorange: bool = True, rect=None):
        Z = np.asarray(Z, float, order="C")
        self.img_item.setImage(Z, autoLevels=autorange)
        try:
            from PySide2.QtCore import QRectF
            if rect is None:
                Ny, Nx = Z.shape; rect = QRectF(0.0, 0.0, float(Nx), float(Ny))
            self.img_item.setRect(rect)
        except Exception: pass
        if autorange: self.plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

    def set_rectilinear(self, x1: np.ndarray, y1: np.ndarray, Z: np.ndarray, autorange: bool = True):
        Zu, xr = self._resample_rectilinear(x1, y1, Z, return_rect=True); self.set_image(Zu, autorange, rect=xr)

    def set_warped(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, autorange: bool = True):
        Zu, xr = self._resample_warped(X, Y, Z, return_rect=True); self.set_image(Zu, autorange, rect=xr)

    # ---------- sample grid overlay on main plot ----------
    def show_sample_grid(self, show: bool, *, x1=None, y1=None, X=None, Y=None, step: int = 10):
        """Draw a subsampled grid on the main plot (not the histogram)."""
        self._clear_grid()
        if not show:
            return
        pen = pg.mkPen((200, 200, 200, 160), width=1)
        items = []
        if X is not None and Y is not None:
            X = np.asarray(X, float); Y = np.asarray(Y, float)
            Ny, Nx = X.shape
            sj = max(1, int(step))
            # vertical (constant column)
            for j in range(0, Nx, sj):
                it = pg.PlotDataItem(X[:, j], Y[:, j], pen=pen); self.plot.addItem(it); items.append(it)
            # horizontal (constant row)
            for i in range(0, Ny, sj):
                it = pg.PlotDataItem(X[i, :], Y[i, :], pen=pen); self.plot.addItem(it); items.append(it)
        elif x1 is not None and y1 is not None:
            x1 = np.asarray(x1, float); y1 = np.asarray(y1, float)
            Ny = y1.size; Nx = x1.size; sj = max(1, int(step))
            for j in range(0, Nx, sj):
                it = pg.PlotDataItem(np.full(Ny, x1[j]), y1, pen=pen); self.plot.addItem(it); items.append(it)
            for i in range(0, Ny, sj):
                it = pg.PlotDataItem(x1, np.full(Nx, y1[i]), pen=pen); self.plot.addItem(it); items.append(it)
        self._grid_items = items

    def _clear_grid(self):
        for it in self._grid_items:
            try: self.plot.removeItem(it)
            except Exception: pass
        self._grid_items = []

    # ---------- resampling helpers ----------
    def _rect_to_qrectf(self, x0, x1, y0, y1):
        from PySide2.QtCore import QRectF; return QRectF(float(x0), float(y0), float(x1 - x0), float(y1 - y0))

    def _resample_rectilinear(self, x1, y1, Z, return_rect=False):
        x1 = np.asarray(x1, float); y1 = np.asarray(y1, float); Z = np.asarray(Z, float)
        Ny, Nx = Z.shape
        xs = np.argsort(x1); ys = np.argsort(y1)
        x_sorted = x1[xs]; y_sorted = y1[ys]; Zs = Z[np.ix_(ys, xs)]
        x_uni = np.linspace(x_sorted[0], x_sorted[-1], Nx); y_uni = np.linspace(y_sorted[0], y_sorted[-1], Ny)
        Zx = np.empty((Ny, Nx), float)
        for i in range(Ny): Zx[i, :] = np.interp(x_uni, x_sorted, Zs[i, :], left=np.nan, right=np.nan)
        Zu = np.empty((Ny, Nx), float)
        for j in range(Nx):
            col = Zx[:, j]; m = np.isfinite(col)
            Zu[:, j] = np.interp(y_uni, y_sorted[m], col[m], left=np.nan, right=np.nan) if m.sum() >= 2 else np.nan
        rect = self._rect_to_qrectf(x_uni[0], x_uni[-1], y_uni[0], y_uni[-1])
        return (Zu, rect) if return_rect else Zu

    def _resample_warped(self, X, Y, Z, return_rect=False):
        try: from scipy.interpolate import griddata
        except Exception:
            rect = self._rect_to_qrectf(0, Z.shape[1], 0, Z.shape[0])
            return (np.asarray(Z, float), rect) if return_rect else np.asarray(Z, float)
        X = np.asarray(X, float); Y = np.asarray(Y, float); Z = np.asarray(Z, float)
        Ny, Nx = Z.shape
        xmin, xmax = np.nanmin(X), np.nanmax(X); ymin, ymax = np.nanmin(Y), np.nanmax(Y)
        x_t = np.linspace(xmin, xmax, Nx); y_t = np.linspace(ymin, ymax, Ny)
        XX, YY = np.meshgrid(x_t, y_t)
        pts = np.column_stack([X.ravel(), Y.ravel()]); vals = Z.ravel()
        Zu = griddata(pts, vals, (XX, YY), method="linear")
        if np.isnan(Zu).any():
            Zun = griddata(pts, vals, (XX, YY), method="nearest"); mask = np.isnan(Zu); Zu[mask] = Zun[mask]
        rect = self._rect_to_qrectf(x_t[0], x_t[-1], y_t[0], y_t[-1])
        return (np.asarray(Zu, float), rect) if return_rect else np.asarray(Zu, float)

