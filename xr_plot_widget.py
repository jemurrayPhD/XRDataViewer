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
    def mouseDoubleClickEvent(self, ev): ev.ignore()
    def contextMenuEvent(self, ev): ev.ignore()
    def _suppress_all_stops(self):
        try:
            g = self.gradient
            if hasattr(g, "ticks"):
                for t in list(getattr(g, "ticks", [])):
                    try: getattr(t, "item", t).setVisible(False)
                    except Exception:
                        try: t.setVisible(False)
                        except Exception: pass
            if hasattr(g, "listTicks"):
                try:
                    for _, _, tick in g.listTicks():
                        try: getattr(tick, "item", tick).setVisible(False)
                        except Exception:
                            try: tick.setVisible(False)
                            except Exception: pass
                except Exception: pass
            try:
                for ch in g.childItems():
                    br = ch.boundingRect()
                    if br.width() < 15 and br.height() < 15:
                        ch.setVisible(False)
            except Exception: pass
        except Exception: pass
    def rehide_stops(self):
        QtCore.QTimer.singleShot(0, self._suppress_all_stops)

class CentralPlotWidget(QtWidgets.QWidget):
    sigInfoMessage = QtCore.Signal(str)
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
        self.glw.ci.addItem(self.lut, row=0, col=1)
        self.lut.setImageItem(self.img_item)

        # Histogram overlay: mapping curve + control points (kept from your last build)
        self.hist_curve = pg.PlotDataItem(pen=pg.mkPen(0, 200, 255, 220), antialias=True)
        try: self.hist_curve.setIgnoreBounds(True)
        except Exception: pass
        ovb = self._ensure_overlay_vb()
        if isinstance(ovb, pg.ViewBox):
            ovb.addItem(self.hist_curve); self.hist_curve.setZValue(1e6)
        else:
            self.hist_curve.setVisible(False)

        self.tf_points = pg.ScatterPlotItem(size=8, brush=(80,170,255), pen=pg.mkPen(20,90,140))
        self.tf_points.setZValue(1e6)
        if isinstance(ovb, pg.ViewBox): ovb.addItem(self.tf_points)
        self._dragging_idx = -1; self._drag_capturing = False
        self._init_transfer_points(7)

        lay = QtWidgets.QHBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.addWidget(self.glw)
        try:
            cmap = pg.colormap.get("viridis"); self.lut.gradient.setColorMap(cmap)
        except Exception: pass

        try: self.lut.gradient.sigGradientChanged.connect(self._update_image_lut_from_gradient)
        except Exception: pass
        try:
            self.lut.sigLevelsChanged.connect(lambda *_: (self._update_hist_overlay_curve(),
                                                          self._push_tf_points_visual(),
                                                          self.lut.rehide_stops()))
            base_vb = self._lut_viewbox()
            if isinstance(base_vb, pg.ViewBox):
                base_vb.sigRangeChanged.connect(lambda *_: self._update_hist_overlay_curve())
                base_vb.sigResized.connect(lambda *_: self._sync_overlay_geom())
        except Exception: pass
        ovb = self._ensure_overlay_vb()
        if isinstance(ovb, pg.ViewBox): ovb.installEventFilter(self)

        QtCore.QTimer.singleShot(0, self._update_image_lut_from_gradient)
        QtCore.QTimer.singleShot(0, self._update_hist_overlay_curve)
        QtCore.QTimer.singleShot(50, self._update_hist_overlay_curve)
        try: self.lut.rehide_stops()
        except Exception: pass

        # sample grid items (on main plot)
        self._grid_items = []

    # ---------- public API ----------
    def set_labels(self, xlabel: str = "X", ylabel: str = "Y"):
        self.plot.setLabel("bottom", xlabel); self.plot.setLabel("left", ylabel)

    def get_levels(self):
        try: return self.lut.getLevels()
        except Exception: return (0.0, 1.0)

    def set_levels(self, lo, hi):
        try:
            self.lut.setLevels(float(lo), float(hi))
            self._update_hist_overlay_curve(); self._update_image_lut_from_gradient()
        except Exception: pass

    def get_view_range(self):
        try: xr, yr = self.plot.vb.viewRange(); return (tuple(xr), tuple(yr))
        except Exception: return ((0,1),(0,1))

    def set_view_range(self, xr=None, yr=None):
        try:
            if xr is not None: self.plot.vb.setXRange(float(xr[0]), float(xr[1]), padding=0.0)
            if yr is not None: self.plot.vb.setYRange(float(yr[0]), float(yr[1]), padding=0.0)
        except Exception: pass

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

    # ---------- histogram overlay (mapping curve) ----------
    def _lut_viewbox(self):
        try:
            vb = self.lut.vb
            if isinstance(vb, pg.ViewBox): return vb
        except Exception: pass
        try:
            vb = self.lut.getViewBox()
            if isinstance(vb, pg.ViewBox): return vb
        except Exception: pass
        try:
            for ch in self.lut.childItems():
                if isinstance(ch, pg.ViewBox): return ch
        except Exception: pass
        return None

    def _sync_overlay_geom(self):
        base_vb = self._lut_viewbox(); ovb = getattr(self, "_lut_overlay_vb", None)
        try:
            if isinstance(base_vb, pg.ViewBox) and isinstance(ovb, pg.ViewBox):
                ovb.setGeometry(base_vb.sceneBoundingRect())
        except Exception: pass

    def _ensure_overlay_vb(self):
        if getattr(self, "_lut_overlay_vb", None) is not None:
            return self._lut_overlay_vb
        base_vb = self._lut_viewbox()
        if not isinstance(base_vb, pg.ViewBox):
            self._lut_overlay_vb = None; return None
        ovb = pg.ViewBox(enableMenu=False); ovb.setMouseEnabled(x=False, y=False); ovb.setZValue(1e6)
        base_vb.scene().addItem(ovb); self._lut_overlay_vb = ovb; self._sync_overlay_geom()
        try: base_vb.sigRangeChanged.connect(lambda *_: self._sync_overlay_geom())
        except Exception: pass
        try: base_vb.sigResized.connect(lambda *_: self._sync_overlay_geom())
        except Exception: pass
        return ovb

    def _update_hist_overlay_curve(self):
        try: lo, hi = self.lut.getLevels()
        except Exception: lo, hi = 0.0, 1.0
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo: lo, hi = 0.0, 1.0
        ovb = self._ensure_overlay_vb()
        if not isinstance(ovb, pg.ViewBox): self.hist_curve.setVisible(False); return
        self.hist_curve.setVisible(True)
        try:
            ovb.enableAutoRange(x=False, y=False); ovb.setXRange(lo, hi, padding=0.0); ovb.setYRange(lo, hi, padding=0.0)
        except Exception: pass
        u = np.linspace(0.0, 1.0, 200); v = self._eval_transfer(u)
        x = lo + v * (hi - lo); y = lo + u * (hi - lo)
        try:
            self.hist_curve.setData(x, y); self.hist_curve.setIgnoreBounds(True); self.lut.rehide_stops()
        except Exception: pass

    # ---------- transfer curve logic (fixed-v controls) ----------
    def _init_transfer_points(self, n=7):
        n = max(2, int(n)); self._v_ctrl = np.linspace(0.0, 1.0, n); self._u_ctrl = self._v_ctrl.copy()
        self._push_tf_points_visual()

    def _push_tf_points_visual(self):
        try: lo, hi = self.lut.getLevels()
        except Exception: lo, hi = 0.0, 1.0
        x = lo + self._v_ctrl * (hi - lo); y = lo + self._u_ctrl * (hi - lo)
        try: self.tf_points.setData(x=x, y=y, data=list(range(len(x)))); self.tf_points.setVisible(True)
        except Exception: pass

    def _on_transfer_changed(self):
        self._push_tf_points_visual(); self._update_image_lut_from_gradient(); self._update_hist_overlay_curve()

    def _eval_transfer(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, float)
        if getattr(self, "_v_ctrl", None) is None or getattr(self, "_u_ctrl", None) is None:
            return np.clip(u, 0.0, 1.0)
        u_of_v = np.maximum.accumulate(self._u_ctrl.copy())
        u_of_v[0] = 0.0; u_of_v[-1] = 1.0; v_grid = self._v_ctrl
        v = np.interp(u, u_of_v, v_grid); return np.clip(v, 0.0, 1.0)

    def reset_transfer_curve(self, n_points: int = 7):
        n = max(2, int(n_points)); self._v_ctrl = np.linspace(0.0, 1.0, n); self._u_ctrl = self._v_ctrl.copy(); self._on_transfer_changed()

    def _update_image_lut_from_gradient(self):
        try: cm = self.lut.gradient.colorMap()
        except Exception: return
        pos = np.linspace(0.0, 1.0, 256); v = self._eval_transfer(pos)
        try:
            rgba = cm.map(v, mode='byte')
            if isinstance(rgba, np.ndarray) and rgba.ndim == 2 and rgba.shape[1] >= 3: lut = rgba
            else: raise Exception("Bad colormap output")
        except Exception:
            base = cm.getLookupTable(0.0,1.0,256,alpha=True); xi = v*(base.shape[0]-1)
            i0 = np.floor(xi).astype(int); i1 = np.clip(i0+1, 0, base.shape[0]-1); t = (xi - i0)[:,None]
            lut = (base[i0]*(1-t) + base[i1]*t).astype(np.uint8)
        try: self.img_item.setLookupTable(lut)
        except Exception: pass
