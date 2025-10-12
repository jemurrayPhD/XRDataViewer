from __future__ import annotations

import math

import numpy as np
from PySide2 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

FORCE_SOFT_RENDER = False
if FORCE_SOFT_RENDER:
    pg.setConfigOptions(useOpenGL=False, antialias=False)

class ScientificAxisItem(pg.AxisItem):
    """Axis item that formats ticks with scientific notation and limited precision."""

    def __init__(self, orientation, *, significant_figures: int = 4, **kwargs):
        super().__init__(orientation=orientation, **kwargs)
        self._sig_figs = max(1, int(significant_figures))

    def tickStrings(self, values, scale, spacing):  # noqa: N802 (pyqtgraph API)
        formatted = []
        sci_precision = max(0, self._sig_figs - 1)
        for value in values:
            try:
                scaled = float(value) * float(scale)
            except Exception:
                formatted.append("")
                continue
            if not np.isfinite(scaled):
                formatted.append("")
                continue
            if abs(scaled) < 1e-15:
                formatted.append("0")
                continue
            abs_scaled = abs(scaled)
            use_scientific = abs_scaled >= 1e3 or (abs_scaled > 0 and abs_scaled < 1e-3)
            if use_scientific:
                formatted.append(
                    np.format_float_scientific(
                        scaled,
                        precision=sci_precision,
                        exp_digits=2,
                        trim="k",
                    )
                )
                continue
            digits = max(1, self._sig_figs)
            text = format(scaled, f".{digits}g")
            if "e" in text or "E" in text:
                try:
                    decimals = max(0, digits - int(math.floor(math.log10(abs_scaled))) - 1)
                except ValueError:
                    decimals = digits
                text = format(scaled, f".{decimals}f")
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            formatted.append(text)
        return formatted


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
    sigCursorMoved = QtCore.Signal(object, float, float, object, bool, str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.glw = pg.GraphicsLayoutWidget()
        axis_items = {
            "bottom": ScientificAxisItem("bottom"),
            "left": ScientificAxisItem("left"),
        }
        self.plot = self.glw.addPlot(row=0, col=0, axisItems=axis_items)
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
        self._last_data = None
        self._last_rect = None

        self._histogram_menu_getter = None
        self._histogram_menu_setter = None
        self._histogram_menu_enabled_getter = None

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

        # crosshair overlay
        cross_pen = pg.mkPen((255, 230, 150, 200), width=1)
        mirror_pen = pg.mkPen((120, 210, 255, 200), width=1)
        self._crosshair_pen = cross_pen
        self._crosshair_pen_mirror = mirror_pen
        self._crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self._crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=cross_pen)
        self._crosshair_label = pg.TextItem(color=(255, 255, 220), anchor=(0.0, 1.0))
        for it in (self._crosshair_v, self._crosshair_h):
            self.plot.addItem(it, ignoreBounds=True)
            it.setVisible(False)
        self.plot.addItem(self._crosshair_label, ignoreBounds=True)
        self._crosshair_label.setVisible(False)
        self._crosshair_is_mirrored = False
        self._local_crosshair_enabled = False
        self._local_crosshair_visible = False
        self._last_local_crosshair = None

        try:
            self.plot.scene().sigMouseMoved.connect(self._on_scene_mouse_moved)
        except Exception:
            pass
        try:
            self.plot.scene().sigMouseClicked.connect(self._on_scene_mouse_clicked)
        except Exception:
            pass

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
        self._last_data = np.asarray(Z, float)
        try:
            self._last_rect = rect
        except Exception:
            self._last_rect = None
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

    def show_crosshair(self, x: float, y: float, value=None, *, mirrored: bool = False, label: str | None = None):
        try:
            self._crosshair_v.setPen(self._crosshair_pen_mirror if mirrored else self._crosshair_pen)
            self._crosshair_h.setPen(self._crosshair_pen_mirror if mirrored else self._crosshair_pen)
            self._crosshair_v.setPos(float(x))
            self._crosshair_h.setPos(float(y))
        except Exception:
            return
        if label is None:
            label = self._format_crosshair_text(x, y, value)
        if mirrored:
            self._crosshair_label.setColor((185, 235, 255))
            self._crosshair_label.setAnchor((1.0, 0.0))
            self._crosshair_label.setPos(float(x), float(y))
        else:
            self._crosshair_label.setColor((255, 255, 220))
            self._crosshair_label.setAnchor((0.0, 1.0))
            self._crosshair_label.setPos(float(x), float(y))
        self._crosshair_label.setText(label)
        self._crosshair_is_mirrored = bool(mirrored)
        self._crosshair_v.setVisible(True)
        self._crosshair_h.setVisible(True)
        self._crosshair_label.setVisible(True)
        self._local_crosshair_visible = not mirrored

    def hide_crosshair(self):
        for it in (self._crosshair_v, self._crosshair_h):
            try:
                it.setVisible(False)
            except Exception:
                pass
        try:
            self._crosshair_label.setVisible(False)
        except Exception:
            pass
        self._crosshair_is_mirrored = False
        self._local_crosshair_visible = False

    def clear_mirrored_crosshair(self):
        if self._crosshair_is_mirrored:
            self.hide_crosshair()

    def _format_crosshair_text(self, x, y, value):
        def fmt(val):
            if val is None:
                return "—"
            try:
                if isinstance(val, (float, int, np.floating, np.integer)):
                    if np.isnan(val):
                        return "nan"
                    return f"{float(val):.4g}"
            except Exception:
                pass
            return str(val)
        return f"x={fmt(x)}\ny={fmt(y)}\nvalue={fmt(value)}"

    def _value_at(self, x: float, y: float):
        data = self._last_data
        rect = self._last_rect
        if data is None or rect is None:
            return None
        try:
            x0 = float(rect.left()); y0 = float(rect.top())
            w = float(rect.width()); h = float(rect.height())
        except Exception:
            return None
        if w == 0 or h == 0:
            return None
        Ny, Nx = data.shape
        fx = (x - x0) / w * Nx
        fy = (y - y0) / h * Ny
        if fx < 0 or fx >= Nx or fy < 0 or fy >= Ny:
            return None
        try:
            ix = int(np.clip(np.floor(fx + 0.5), 0, Nx - 1))
            iy = int(np.clip(np.floor(fy + 0.5), 0, Ny - 1))
            return data[iy, ix]
        except Exception:
            return None

    def value_at(self, x: float, y: float):
        """Return the value currently displayed at the given coordinates."""
        return self._value_at(x, y)

    def _set_last_local_crosshair(self, x: float, y: float, value, label: str | None):
        self._last_local_crosshair = (x, y, value, label)

    def set_local_crosshair_enabled(self, enabled: bool):
        enabled = bool(enabled)
        if self._local_crosshair_enabled == enabled:
            return
        self._local_crosshair_enabled = enabled
        self._update_local_crosshair()

    def _update_local_crosshair(self):
        if self._local_crosshair_enabled and self._last_local_crosshair:
            x, y, value, label = self._last_local_crosshair
            self.show_crosshair(x, y, value, mirrored=False, label=label)
        elif self._local_crosshair_visible:
            self.hide_crosshair()

    def _on_scene_mouse_moved(self, pos):
        try:
            scene_rect = self.plot.sceneBoundingRect()
        except Exception:
            scene_rect = None
        inside = bool(scene_rect and scene_rect.contains(pos))
        if not inside:
            self._last_local_crosshair = None
            self._update_local_crosshair()
            try:
                self.sigCursorMoved.emit(self, float("nan"), float("nan"), None, False, "")
            except Exception:
                pass
            return
        try:
            mouse_point = self.plot.vb.mapSceneToView(pos)
        except Exception:
            return
        x = float(mouse_point.x())
        y = float(mouse_point.y())
        value = self._value_at(x, y)
        label = self._format_crosshair_text(x, y, value)
        self._set_last_local_crosshair(x, y, value, label)
        self._update_local_crosshair()
        try:
            self.sigCursorMoved.emit(self, x, y, value, True, label)
        except Exception:
            pass

    def _on_scene_mouse_clicked(self, event):
        try:
            button = event.button()
        except Exception:
            return
        if button != QtCore.Qt.RightButton:
            return
        try:
            scene_pos = event.scenePos()
        except Exception:
            scene_pos = None
        if scene_pos is None:
            return
        try:
            scene_rect = self.plot.sceneBoundingRect()
        except Exception:
            scene_rect = None
        if not (scene_rect and scene_rect.contains(scene_pos)):
            return
        event.accept()
        try:
            view_point = self.plot.vb.mapSceneToView(scene_pos)
        except Exception:
            view_point = None
        if view_point is not None:
            x = float(view_point.x())
            y = float(view_point.y())
            value = self._value_at(x, y)
            label = self._format_crosshair_text(x, y, value)
            self._set_last_local_crosshair(x, y, value, label)
            if self._local_crosshair_enabled:
                self._update_local_crosshair()
        menu = QtWidgets.QMenu()
        act_crosshair = menu.addAction("Show crosshair")
        act_crosshair.setCheckable(True)
        act_crosshair.setChecked(self._local_crosshair_enabled)
        act_crosshair.toggled.connect(self.set_local_crosshair_enabled)

        if self._histogram_menu_getter and self._histogram_menu_setter:
            menu.addSeparator()
            act_hist = menu.addAction("Show histogram")
            act_hist.setCheckable(True)
            try:
                act_hist.setChecked(bool(self._histogram_menu_getter()))
            except Exception:
                act_hist.setChecked(True)
            if self._histogram_menu_enabled_getter is not None:
                try:
                    act_hist.setEnabled(bool(self._histogram_menu_enabled_getter()))
                except Exception:
                    pass
            act_hist.toggled.connect(self._histogram_menu_setter)
        try:
            screen_pos = event.screenPos()
        except Exception:
            screen_pos = None
        if screen_pos is not None:
            try:
                point = QtCore.QPoint(int(screen_pos.x()), int(screen_pos.y()))
            except Exception:
                point = QtGui.QCursor.pos()
        else:
            point = QtGui.QCursor.pos()
        menu.exec_(point)

    def configure_histogram_toggle(self, *, getter=None, setter=None, enabled_getter=None):
        self._histogram_menu_getter = getter
        self._histogram_menu_setter = setter
        self._histogram_menu_enabled_getter = enabled_getter

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

