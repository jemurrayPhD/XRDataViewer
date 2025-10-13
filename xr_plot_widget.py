from __future__ import annotations

import math
import html
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide2 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

FORCE_SOFT_RENDER = False
if FORCE_SOFT_RENDER:
    pg.setConfigOptions(useOpenGL=False, antialias=False)


@dataclass
class LineStyleConfig:
    """Visual customization for 1D line plots."""

    color: QtGui.QColor = field(default_factory=lambda: QtGui.QColor("#4fc3f7"))
    opacity: float = 1.0
    width: float = 2.0
    pen_style: str = "solid"
    curve_mode: str = "linear"  # "linear", "smooth", "step"
    smooth_span: int = 3
    markers: bool = False
    marker_style: str = "o"
    marker_size: int = 7

    def normalized_opacity(self) -> float:
        return max(0.0, min(1.0, float(self.opacity)))

    def effective_color(self) -> QtGui.QColor:
        color = QtGui.QColor(self.color)
        if not color.isValid():
            color = QtGui.QColor("#4fc3f7")
        return color

    def smooth_window(self, length: int) -> int:
        length = int(max(1, length))
        if length < 3:
            return 3
        span = max(1, int(self.smooth_span))
        window = span * 2 + 1
        if window > length:
            window = length if length % 2 else max(3, length - 1)
        if window < 3:
            window = 3
        if window % 2 == 0:
            window += 1
        return window


def clone_line_style(style: Optional[LineStyleConfig]) -> LineStyleConfig:
    src = style or LineStyleConfig()
    return LineStyleConfig(
        color=src.effective_color(),
        opacity=src.opacity,
        width=src.width,
        pen_style=src.pen_style,
        curve_mode=src.curve_mode,
        smooth_span=src.smooth_span,
        markers=src.markers,
        marker_style=src.marker_style,
        marker_size=src.marker_size,
    )


@dataclass
class PlotAnnotationConfig:
    """Configuration describing plot annotations and aesthetics."""

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    colorbar_label: str = ""
    font_family: str = ""
    title_size: int = 14
    axis_size: int = 12
    tick_size: int = 10
    colorbar_size: int = 12
    background: QtGui.QColor = field(default_factory=lambda: QtGui.QColor("#1b1b1b"))
    apply_to_all: bool = False
    legend_visible: bool = False
    legend_entries: List[str] = field(default_factory=list)
    legend_position: str = "top-right"


_LATEX_SYMBOLS = {
    "\\alpha": "α",
    "\\beta": "β",
    "\\gamma": "γ",
    "\\delta": "δ",
    "\\epsilon": "ϵ",
    "\\theta": "θ",
    "\\lambda": "λ",
    "\\mu": "μ",
    "\\nu": "ν",
    "\\pi": "π",
    "\\phi": "φ",
    "\\psi": "ψ",
    "\\omega": "ω",
    "\\times": "×",
    "\\cdot": "·",
    "\\pm": "±",
    "\\geq": "≥",
    "\\leq": "≤",
    "\\infty": "∞",
    "\\sqrt": "√",
    "\\degree": "°",
}


class _SafeFormatDict(dict):
    def __missing__(self, key):  # pragma: no cover - defensive
        return "{" + key + "}"


def _format_with_context(text: str, context: Optional[Dict[str, Any]]) -> str:
    if not text:
        return ""
    if not context:
        return text
    try:
        return str(text).format_map(_SafeFormatDict(context))
    except Exception:
        return str(text)


def _parse_basic_latex(text: str) -> str:
    """Convert a lightweight subset of LaTeX-like syntax to HTML."""

    if text is None:
        return ""

    def parse_segment(segment: str) -> str:
        i = 0
        out_parts: list[str] = []
        length = len(segment)
        while i < length:
            ch = segment[i]
            if ch == "\\":
                if i + 1 < length and segment[i + 1] == "\\":
                    out_parts.append("<br/>")
                    i += 2
                    continue
                if i + 1 < length and segment[i + 1] in ("n", "r"):
                    out_parts.append("<br/>")
                    i += 2
                    continue
                j = i + 1
                while j < length and segment[j].isalpha():
                    j += 1
                command = segment[i:j]
                replacement = _LATEX_SYMBOLS.get(command)
                if replacement is not None:
                    out_parts.append(replacement)
                    i = j
                    continue
                if j < length and segment[j] in "{}":
                    out_parts.append(html.escape(segment[j]))
                    i = j + 1
                    continue
                out_parts.append(html.escape(command.lstrip("\\")))
                i = j
                continue
            if ch in "^_":
                sup = ch == "^"
                i += 1
                if i < length and segment[i] == "{":
                    depth = 1
                    j = i + 1
                    while j < length and depth > 0:
                        if segment[j] == "{":
                            depth += 1
                        elif segment[j] == "}":
                            depth -= 1
                            if depth == 0:
                                break
                        j += 1
                    content = segment[i + 1:j] if j < length else segment[i + 1 :]
                    i = j + 1 if j < length else length
                elif i < length:
                    content = segment[i]
                    i += 1
                else:
                    content = ""
                parsed = parse_segment(content)
                tag = "sup" if sup else "sub"
                out_parts.append(f"<{tag}>{parsed}</{tag}>")
                continue
            if ch == "\n":
                out_parts.append("<br/>")
                i += 1
                continue
            out_parts.append(html.escape(ch))
            i += 1
        return "".join(out_parts)

    return parse_segment(str(text))


def latex_to_html(text: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Public helper to convert a label with optional formatting context."""

    formatted = _format_with_context(text, context)
    return _parse_basic_latex(formatted)


def _ensure_qcolor(value: Any) -> QtGui.QColor:
    if isinstance(value, QtGui.QColor):
        return QtGui.QColor(value)
    if isinstance(value, QtGui.QBrush):
        return QtGui.QColor(value.color())
    if isinstance(value, str):
        return QtGui.QColor(value)
    color = QtGui.QColor()
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        color.setRgb(int(value[0]), int(value[1]), int(value[2]))
    return color if color.isValid() else QtGui.QColor("#1b1b1b")


def _make_font(base: Optional[QtGui.QFont], family: str, size: int) -> QtGui.QFont:
    font = QtGui.QFont(base) if base is not None else QtGui.QFont()
    if family:
        font.setFamily(family)
    if size > 0:
        font.setPointSize(int(size))
    elif base is not None and base.pointSize() > 0:
        font.setPointSize(base.pointSize())
    return font


def plotitem_annotation_state(
    plot: pg.PlotItem,
    colorbar_label: Optional[QtWidgets.QLabel] = None,
) -> PlotAnnotationConfig:
    title = ""
    try:
        title = plot.titleLabel.text
    except Exception:
        title = ""
    stored = getattr(plot, "_annotation_sources", {})
    xlabel = ""
    ylabel = ""
    axis_font_family = ""
    axis_font_size = 12
    tick_font_size = 10
    bottom = plot.getAxis("bottom")
    if bottom is not None:
        try:
            xlabel = bottom.labelText
        except Exception:
            xlabel = ""
        label_item = getattr(bottom, "label", None)
        if label_item is not None:
            try:
                font = label_item.font()
            except Exception:
                font = None
            if font is not None:
                axis_font_family = font.family()
                if font.pointSize() > 0:
                    axis_font_size = font.pointSize()
        try:
            tick_font = bottom.tickFont
        except Exception:
            tick_font = None
        if tick_font is not None and tick_font.pointSize() > 0:
            tick_font_size = tick_font.pointSize()
    left = plot.getAxis("left")
    if left is not None:
        try:
            ylabel = left.labelText
        except Exception:
            ylabel = ""
        if not axis_font_family:
            label_item = getattr(left, "label", None)
            if label_item is not None:
                try:
                    font = label_item.font()
                except Exception:
                    font = None
                if font is not None:
                    axis_font_family = font.family()
                    if font.pointSize() > 0:
                        axis_font_size = font.pointSize()
        try:
            tick_font = left.tickFont
        except Exception:
            tick_font = None
        if tick_font is not None and tick_font.pointSize() > 0:
            tick_font_size = tick_font.pointSize()
    colorbar_text = ""
    colorbar_size = axis_font_size
    if colorbar_label is not None:
        try:
            colorbar_text = colorbar_label.text()
        except Exception:
            colorbar_text = ""
        try:
            cb_font = colorbar_label.font()
        except Exception:
            cb_font = None
        if cb_font is not None and cb_font.pointSize() > 0:
            colorbar_size = cb_font.pointSize()
    try:
        background_brush = plot.getBackgroundBrush()
    except Exception:
        background_brush = None
    config = PlotAnnotationConfig(
        title=str(stored.get("title", title) or ""),
        xlabel=str(stored.get("xlabel", xlabel) or ""),
        ylabel=str(stored.get("ylabel", ylabel) or ""),
        colorbar_label=str(stored.get("colorbar", colorbar_text) or ""),
        font_family=str(axis_font_family or ""),
        axis_size=int(axis_font_size),
        tick_size=int(tick_font_size),
        colorbar_size=int(colorbar_size),
    )
    config.background = _ensure_qcolor(background_brush)
    try:
        title_font = plot.titleLabel.item.font()
        if title_font.pointSize() > 0:
            config.title_size = int(title_font.pointSize())
        if title_font.family() and not config.font_family:
            config.font_family = title_font.family()
    except Exception:
        pass
    legend_config = getattr(plot, "_legend_config", None)
    if isinstance(legend_config, dict):
        config.legend_visible = bool(legend_config.get("visible"))
        entries = legend_config.get("entries")
        if isinstance(entries, (list, tuple)):
            config.legend_entries = [str(e) for e in entries]
        else:
            config.legend_entries = []
        position = legend_config.get("position")
        if isinstance(position, str):
            config.legend_position = position
    return config


def apply_plotitem_annotation(
    plot: pg.PlotItem,
    config: PlotAnnotationConfig,
    *,
    context: Optional[Dict[str, Any]] = None,
    colorbar_label: Optional[QtWidgets.QLabel] = None,
    background_widget: Optional[QtWidgets.QWidget] = None,
    legend_handler: Optional[Callable[[PlotAnnotationConfig], None]] = None,
):
    sources = getattr(plot, "_annotation_sources", {})
    if not isinstance(sources, dict):
        sources = {}
    sources.update(
        {
            "title": config.title,
            "xlabel": config.xlabel,
            "ylabel": config.ylabel,
            "colorbar": config.colorbar_label,
        }
    )
    setattr(plot, "_annotation_sources", sources)
    setattr(
        plot,
        "_legend_config",
        {
            "visible": bool(config.legend_visible),
            "entries": list(config.legend_entries or []),
            "position": str(config.legend_position or "top-right"),
        },
    )

    title_html = latex_to_html(config.title, context)
    xlabel_html = latex_to_html(config.xlabel, context)
    ylabel_html = latex_to_html(config.ylabel, context)
    colorbar_html = latex_to_html(config.colorbar_label, context)

    plot.setTitle(f"<span>{title_html}</span>" if title_html else "")
    try:
        title_item = plot.titleLabel.item
    except Exception:
        title_item = None
    if title_item is not None:
        title_item.setFont(_make_font(title_item.font(), config.font_family, config.title_size))

    for axis_name, html_text in (("bottom", xlabel_html), ("left", ylabel_html)):
        axis = plot.getAxis(axis_name)
        if axis is None:
            continue
        axis.setLabel(text=f"<span>{html_text}</span>" if html_text else "")
        label_item = getattr(axis, "label", None)
        if label_item is not None:
            try:
                current_font = label_item.font()
            except Exception:
                current_font = None
            label_item.setFont(_make_font(current_font, config.font_family, config.axis_size))
        tick_font = _make_font(None, config.font_family, config.tick_size)
        try:
            axis.setTickFont(tick_font)
        except Exception:
            pass

    if colorbar_label is not None:
        colorbar_label.setText(colorbar_html)
        colorbar_label.setVisible(bool(colorbar_html))
        colorbar_label.setAlignment(QtCore.Qt.AlignCenter)
        colorbar_label.setWordWrap(True)
        try:
            base_font = colorbar_label.font()
        except Exception:
            base_font = None
        colorbar_label.setFont(_make_font(base_font, config.font_family, config.colorbar_size))

    background = _ensure_qcolor(config.background)

    try:
        view_box = plot.getViewBox()
    except Exception:
        view_box = None

    if view_box is not None:
        try:
            view_box.setBackgroundColor(background)
        except Exception:
            try:
                view_box.setBackground(background)
            except Exception:
                pass

    if hasattr(plot, "setBackground"):
        try:
            plot.setBackground(background)
        except Exception:
            try:
                plot.getViewWidget().setBackground(background)
            except Exception:
                pass

    if background_widget is not None:
        try:
            if hasattr(background_widget, "setBackground"):
                background_widget.setBackground(background)
            elif hasattr(background_widget, "setBackgroundBrush"):
                background_widget.setBackgroundBrush(background)
            elif hasattr(background_widget, "setBackgroundBrush"):
                background_widget.setBackgroundBrush(background)
            elif hasattr(background_widget, "setBackgroundColor"):
                background_widget.setBackgroundColor(background)
            else:
                raise AttributeError
        except Exception:
            try:
                palette = background_widget.palette()
                palette.setColor(background_widget.backgroundRole(), background)
                background_widget.setPalette(palette)
            except Exception:
                background_widget.setStyleSheet(f"background-color: {background.name()};")

    if callable(legend_handler):
        try:
            legend_handler(config)
        except Exception:
            pass



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
    sigLocalCrosshairToggled = QtCore.Signal(bool)
    sigCursorMoved = QtCore.Signal(object, float, float, object, bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.glw = pg.GraphicsLayoutWidget()
        axis_items = {
            "bottom": ScientificAxisItem("bottom"),
            "left": ScientificAxisItem("left"),
        }
        self.plot = self.glw.addPlot(row=0, col=0, axisItems=axis_items)
        self.plot.invertY(False)
        self.plot.setMenuEnabled(False)
        self.plot.setLabel("left", "Y")
        self.plot.setLabel("bottom", "X")
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)
        self._line_item = pg.PlotDataItem()
        self._line_item.setVisible(False)
        self.plot.addItem(self._line_item)
        self.lut = MyHistogramLUT()
        self.lut.setImageItem(self.img_item)

        self._hist_container = None
        self._block_levels_emit = False
        self._block_view_emit = False
        self._last_data = None
        self._last_rect = None
        self._line_x: Optional[np.ndarray] = None
        self._line_y: Optional[np.ndarray] = None
        self._line_plot_x: Optional[np.ndarray] = None
        self._line_plot_y: Optional[np.ndarray] = None
        self._mode: str = "image"
        self._line_style = LineStyleConfig()

        self._legend_item: Optional[pg.LegendItem] = None
        self._legend_sources: List[Tuple[pg.GraphicsObject, str]] = []

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

        # annotation helpers
        self._colorbar_label_widget: Optional[QtWidgets.QLabel] = None
        self._background_color = QtGui.QColor()
        try:
            brush = self.plot.getBackgroundBrush()
            if isinstance(brush, QtGui.QBrush):
                self._background_color = brush.color()
        except Exception:
            self._background_color = QtGui.QColor("#1b1b1b")

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

    # ---------- annotation helpers ----------
    def annotation_defaults(self) -> PlotAnnotationConfig:
        config = plotitem_annotation_state(self.plot, self._colorbar_label_widget)
        if self._background_color.isValid():
            config.background = QtGui.QColor(self._background_color)
        return config

    def apply_annotation(
        self,
        config: PlotAnnotationConfig,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        apply_plotitem_annotation(
            self.plot,
            config,
            context=context,
            colorbar_label=self._colorbar_label_widget,
            background_widget=self.glw,
            legend_handler=self._apply_legend,
        )
        self._background_color = _ensure_qcolor(config.background)

    def set_colorbar_label(self, text: str, context: Optional[Dict[str, Any]] = None):
        if self._colorbar_label_widget is None:
            return
        html_text = latex_to_html(text, context)
        self._colorbar_label_widget.setText(html_text)
        self._colorbar_label_widget.setVisible(bool(html_text))

    def set_legend_sources(self, items: Sequence[Tuple[object, str]]):
        processed: List[Tuple[object, str]] = []
        for obj, label in items:
            if obj is None:
                continue
            processed.append((obj, str(label)))
        self._legend_sources = processed

    def _ensure_legend(self) -> pg.LegendItem:
        if self._legend_item is None:
            legend = pg.LegendItem(offset=(10, 10))
            legend.setParentItem(self.plot.graphicsItem())
            self._legend_item = legend
        return self._legend_item

    def _apply_legend(self, config: PlotAnnotationConfig):
        if not config.legend_visible or not self._legend_sources:
            if self._legend_item is not None:
                try:
                    self._legend_item.hide()
                except Exception:
                    pass
            return
        legend = self._ensure_legend()
        try:
            legend.clear()
        except Exception:
            pass
        entries = list(config.legend_entries or [])
        if not entries:
            entries = [label for _, label in self._legend_sources]
        for idx, (item, default_label) in enumerate(self._legend_sources):
            label = entries[idx] if idx < len(entries) else default_label
            try:
                if hasattr(item, "setName"):
                    item.setName(label)
            except Exception:
                pass
            try:
                legend.addItem(item, label)
            except Exception:
                proxy = pg.PlotDataItem([0], [0])
                proxy.setPen(pg.mkPen((200, 200, 200)))
                legend.addItem(proxy, label)
        anchor_map = {
            "top-left": ((0, 0), (0, 0)),
            "top-right": ((1, 0), (1, 0)),
            "bottom-left": ((0, 1), (0, 1)),
            "bottom-right": ((1, 1), (1, 1)),
        }
        pos = anchor_map.get(config.legend_position, anchor_map["top-right"])
        try:
            legend.anchor(pos[0], pos[1])
        except Exception:
            pass
        try:
            legend.show()
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
        if self._mode == "line":
            xs = self._line_plot_x if self._line_plot_x is not None else self._line_x
            ys = self._line_plot_y if self._line_plot_y is not None else self._line_y
            if xs is None or ys is None or xs.size == 0:
                return
            rect = self._line_data_rect(xs, ys)
            if rect is None:
                return
            try:
                self.plot.vb.setXRange(rect.left(), rect.right(), padding=0.05)
                self.plot.vb.setYRange(rect.top(), rect.bottom(), padding=0.05)
            except Exception:
                pass
            return
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
                container = QtWidgets.QWidget()
                container.setObjectName("HistogramContainer")
                layout = QtWidgets.QVBoxLayout(container)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(4)
                label = QtWidgets.QLabel("")
                label.setAlignment(QtCore.Qt.AlignCenter)
                label.setWordWrap(True)
                label.setVisible(False)
                layout.addWidget(label, 0)
                glw = pg.GraphicsLayoutWidget()
                glw.addItem(self.lut, row=0, col=0)
                glw.setObjectName("HistogramLUTContainer")
                glw.setSizePolicy(
                    QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
                )
                layout.addWidget(glw, 1)
                self._colorbar_label_widget = label
                self._hist_container = container
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
    def image_item(self) -> pg.ImageItem:
        return self.img_item

    def line_item(self) -> pg.PlotDataItem:
        return self._line_item

    def set_image(self, Z: np.ndarray, autorange: bool = True, rect=None):
        Z = np.asarray(Z, float, order="C")
        self.img_item.setImage(Z, autoLevels=autorange)
        try:
            from PySide2.QtCore import QRectF
            if rect is None:
                Ny, Nx = Z.shape
                rect = QRectF(0.0, 0.0, float(Nx), float(Ny))
            self.img_item.setRect(rect)
        except Exception:
            rect = None
        self._last_data = np.asarray(Z, float)
        try:
            self._last_rect = rect
        except Exception:
            self._last_rect = None
        self._line_x = None
        self._line_y = None
        self._mode = "image"
        try:
            self.img_item.setVisible(True)
            self._line_item.setVisible(False)
        except Exception:
            pass
        container = self.histogram_widget()
        if container is not None:
            container.setVisible(True)
        try:
            self.lut.setVisible(True)
        except Exception:
            pass
        if autorange:
            try:
                self.plot.enableAutoRange(x=True, y=True)
            except Exception:
                pass

    def set_rectilinear(self, x1: np.ndarray, y1: np.ndarray, Z: np.ndarray, autorange: bool = True):
        Zu, xr = self._resample_rectilinear(x1, y1, Z, return_rect=True)
        self.set_image(Zu, autorange, rect=xr)

    def set_warped(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, autorange: bool = True):
        Zu, xr = self._resample_warped(X, Y, Z, return_rect=True)
        self.set_image(Zu, autorange, rect=xr)

    def set_line(self, x: np.ndarray, y: np.ndarray, autorange: bool = True):
        xs = np.asarray(x, float)
        ys = np.asarray(y, float)
        if xs.ndim != 1:
            xs = xs.ravel()
        if ys.ndim != 1:
            ys = ys.ravel()
        if xs.size == 0 and ys.size:
            xs = np.arange(ys.size, dtype=float)
        if ys.size != xs.size:
            if ys.size == 0:
                xs = np.array([], dtype=float)
            else:
                xs = np.linspace(0.0, float(ys.size - 1), ys.size)
        self._line_x = xs.astype(float, copy=False)
        self._line_y = ys.astype(float, copy=False)
        self._last_data = np.array(self._line_y, copy=True)
        self._last_rect = None
        self._mode = "line"
        self._render_line_data(autorange=autorange)

    def line_style(self) -> LineStyleConfig:
        return clone_line_style(self._line_style)

    def set_line_style(self, style: LineStyleConfig, *, refresh: bool = True):
        self._line_style = clone_line_style(style)
        if refresh and self._mode == "line":
            self._render_line_data(autorange=False)

    def _render_line_data(self, *, autorange: bool):
        if self._line_x is None or self._line_y is None:
            return
        style = self._line_style or LineStyleConfig()
        xs = np.asarray(self._line_x, float)
        ys = np.asarray(self._line_y, float)
        x_plot = xs
        y_plot = ys
        step_mode = style.curve_mode == "step"
        if style.curve_mode == "smooth" and ys.size >= 3:
            try:
                window = style.smooth_window(ys.size)
                y_plot = pg.functions.smooth(ys, window=window)
            except Exception:
                y_plot = ys

        color = style.effective_color()
        color.setAlphaF(style.normalized_opacity())
        width = max(0.1, float(style.width))
        pen = pg.mkPen(color, width=width)
        try:
            pen_style = {
                "solid": QtCore.Qt.SolidLine,
                "dashed": QtCore.Qt.DashLine,
                "dotted": QtCore.Qt.DotLine,
                "dashdot": QtCore.Qt.DashDotLine,
            }.get(style.pen_style, QtCore.Qt.SolidLine)
            pen.setStyle(pen_style)
        except Exception:
            pass

        kwargs: Dict[str, object] = {}
        if step_mode:
            kwargs["stepMode"] = True
            x_plot = self._step_edges(xs)

        symbol = None
        marker_pen: Optional[QtGui.QPen] = None
        marker_brush: Optional[QtGui.QBrush] = None
        marker_size: Optional[int] = None
        symbol_kwargs: Dict[str, object] = {}
        if style.markers and not step_mode:
            key = str(style.marker_style).strip().lower()
            symbol_map = {
                "o": "o",
                "circle": "o",
                "●": "o",
                "•": "o",
                "s": "s",
                "square": "s",
                "□": "s",
                "t": "t",
                "triangle": "t",
                "triangleup": "t",
                "^": "t",
                "d": "d",
                "diamond": "d",
                "+": "+",
                "plus": "+",
                "x": "x",
                "cross": "x",
            }
            symbol = symbol_map.get(key, "o")
            marker_color = QtGui.QColor(color)
            marker_pen = QtGui.QPen(marker_color)
            marker_pen.setWidthF(max(1.0, width * 0.75))
            marker_brush = QtGui.QBrush(marker_color)
            marker_size = max(1, int(style.marker_size))
            symbol_kwargs["symbol"] = symbol
            symbol_kwargs["symbolBrush"] = marker_brush
            symbol_kwargs["symbolPen"] = marker_pen
            symbol_kwargs["symbolSize"] = marker_size
        else:
            symbol_kwargs["symbol"] = None

        try:
            self._line_item.setData(x_plot, y_plot, pen=pen, **kwargs, **symbol_kwargs)
            self._line_item.setVisible(True)
            self.img_item.setVisible(False)
            if symbol_kwargs.get("symbol"):
                try:
                    self._line_item.setSymbol(symbol_kwargs.get("symbol"))
                    if marker_brush is not None:
                        self._line_item.setSymbolBrush(marker_brush)
                    if marker_pen is not None:
                        self._line_item.setSymbolPen(marker_pen)
                    if marker_size is not None:
                        self._line_item.setSymbolSize(marker_size)
                except Exception:
                    pass
            else:
                try:
                    self._line_item.setSymbol(None)
                except Exception:
                    pass
        except Exception:
            pass

        self._line_plot_x = np.array(x_plot, copy=True)
        self._line_plot_y = np.array(y_plot, copy=True)
        container = self.histogram_widget()
        if container is not None:
            container.setVisible(False)
        try:
            self.lut.setVisible(False)
        except Exception:
            pass
        if autorange:
            self.auto_view_range()

    def _step_edges(self, xs: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, float)
        n = xs.size
        if n == 0:
            return xs
        if n == 1:
            x0 = float(xs[0])
            return np.array([x0 - 0.5, x0 + 0.5], dtype=float)

        diffs = np.diff(xs)
        if np.all(diffs > 0):
            edges = np.empty(n + 1, dtype=float)
            edges[1:-1] = xs[:-1] + diffs / 2.0
            edges[0] = xs[0] - diffs[0] / 2.0
            edges[-1] = xs[-1] + diffs[-1] / 2.0
            return edges

        # Fallback for unsorted or repeated x values: pad with duplicates.
        edges = np.empty(n + 1, dtype=float)
        edges[:-1] = xs
        edges[-1] = xs[-1]
        return edges

    def _line_data_rect(self, xs: np.ndarray, ys: np.ndarray) -> Optional[QtCore.QRectF]:
        xs = np.asarray(xs, float).reshape(-1)
        ys = np.asarray(ys, float).reshape(-1)
        if xs.size == 0 or ys.size == 0:
            return None
        if ys.size != xs.size:
            m = min(xs.size, ys.size)
            xs = xs[:m]
            ys = ys[:m]
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not mask.any():
            return None
        x_vals = xs[mask]
        y_vals = ys[mask]
        x0 = float(np.nanmin(x_vals))
        x1 = float(np.nanmax(x_vals))
        y0 = float(np.nanmin(y_vals))
        y1 = float(np.nanmax(y_vals))
        if not np.isfinite(x0) or not np.isfinite(x1):
            x0, x1 = 0.0, float(xs.size - 1 if xs.size > 1 else 1.0)
        if not np.isfinite(y0) or not np.isfinite(y1):
            y0, y1 = 0.0, 1.0
        if x0 == x1:
            pad = 0.5 if xs.size <= 1 else max(1e-6, abs(x0) * 0.01)
            x0 -= pad
            x1 += pad
        if y0 == y1:
            pad = 0.5 if ys.size <= 1 else max(1e-6, abs(y0) * 0.05)
            y0 -= pad
            y1 += pad
        return QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

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
        if self._mode == "line":
            xs = self._line_x
            ys = self._line_y
            if xs is None or ys is None or xs.size == 0:
                return None
            try:
                diffs = np.abs(xs - float(x))
            except Exception:
                return None
            try:
                idx = int(np.nanargmin(diffs))
            except Exception:
                idx = 0
            idx = max(0, min(idx, ys.size - 1))
            try:
                return float(ys[idx])
            except Exception:
                return ys[idx]

        data = self._last_data
        rect = self._last_rect
        if data is None or rect is None:
            return None
        try:
            x0 = float(rect.left())
            y0 = float(rect.top())
            w = float(rect.width())
            h = float(rect.height())
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
        try:
            self.sigLocalCrosshairToggled.emit(self._local_crosshair_enabled)
        except Exception:
            pass

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

    def local_crosshair_enabled(self) -> bool:
        return bool(self._local_crosshair_enabled)

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

