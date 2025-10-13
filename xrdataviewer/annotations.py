from __future__ import annotations

from typing import Optional

from PySide2 import QtCore, QtGui, QtWidgets

from xr_plot_widget import LineStyleConfig, PlotAnnotationConfig


class ColorButton(QtWidgets.QPushButton):
    colorChanged = QtCore.Signal(QtGui.QColor)

    def __init__(self, color: object = None, parent=None):
        super().__init__(parent)
        self._color = self._coerce_color(color)
        if not self._color.isValid():
            self._color = QtGui.QColor("#1b1b1b")
        self.setFixedWidth(60)
        self.clicked.connect(self._choose_color)
        self._update_style()

    def color(self) -> QtGui.QColor:
        return QtGui.QColor(self._color)

    def setColor(self, color: object):
        qcolor = self._coerce_color(color)
        if not qcolor.isValid() or qcolor == self._color:
            return
        self._color = qcolor
        self._update_style()
        self.colorChanged.emit(QtGui.QColor(self._color))

    def _coerce_color(self, value: object) -> QtGui.QColor:
        if isinstance(value, QtGui.QColor):
            return QtGui.QColor(value)
        if isinstance(value, QtGui.QBrush):
            return QtGui.QColor(value.color())
        if isinstance(value, str):
            return QtGui.QColor(value)
        if isinstance(value, tuple) or isinstance(value, list):
            if len(value) == 3:
                r, g, b = value
                return QtGui.QColor(int(r), int(g), int(b))
            if len(value) >= 4:
                r, g, b, a = value[:4]
                return QtGui.QColor(int(r), int(g), int(b), int(a))
        if isinstance(value, (int, float)):
            c = int(value)
            return QtGui.QColor(c)
        return QtGui.QColor(value) if value is not None else QtGui.QColor()

    def _choose_color(self):
        dialog = QtWidgets.QColorDialog(self._color, self)
        dialog.setOption(QtWidgets.QColorDialog.DontUseNativeDialog, True)
        dialog.setOption(QtWidgets.QColorDialog.ShowAlphaChannel, True)
        dialog.setWindowTitle("Select color")
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            chosen = dialog.selectedColor()
            if chosen.isValid():
                self.setColor(chosen)

    def _update_style(self):
        self.setStyleSheet(
            "QPushButton { border: 1px solid #888; border-radius: 3px; background-color: %s; }"
            % self._color.name()
        )

class PlotAnnotationDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent=None,
        *,
        initial: Optional[PlotAnnotationConfig] = None,
        allow_apply_all: bool = True,
        template_hint: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Set plot annotations")
        self._result: Optional[PlotAnnotationConfig] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.edit_title = QtWidgets.QLineEdit(initial.title if initial else "")
        form.addRow("Plot title", self.edit_title)
        self.edit_xlabel = QtWidgets.QLineEdit(initial.xlabel if initial else "")
        form.addRow("X-axis label", self.edit_xlabel)
        self.edit_ylabel = QtWidgets.QLineEdit(initial.ylabel if initial else "")
        form.addRow("Y-axis label", self.edit_ylabel)
        self.edit_colorbar = QtWidgets.QLineEdit(initial.colorbar_label if initial else "")
        form.addRow("Colorbar label", self.edit_colorbar)
        layout.addLayout(form)

        aesthetics = QtWidgets.QGroupBox("Aesthetics")
        grid = QtWidgets.QGridLayout(aesthetics)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(6)
        grid.addWidget(QtWidgets.QLabel("Font family"), 0, 0)
        self.font_combo = QtWidgets.QFontComboBox()
        if initial and initial.font_family:
            self.font_combo.setCurrentFont(QtGui.QFont(initial.font_family))
        grid.addWidget(self.font_combo, 0, 1, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Title size"), 1, 0)
        self.spin_title = QtWidgets.QSpinBox()
        self.spin_title.setRange(6, 72)
        self.spin_title.setValue(initial.title_size if initial else 14)
        grid.addWidget(self.spin_title, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Axis label size"), 2, 0)
        self.spin_axis = QtWidgets.QSpinBox()
        self.spin_axis.setRange(6, 60)
        self.spin_axis.setValue(initial.axis_size if initial else 12)
        grid.addWidget(self.spin_axis, 2, 1)

        grid.addWidget(QtWidgets.QLabel("Tick size"), 3, 0)
        self.spin_tick = QtWidgets.QSpinBox()
        self.spin_tick.setRange(6, 48)
        self.spin_tick.setValue(initial.tick_size if initial else 10)
        grid.addWidget(self.spin_tick, 3, 1)

        grid.addWidget(QtWidgets.QLabel("Colorbar size"), 4, 0)
        self.spin_colorbar = QtWidgets.QSpinBox()
        self.spin_colorbar.setRange(6, 60)
        self.spin_colorbar.setValue(initial.colorbar_size if initial else 12)
        grid.addWidget(self.spin_colorbar, 4, 1)

        grid.addWidget(QtWidgets.QLabel("Background"), 5, 0)
        self.btn_color = ColorButton(initial.background if initial else QtGui.QColor("#1b1b1b"))
        grid.addWidget(self.btn_color, 5, 1)
        grid.setColumnStretch(2, 1)
        layout.addWidget(aesthetics)

        legend_box = QtWidgets.QGroupBox("Legend")
        legend_layout = QtWidgets.QGridLayout(legend_box)
        legend_layout.setContentsMargins(8, 8, 8, 8)
        legend_layout.setSpacing(6)
        self.chk_legend = QtWidgets.QCheckBox("Show legend")
        self.chk_legend.setChecked(bool(initial.legend_visible) if initial else False)
        legend_layout.addWidget(self.chk_legend, 0, 0, 1, 2)
        legend_layout.addWidget(QtWidgets.QLabel("Position"), 1, 0)
        self.cmb_legend_position = QtWidgets.QComboBox()
        self.cmb_legend_position.addItems(["top-right", "top-left", "bottom-right", "bottom-left"])
        pos = (initial.legend_position if initial else "top-right") or "top-right"
        idx = self.cmb_legend_position.findText(pos)
        self.cmb_legend_position.setCurrentIndex(max(0, idx))
        legend_layout.addWidget(self.cmb_legend_position, 1, 1)
        legend_layout.addWidget(QtWidgets.QLabel("Entries (one per line)"), 2, 0, 1, 2)
        self.edit_legend_entries = QtWidgets.QPlainTextEdit()
        entries = "\n".join(initial.legend_entries) if initial and initial.legend_entries else ""
        self.edit_legend_entries.setPlainText(entries)
        self.edit_legend_entries.setPlaceholderText("Layer A\nLayer B")
        legend_layout.addWidget(self.edit_legend_entries, 3, 0, 1, 2)
        self.chk_legend.toggled.connect(self._update_legend_controls)
        self._update_legend_controls(self.chk_legend.isChecked())
        layout.addWidget(legend_box)

        if allow_apply_all:
            self.chk_apply_all = QtWidgets.QCheckBox("Apply to all plots in this tab")
            self.chk_apply_all.setChecked(bool(initial.apply_to_all) if initial else False)
            layout.addWidget(self.chk_apply_all)
        else:
            self.chk_apply_all = QtWidgets.QCheckBox()
            self.chk_apply_all.hide()

        if template_hint:
            hint = QtWidgets.QLabel(template_hint)
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #555;")
            layout.addWidget(hint)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def annotation_config(self) -> Optional[PlotAnnotationConfig]:
        return self._result

    def _update_legend_controls(self, enabled: bool):
        self.cmb_legend_position.setEnabled(enabled)
        self.edit_legend_entries.setEnabled(enabled)

    def accept(self):
        font = self.font_combo.currentFont()
        config = PlotAnnotationConfig(
            title=self.edit_title.text(),
            xlabel=self.edit_xlabel.text(),
            ylabel=self.edit_ylabel.text(),
            colorbar_label=self.edit_colorbar.text(),
            font_family=font.family(),
            title_size=self.spin_title.value(),
            axis_size=self.spin_axis.value(),
            tick_size=self.spin_tick.value(),
            colorbar_size=self.spin_colorbar.value(),
            background=self.btn_color.color(),
            apply_to_all=self.chk_apply_all.isChecked(),
            legend_visible=self.chk_legend.isChecked(),
            legend_entries=[line.strip() for line in self.edit_legend_entries.toPlainText().splitlines() if line.strip()],
            legend_position=self.cmb_legend_position.currentText(),
        )
        self._result = config
        super().accept()


class LineStyleDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, initial: Optional[LineStyleConfig] = None):
        super().__init__(parent)
        self.setWindowTitle("Configure line style")
        self._result: Optional[LineStyleConfig] = None
        initial = initial or LineStyleConfig()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)

        self.btn_color = ColorButton(initial.color)
        form.addRow("Line color", self.btn_color)

        self.spin_width = QtWidgets.QDoubleSpinBox()
        self.spin_width.setRange(0.1, 20.0)
        self.spin_width.setDecimals(2)
        self.spin_width.setSingleStep(0.1)
        self.spin_width.setValue(max(0.1, float(initial.width)))
        form.addRow("Line width", self.spin_width)

        self.cmb_pen_style = QtWidgets.QComboBox()
        self.cmb_pen_style.addItem("Solid", "solid")
        self.cmb_pen_style.addItem("Dashed", "dashed")
        self.cmb_pen_style.addItem("Dotted", "dotted")
        self.cmb_pen_style.addItem("Dash-dot", "dashdot")
        idx = self.cmb_pen_style.findData(initial.pen_style)
        self.cmb_pen_style.setCurrentIndex(max(0, idx))
        form.addRow("Line pattern", self.cmb_pen_style)

        self.cmb_curve_mode = QtWidgets.QComboBox()
        self.cmb_curve_mode.addItem("Straight (piecewise)", "linear")
        self.cmb_curve_mode.addItem("Smooth curve", "smooth")
        self.cmb_curve_mode.addItem("Step (staircase)", "step")
        idx = self.cmb_curve_mode.findData(initial.curve_mode)
        self.cmb_curve_mode.setCurrentIndex(max(0, idx))
        form.addRow("Curve mode", self.cmb_curve_mode)

        self._smooth_row = QtWidgets.QWidget()
        smooth_layout = QtWidgets.QHBoxLayout(self._smooth_row)
        smooth_layout.setContentsMargins(0, 0, 0, 0)
        smooth_layout.setSpacing(6)
        smooth_layout.addWidget(QtWidgets.QLabel("Smoothing strength"))
        self.spin_smooth = QtWidgets.QSpinBox()
        self.spin_smooth.setRange(1, 20)
        self.spin_smooth.setValue(max(1, int(initial.smooth_span)))
        smooth_layout.addWidget(self.spin_smooth)
        smooth_layout.addStretch(1)
        form.addRow("", self._smooth_row)

        opacity_row = QtWidgets.QHBoxLayout()
        opacity_row.setContentsMargins(0, 0, 0, 0)
        opacity_row.setSpacing(6)
        self.sld_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_opacity.setRange(0, 100)
        self.sld_opacity.setValue(int(round(initial.normalized_opacity() * 100)))
        opacity_row.addWidget(self.sld_opacity, 1)
        self.lbl_opacity = QtWidgets.QLabel(f"{int(round(initial.normalized_opacity() * 100))}%")
        opacity_row.addWidget(self.lbl_opacity)
        form.addRow("Opacity", opacity_row)

        self.chk_markers = QtWidgets.QCheckBox("Show point markers")
        self.chk_markers.setChecked(bool(initial.markers))
        form.addRow(self.chk_markers)

        marker_row = QtWidgets.QHBoxLayout()
        marker_row.setContentsMargins(0, 0, 0, 0)
        marker_row.setSpacing(6)
        self.cmb_marker_style = QtWidgets.QComboBox()
        self.cmb_marker_style.addItem("Circle", "o")
        self.cmb_marker_style.addItem("Square", "s")
        self.cmb_marker_style.addItem("Triangle", "t")
        self.cmb_marker_style.addItem("Diamond", "d")
        self.cmb_marker_style.addItem("Plus", "+")
        self.cmb_marker_style.addItem("Cross", "x")
        idx = self.cmb_marker_style.findData(initial.marker_style)
        self.cmb_marker_style.setCurrentIndex(max(0, idx))
        marker_row.addWidget(self.cmb_marker_style, 1)
        self.spin_marker_size = QtWidgets.QSpinBox()
        self.spin_marker_size.setRange(2, 48)
        self.spin_marker_size.setValue(max(2, int(initial.marker_size)))
        marker_row.addWidget(QtWidgets.QLabel("Size"))
        marker_row.addWidget(self.spin_marker_size)
        form.addRow("Marker style", marker_row)

        layout.addLayout(form)

        self.cmb_curve_mode.currentIndexChanged.connect(self._update_curve_controls)
        self.chk_markers.toggled.connect(self._update_marker_controls)
        self.sld_opacity.valueChanged.connect(self._update_opacity_label)

        self._update_curve_controls()
        self._update_marker_controls(self.chk_markers.isChecked())
        self._update_opacity_label(self.sld_opacity.value())

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_curve_controls(self):
        mode = self.cmb_curve_mode.currentData()
        self._smooth_row.setVisible(mode == "smooth")

    def _update_marker_controls(self, checked: bool):
        self.cmb_marker_style.setEnabled(checked)
        self.spin_marker_size.setEnabled(checked)

    def _update_opacity_label(self, value: int):
        self.lbl_opacity.setText(f"{int(value)}%")

    def line_style(self) -> Optional[LineStyleConfig]:
        return self._result

    def accept(self):
        config = LineStyleConfig(
            color=self.btn_color.color(),
            opacity=self.sld_opacity.value() / 100.0,
            width=self.spin_width.value(),
            pen_style=self.cmb_pen_style.currentData(),
            curve_mode=self.cmb_curve_mode.currentData(),
            smooth_span=self.spin_smooth.value(),
            markers=self.chk_markers.isChecked(),
            marker_style=self.cmb_marker_style.currentData(),
            marker_size=self.spin_marker_size.value(),
        )
        self._result = config
        super().accept()
