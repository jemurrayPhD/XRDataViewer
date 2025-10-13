from __future__ import annotations

from typing import Optional

from PySide2 import QtCore, QtGui, QtWidgets

from xr_plot_widget import PlotAnnotationConfig


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
        )
        self._result = config
        super().accept()
