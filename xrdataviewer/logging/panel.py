from __future__ import annotations

from pathlib import Path

from PySide2 import QtCore, QtWidgets

from app_logging import ActionLogger


class LoggingDockWidget(QtWidgets.QDockWidget):
    """Dockable pane that displays log entries in real time."""

    def __init__(self, logger: ActionLogger, parent=None):
        super().__init__("Action Log", parent)
        self.setObjectName("actionLogDock")
        self._logger = logger
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        self._collapsed = False

        self._view = QtWidgets.QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self._view.setMaximumBlockCount(2000)
        self._view.setPlaceholderText("User actions will be logged here.")
        self.setWidget(self._view)

        self._build_title_bar()

        self.setCollapsed(False)

        clear_action = QtWidgets.QAction("Clear log", self)
        clear_action.triggered.connect(self._clear_log)
        self.addAction(clear_action)
        export_action = QtWidgets.QAction("Export log…", self)
        export_action.triggered.connect(self._export_log)
        self.addAction(export_action)
        self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

        for entry in logger.entries():
            self._append(entry)

        logger.log_added.connect(self._append)
        logger.reset.connect(self._reset)

    def _build_title_bar(self):
        title_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(title_widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self._toggle_btn = QtWidgets.QToolButton()
        self._toggle_btn.setArrowType(QtCore.Qt.DownArrow)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(True)
        self._toggle_btn.toggled.connect(lambda checked: self.setCollapsed(not checked))
        layout.addWidget(self._toggle_btn, 0)

        label = QtWidgets.QLabel("Action Log")
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        layout.addWidget(label, 1)

        self._export_btn = QtWidgets.QToolButton()
        self._export_btn.setText("Export…")
        self._export_btn.clicked.connect(self._export_log)
        layout.addWidget(self._export_btn, 0)

        layout.addStretch(0)
        self.setTitleBarWidget(title_widget)

    def setCollapsed(self, collapsed: bool) -> None:
        collapsed = bool(collapsed)
        widget = self.widget()
        self._collapsed = collapsed
        block = self._toggle_btn.blockSignals(True)
        self._toggle_btn.setChecked(not collapsed)
        self._toggle_btn.blockSignals(block)
        if widget is not None:
            widget.setVisible(not collapsed)
        self._toggle_btn.setArrowType(QtCore.Qt.RightArrow if collapsed else QtCore.Qt.DownArrow)
        if collapsed:
            title_height = self.titleBarWidget().sizeHint().height() if self.titleBarWidget() else 24
            self.setMaximumHeight(title_height + 8)
        else:
            self.setMaximumHeight(16777215)
        self.updateGeometry()

    def _append(self, entry: str):
        self._view.appendPlainText(entry)
        bar = self._view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _reset(self):
        self._view.clear()

    def _clear_log(self):
        self._logger.clear()

    def _export_log(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export action log",
            "action-log.txt",
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            Path(path).write_text(self._logger.to_text())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))
