from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide2")

import pyqtgraph as pg
from PySide2 import QtCore, QtGui, QtWidgets

from app_logging import ACTION_LOGGER, log_action

from .datasets import DatasetsPane, SliceDataTab
from .interactive import InteractiveBridgeServer, InteractiveProcessingTab
from .logging.panel import LoggingDockWidget
from .preferences import PreferencesDialog, PreferencesManager
from .processing import ProcessingDockContainer, ProcessingDockWidget, ProcessingManager
from .views.multiview import MultiViewGrid
from .views.overlay import OverlayView
from .views.sequential import SequentialView


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Multi-Viewer")

        main = QtWidgets.QSplitter()
        self.setCentralWidget(main)

        self.preferences = PreferencesManager()
        self.processing_manager = ProcessingManager()

        self._build_menus()

        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_splitter.setChildrenCollapsible(False)
        left_splitter.setHandleWidth(8)
        self.datasets = DatasetsPane()
        self.bridge_server = InteractiveBridgeServer(self.datasets, self)
        self.bridge_server.start()
        left_splitter.addWidget(self.datasets)
        self.processing_dock = ProcessingDockWidget(self.processing_manager)
        self.processing_panel = ProcessingDockContainer("Processing Pipelines", self.processing_dock)
        left_splitter.addWidget(self.processing_panel)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 1)
        main.addWidget(left_splitter)

        QtCore.QTimer.singleShot(0, lambda: left_splitter.setSizes([700, 400]))

        self.tabs = QtWidgets.QTabWidget()
        main.addWidget(self.tabs)
        main.setStretchFactor(1, 1)

        self.tab_multiview = MultiViewGrid(self.processing_manager, self.preferences)
        self.tabs.addTab(self.tab_multiview, "MultiView")
        self.tab_sequential = SequentialView(self.processing_manager, self.preferences)
        self.tabs.addTab(self.tab_sequential, "Sequential View")
        self.tab_slice = SliceDataTab(self.datasets)
        self.tabs.addTab(self.tab_slice, "Slice Data")
        self.tab_overlay = OverlayView(self.processing_manager, self.preferences)
        self.tabs.addTab(self.tab_overlay, "Overlay")
        self.tab_overlay.set_processing_manager(self.processing_manager)
        self.tab_interactive = InteractiveProcessingTab(self.datasets, self.bridge_server)
        self.tabs.addTab(self.tab_interactive, "Interactive Processing")

        self.log_dock = LoggingDockWidget(ACTION_LOGGER, self)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.setFloating(False)
        self.log_dock.resize(800, 200)

        self.resize(1500, 900)

    def closeEvent(self, event: QtGui.QCloseEvent):  # type: ignore[override]
        try:
            self.tab_interactive.shutdown()
        except Exception:
            pass
        try:
            self.bridge_server.stop()
        except Exception:
            pass
        super().closeEvent(event)

    def _build_menus(self):
        menubar = self.menuBar()
        prefs_menu = menubar.addMenu("Preferences")

        act_edit = prefs_menu.addAction("Edit preferences…")
        act_edit.triggered.connect(self._edit_preferences)

        prefs_menu.addSeparator()

        act_load = prefs_menu.addAction("Load preferences…")
        act_load.triggered.connect(self._load_preferences)

        act_save = prefs_menu.addAction("Save preferences…")
        act_save.triggered.connect(self._save_preferences)

    def _edit_preferences(self):
        dialog = PreferencesDialog(self.preferences, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            data = dialog.result_data()
            self.preferences.update(data)
            log_action("Updated preferences")

    def _load_preferences(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load preferences",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            self.preferences.load_from_file(Path(path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            return
        log_action(f"Loaded preferences from {path}")

    def _save_preferences(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save preferences",
            "preferences.json",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            self.preferences.save_to_file(Path(path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))
            return
        log_action(f"Saved preferences to {path}")


def main() -> None:
    app = QtWidgets.QApplication([])
    pg.setConfigOptions(imageAxisOrder="row-major")
    window = MainWindow()
    window.show()
    app.exec_()
